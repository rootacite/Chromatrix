#!/usr/bin/env python3
"""GPU probe: does `[training].val_data_dir` really replace the split, and can it leak into training?

Run from the repo root in the conda env `environment.yml` names:

    conda activate axl
    python test/probe_val_dir_gpu.py --steps 50 --train-images 10 --val-images 4

## The proposition

With `val_data_dir` set to a directory of its own and `val_split_percent` left at its default,
all four of these hold:

  P1  the validation pool is exactly that directory's images, and **every** training image stays in
      training (the split no longer takes any draw out of the epoch);
  P2  both validation passes score images from that directory only;
  P3  no training forward ever carries an image from that directory;
  P4  the three validation scalars are logged at the cadence, and the fixed pass scores the same
      images at every cadence step.

Refuted by any one of: a training forward whose `image_path` lies under the directory; a pass
forward outside it; `val_image_count` differing from the directory's image count; `total_samples`
reduced by `val_split_percent`; a cadence step with a missing or extra point.

## Arms (one variable apart)

| arm | `val_data_dir` | `val_split_percent` | steps | what it is for |
| --- | --- | --- | --- | --- |
| A1 treatment | the `val` copy (4 images) | 10 (left on, to be ignored) | `--steps` (50) on, and the same 50 with the pass off | P1–P4 |
| A2 integrity | the **train** copy (10 images) | 10 | `--short-steps` (6) | must be *reported as leaking* |
| A3 split control | `""` | 10 | `--short-steps` (6) | the factor removed: a pass pool drawn from the training folder, with draws held out |

A2 is the instrument's own control: it makes the two pools the same folder on purpose, so the leak
detector has to fire. Without it, "no overlap" could just as well mean a detector that never fires.
A3 keeps the same training images and turns the custom directory off, so its pass pool and its epoch
length are the `val_split_percent` ones — the contrast that shows A1's numbers come from the new key
rather than from "some validation ran".

## The deciding measurement

`image_path`, the identity of an image, for every forward on the real path: `StepInstrument` wraps
`loop.build_group_inputs` (the single junction both training steps and validation passes go through)
and records the batch's own `image_path` values, split by whether a pass is running. The verdict is
counted overlap: `|train_saw ∩ val_dir|` (must be 0 in A1, must be ≥ 1 in A2), `|val_saw − val_dir|`
(must be 0 in A1), the per-tag point counts, and the dataset counters (`val_image_count`,
`total_samples`) against the copy's file lists. A caption cannot identify an image, so the probe
never compares prompts.

## Prediction (written before the run)

A1: 10 distinct training paths, all of them the train copy's; 4 distinct pass paths, all of them the
val copy's; overlap 0 both ways; 10 points in each of `Val/Loss` / `Val/Avg_Loss` / `Val/Fixed_Loss`
at steps 1, 6, …, 46; one single distinct fixed sample; the random pass moving between steps (4
pooled images against `val_sample_count = 3`); `total_samples` 10 against A3's 9; and the pass-off /
pass-on pair bit-identical, since the pass rides the custom directory without perturbing the step.
A2: overlap ≥ 1. A3: its pass pool is inside the training folder, no pass path under the val copy,
and its held-out images absent from its training forwards.

## Cost, and what this does not cover

One model load, one cache warm, then `2 × --steps` + `2 × --short-steps` optimizer steps at
`--batch-size 1` / `--resolution 512` with the LoRA dims below (a 16 GB card is not close to full).
Only the in-process loop is exercised: the probe drives `loop.train_one_epoch`, not `main.py`, so the
*run log line* and `state.json` are not observed here; the four propositions are about the dataset,
the passes and the forwards, and it says nothing about sampling, resume or the Dashboard.

Nothing in `config.toml` or the configured dataset is written: images are copied into the scratch
directory first and every cache belongs to that copy. Pass `--clean` to delete it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
# A script's sys.path[0] is its own directory, which is what makes the sibling probe imports work;
# the repo root is what the trainer's package imports and `config.toml` need.
sys.path.insert(0, str(REPO_ROOT))

from probe_val_loss_gpu import (  # noqa: E402 - the paths above are what make these imports work
    HOOK_MARKER,
    StateSnapshot,
    StepInstrument,
    bit_identical,
    configured_train_dirs,
    launcher_alloc_conf,
    maybe_reexec_under_hook,
    measure,
    memory_snapshot,
    prepare_artifacts,
)
from verify_mask_pipeline import (  # noqa: E402
    Report,
    base_config,
    guard_environment,
    live_training_runs,
)

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
TIER = "val-dir"
SERIES_TAG = "Train/Loss"
VAL_TAGS = ("Val/Loss", "Val/Avg_Loss", "Val/Fixed_Loss")
PREDICTION = (
    "A custom val_data_dir replaces the split: the validation pool is exactly the directory's "
    "images, every training image keeps its draws, both passes score that directory only, no "
    "training forward carries one of its images, and the three scalars appear at the cadence. The "
    "integrity arm (the directory pointed at the training folder) must be reported as leaking."
)


# --------------------------------------------------------------------------------------
# the two image subsets, copied out of the configured dataset (read-only on the source)
# --------------------------------------------------------------------------------------


def _images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXT
        and not path.name.lower().endswith(".mask.png")
    )


def copy_subset(source: Path, dest: Path, names: Iterable[str]) -> list[Path]:
    """Copy the named images (and their captions) from [source] into [dest]."""
    dest.mkdir(parents=True, exist_ok=True)
    placed: list[Path] = []
    for name in names:
        path = source / name
        if not path.is_file():
            continue
        target = dest / name
        if not target.exists():
            shutil.copy2(path, target)
            sidecar = path.with_suffix(".txt")
            if sidecar.is_file():
                shutil.copy2(sidecar, target.with_suffix(".txt"))
        placed.append(target)
    return placed


def prepare_subsets(work: Path, train_images: int, val_images: int) -> dict[str, Any]:
    """A train copy and a val copy, taken from the configured folders (two of them, when present).

    With two configured folders the train subset comes from the first and the val subset from the
    second — two real datasets, which is what a user pointing `val_data_dir` at a held-back folder
    has. With one folder both subsets come from it, on disjoint file names.
    """
    sources = configured_train_dirs()
    if not sources:
        raise SystemExit(
            "config.toml names no existing training folder ([[environment.train_data]] / "
            "train_data_dir); the probe needs a dataset to copy"
        )
    train_source = sources[0]
    val_source = sources[1] if len(sources) > 1 else sources[0]
    train_names = [path.name for path in _images(train_source)]
    if len(train_names) < max(4, train_images):
        raise SystemExit(f"{train_source} holds only {len(train_names)} image(s)")
    train_taken = train_names[:train_images]
    val_names = [name for name in (path.name for path in _images(val_source)) if name not in set(train_taken)]
    if len(val_names) < max(2, val_images):
        raise SystemExit(f"{val_source} holds only {len(val_names)} usable image(s) outside the train subset")
    if not train_source.is_dir() or not val_source.is_dir():
        raise SystemExit("a configured training folder is not a directory")

    train_dir = work / "train"
    val_dir = work / "val"
    train_paths = copy_subset(train_source, train_dir, train_taken)
    val_paths = copy_subset(val_source, val_dir, val_names[:val_images])
    return {
        "train_dir": train_dir,
        "val_dir": val_dir,
        "train_paths": {str(path) for path in train_paths},
        "val_paths": {str(path) for path in val_paths},
        "sources": [str(train_source), str(val_source)],
        "same_folder": train_source == val_source,
    }


# --------------------------------------------------------------------------------------
# the measured configuration and pipeline
# --------------------------------------------------------------------------------------


def probe_config(work: Path, subsets: dict[str, Any], args: argparse.Namespace) -> Any:
    cfg = base_config(
        train_data_dir=str(subsets["train_dir"]),
        val_data_dir=str(subsets["val_dir"]),
        output_dir=str(work / "outputs"),
        logging_dir=str(work / "logs"),
        output_name="valdirprobe",
        seed=args.seed,
        train_batch_size=args.batch_size,
        epoch=1,
        save_every_n_steps=0,
        sampling_enabled=False,
        val_split_percent=args.split_percent,
        val_sample_count=args.sample_count,
        val_interval=0,
        cache_latents=True,
        cache_latents_to_disk=True,
        # Light on purpose: the question is which images reach which forward, and a small rank at a
        # small resolution answers it on a card that is otherwise busy. 512 / 8 = 64 latent pixels,
        # divisible by 16, which is the ROCm rule `bucket_reso_steps` exists for.
        network_type="locon",
        network_dim=args.rank,
        network_alpha=max(1, args.rank // 2),
        conv_dim=args.rank,
        conv_alpha=max(1, args.rank // 2),
        train_resolution=args.resolution,
        enable_bucket=False,
        gradient_checkpointing_unet=False,
        gradient_checkpointing_te=False,
    )
    cfg.run_dir = str(work / "outputs" / "valdirprobe_probe")
    Path(cfg.run_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.logging_dir).mkdir(parents=True, exist_ok=True)
    return cfg


def rows(dataset: Any) -> tuple[set[str], set[str]]:
    """`(training paths, validation paths)` of one dataset, from its own records."""
    train = {str(record["path"]) for record in dataset.records if not record["is_val"]}
    val = {str(record["path"]) for record in dataset.records if record["is_val"]}
    return train, val


def count_cache_entries(folder: Path) -> int:
    """How many latents the trainer encoded into `folder/.latents_cache`.

    A second, independently built route to "which images did this run's dataset hold": the trainer's
    own warm-cache plan is computed from its dataset's records, so a directory nobody referenced
    stays empty. It cannot say which phase read an image, only that the run's dataset held it.
    """
    cache = folder / ".latents_cache"
    return len(list(cache.glob("*.pt"))) if cache.is_dir() else 0


def recorded_steps_per_epoch(result: Any) -> tuple[int | None, str | None]:
    """`steps_per_epoch` the child run itself recorded, and the file it came from.

    `main.py` writes it beside the run's *logs* (`{logging_dir}/{run_id}/steps_per_epoch.json`),
    while `launch_run`'s `run_dir` is the run's *output* directory — the one the checkpoint paths
    hang off. This probe first read the output directory and reported `None`, which looked like a
    missing record rather than a wrong lookup; the fallback below is what a re-run uses.
    """
    run_id = result.run_dir.name if result.run_dir is not None else None
    candidates = [result.logging_dir / run_id / "steps_per_epoch.json"] if run_id else []
    candidates += sorted(result.logging_dir.glob("*/steps_per_epoch.json"))
    for path in candidates:
        if not path.is_file():
            continue
        try:
            return int(json.loads(path.read_text(encoding="utf-8"))["steps_per_epoch"]), str(path)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None, None


def tier_end_to_end(rep: Report, args: argparse.Namespace, subsets: dict[str, Any],
                    work: Path) -> dict[str, Any]:
    """The same claim through the real entry point: a mirror `config.toml` and `trainer/main.py`.

    The in-process arms build their config in memory, so they cannot show that the key in a *file*
    reaches the dataset. This child run writes `[training].val_data_dir` into a mirror config and
    runs the unmodified `trainer/main.py` for whole epochs, and the evidence read back is its own:
    the run log's validation line, its `steps_per_epoch.json`, the latents it encoded per folder,
    and the scalars in its TensorBoard events. Reproduced from `verify_mask_pipeline.launch_run`,
    which already builds the mirror and quietens ROCm.
    """
    from verify_mask_pipeline import launch_run

    # The parent of a re-executed probe is already under the allocation patch, and `child_env`
    # passes LD_PRELOAD on to the child, so asking start_hook.sh to preload again would interpose
    # the same .so twice. An unhooked parent (--no-hook) does need the child started through it.
    already_hooked = os.environ.get(HOOK_MARKER) == "1"
    result = launch_run(
        name="valdir",
        data_dir=subsets["train_dir"],
        seed=args.seed,
        steps=args.steps,
        work=work / "end-to-end",
        masked=False,
        tier=TIER,
        batch_size=args.batch_size,
        save_every_override=1,
        quiet=True,
        hook=not already_hooked,
        extra_sections={
            "training": {
                "val_data_dir": str(subsets["val_dir"]),
                "val_split_percent": args.split_percent,
                "val_sample_count": args.sample_count,
                "val_interval": args.interval,
                # A real save point would render samples and write weights this probe does not need.
                "save_every_n_steps": 0,
                "sampling_enabled": False,
            },
            "network": {
                "network_type": "locon",
                "network_dim": args.rank,
                "network_alpha": max(1, args.rank // 2),
                "conv_dim": args.rank,
                "conv_alpha": max(1, args.rank // 2),
            },
            "bucketing": {"enable_bucket": False, "train_resolution": args.resolution},
            "infrastructure": {"max_data_loader_n_workers": 0, "persistent_workers": False},
        },
    )
    text = result.log_path.read_text(encoding="utf-8", errors="replace")
    val_line = next((line.strip() for line in text.splitlines() if "Validation set:" in line), None)
    split_line = next((line.strip() for line in text.splitlines() if "Validation split:" in line), None)
    steps_recorded, steps_record_path = recorded_steps_per_epoch(result)
    val_cache = count_cache_entries(subsets["val_dir"])
    train_cache = count_cache_entries(subsets["train_dir"])
    expected_points = 1 + (args.steps - 1) // args.interval
    logged: dict[str, int] = {}
    if result.logging_dir.is_dir():
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        for run_dir in sorted(result.logging_dir.iterdir()):
            accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
            try:
                accumulator.Reload()
            except Exception:  # noqa: BLE001 - an unreadable dir is reported by the check below
                continue
            for tag in (SERIES_TAG, *VAL_TAGS):
                if tag in accumulator.Tags().get("scalars", []):
                    logged[tag] = logged.get(tag, 0) + len(accumulator.Scalars(tag))

    details = {
        "exit": result.returncode,
        "gpu_fault": bool(result.gpu_fault),
        "memory_after_child": memory_snapshot(),
        "seconds": round(result.seconds, 1),
        "validation_line": val_line,
        "split_line": split_line,
        "steps_per_epoch_recorded": steps_recorded,
        "steps_per_epoch_record": steps_record_path,
        "latents": {"train_copy": train_cache, "val_copy": val_cache},
        "logged": logged,
        "mirror_config": result.claims.get("mirror_config"),
        "run_dir": str(result.run_dir) if result.run_dir else None,
        "already_hooked_parent": already_hooked,
    }
    if result.returncode != 0 or result.gpu_fault:
        # A child that died for the gfx1201 fault leaves the card's memory held until it is reaped,
        # and the in-process arms would then be measured on that wreckage rather than on a clean
        # card. Stop with the child's own log instead of reporting a number from it.
        raise SystemExit(
            f"stopping before the in-process arms: the mirror-config run ended with exit "
            f"{result.returncode}"
            + (" (gfx1201 GPU memory fault)" if result.gpu_fault else "")
            + f"; see {result.log_path}"
        )
    rep.check(
        TIER, "main.py ran the key from a config file to completion",
        result.returncode == 0 and not result.gpu_fault,
        f"exit {result.returncode} in {result.seconds:.0f}s, gfx1201 fault "
        f"{'detected' if result.gpu_fault else 'none'}; mirror "
        f"{result.claims.get('mirror_config')}",
    )
    rep.check(
        TIER, "the run log reports the custom validation set, and not the split",
        bool(val_line)
        and str(subsets["val_dir"]) in (val_line or "")
        and f"({len(subsets['val_paths'])} images)" in (val_line or "")
        and f"all {len(subsets['train_paths'])} training images kept" in (val_line or "")
        and split_line is None,
        (val_line or "no 'Validation set:' line in " + str(result.log_path))
        + (f" | unexpected split line: {split_line}" if split_line else " | no 'Validation split:' line"),
    )
    rep.check(
        TIER, "the run's own epoch length is the unsplit one",
        steps_recorded == len(subsets["train_paths"]),
        f"steps_per_epoch.json = {steps_recorded} ({steps_record_path}), the train copy's "
        f"{len(subsets['train_paths'])} image(s) at batch {args.batch_size} "
        f"(the split's own epoch would be shorter)",
    )
    rep.check(
        TIER, "the trainer encoded the directory's images and the training folder's, and nothing else",
        val_cache == len(subsets["val_paths"]) and train_cache == len(subsets["train_paths"]),
        f"{train_cache} latent(s) under the train copy's .latents_cache (expected "
        f"{len(subsets['train_paths'])}), {val_cache} under the val copy's (expected "
        f"{len(subsets['val_paths'])}); a directory the run never referenced would stay empty",
    )
    rep.check(
        TIER, "the run's events carry the cadence's validation scalars",
        all(logged.get(tag, 0) == expected_points for tag in VAL_TAGS)
        and logged.get(SERIES_TAG, 0) >= args.steps,
        " + ".join(f"{tag} {logged.get(tag, 0)}" for tag in (SERIES_TAG, *VAL_TAGS))
        + f" point(s), {expected_points} of each validation tag expected",
    )
    return details


def arm_summary(result: Any, val_pool: set[str]) -> dict[str, Any]:
    overlap = result.train_saw_paths & val_pool
    outside = result.val_saw_paths - val_pool
    return {
        "steps": result.steps,
        "train_paths": len(result.train_saw_paths),
        "pass_paths": len(result.val_saw_paths),
        "train_overlap_with_val_pool": len(overlap),
        "pass_paths_outside_val_pool": len(outside),
        "pass_forwards": len(result.val_chunk_sizes),
        "val_points": {
            "loss": len(result.val_losses),
            "avg": len(result.val_avg_losses),
            "fixed": len(result.val_fixed_losses),
        },
        "val_steps": sorted(result.val_losses),
        "pass_seconds_median": (round(statistics.median(result.val_times), 3)
                                if result.val_times else None),
        "step_seconds_median": (round(statistics.median(result.step_times), 3)
                                if result.step_times else None),
        "memory_peak_drm_used_gib": (max(sample["drm_used"] for sample in result.memory)
                                     if result.memory else None),
    }


# --------------------------------------------------------------------------------------
# the experiment
# --------------------------------------------------------------------------------------


def tier_val_dir(rep: Report, args: argparse.Namespace, work: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    from trainer import control, loop
    from trainer.cache import warm_latent_cache
    from trainer.config import tracker_hparams
    from trainer.device_swap import SwapContext
    from trainer.setup import build_dataloader, build_train_objects

    subsets = prepare_subsets(work, args.train_images, args.val_images)
    train_dir, val_dir = subsets["train_dir"], subsets["val_dir"]
    rep.note(
        f"train copy: {len(subsets['train_paths'])} image(s) from {subsets['sources'][0]} -> {train_dir}"
    )
    rep.note(
        f"val copy: {len(subsets['val_paths'])} image(s) from {subsets['sources'][1]} -> {val_dir}"
        + (" (the same folder as the train copy, on disjoint names)"
           if subsets["same_folder"] else "")
    )

    end_to_end: dict[str, Any] = {}
    if not args.no_e2e:
        rep.note("route 2, the real entry point: a mirror config.toml + trainer/main.py for the "
                 "same two copies (its own run log, steps_per_epoch.json, latent caches and events)")
        end_to_end = tier_end_to_end(rep, args, subsets, work)
    else:
        rep.note("--no-e2e: the mirror-config run through trainer/main.py was skipped, so the "
                 "config *file* path is not observed in this report")

    cfg = probe_config(work, subsets, args)
    os.environ["AXL_RUNTIME_DIR"] = str(work / "runtime")
    Path(os.environ["AXL_RUNTIME_DIR"]).mkdir(parents=True, exist_ok=True)
    control.begin_run(os.getpid(), cfg.output_name)
    control.set_status(control.STATUS_TRAINING)

    device = torch.device("cuda")
    artifacts = build_train_objects(cfg)
    artifacts.modules.vae.to(device=device).eval()
    swap_ctx = SwapContext(
        device=device,
        vae=artifacts.modules.vae,
        denoise=artifacts.modules.denoise,
        text_encoders=list(artifacts.modules.text_encoders),
        denoise_optimizer=artifacts.denoise_optimizer,
        te_optimizer=artifacts.te_optimizer,
    )
    rep.note("warming the copy's latent cache (the measured steps must not encode pixels)")
    warm_latent_cache(artifacts.train_dataset, artifacts.modules.vae, cfg, device,
                      artifacts.weight_dtype, swap_ctx=swap_ctx)
    artifacts.modules.vae.to("cpu")
    prepare_artifacts(artifacts)
    swap_ctx.denoise = artifacts.modules.denoise
    swap_ctx.text_encoders = list(artifacts.modules.text_encoders)
    swap_ctx.denoise_optimizer = artifacts.denoise_optimizer
    swap_ctx.te_optimizer = artifacts.te_optimizer
    artifacts.accelerator.init_trackers(project_name="valdirprobe", config=tracker_hparams(cfg))
    rep.note(f"memory after the pipeline is built and the cache warmed: {memory_snapshot()}")

    dataset = artifacts.train_dataset
    train_pool, val_pool = rows(dataset)
    val_files = {str(path) for path in sorted(val_dir.iterdir()) if path.suffix.lower() in IMAGE_EXT}
    details: dict[str, Any] = {
        "config": {
            "val_data_dir": str(cfg.val_data_dir),
            "val_split_percent": float(cfg.val_split_percent),
            "val_sample_count": int(cfg.val_sample_count),
            "val_interval": args.interval,
            "train_batch_size": int(cfg.train_batch_size),
            "train_resolution": int(cfg.train_resolution),
            "network_dim": int(cfg.network_dim),
        },
        "pools": {
            "train_records": len(train_pool),
            "val_records": len(val_pool),
            "val_files_on_disk": len(val_files),
        },
    }

    # P1: the custom directory is the pool, and it took nothing out of training.
    rep.check(
        TIER, "the validation pool is the directory's own images",
        dataset.custom_validation_root is not None
        and val_pool == val_files
        and val_pool == subsets["val_paths"],
        f"{len(val_pool)} validation record(s) == {len(val_files)} file(s) under {val_dir}; "
        f"custom_validation_root={dataset.custom_validation_root}",
    )
    rep.check(
        TIER, "every training image stayed in training",
        train_pool == subsets["train_paths"]
        and dataset.total_samples == len(subsets["train_paths"])
        and dataset.val_image_count == len(subsets["val_paths"]),
        f"{len(train_pool)} training record(s), {dataset.total_samples} draw(s)/epoch (repeat 1 "
        f"each), val_image_count {dataset.val_image_count}, val_split_percent "
        f"{float(cfg.val_split_percent):g} left at its default and ignored",
    )
    instrument = StepInstrument()
    instrument.install(loop)
    snapshot = StateSnapshot(artifacts.modules,
                             [artifacts.denoise_optimizer, artifacts.te_optimizer])
    arms: dict[str, Any] = {}
    try:
        # --- A1: the treatment. The pass-off run of the same 50 steps is the isolation control the
        # sibling probe uses: the pass must ride the custom directory without moving the step.
        off = measure(
            artifacts=artifacts, cfg=cfg, loop_module=loop, control=control, swap_ctx=swap_ctx,
            snapshot=snapshot, arm_name="A1off", interval=0, sample_count=args.sample_count,
            steps=args.steps, instrument=instrument,
        )
        on = measure(
            artifacts=artifacts, cfg=cfg, loop_module=loop, control=control, swap_ctx=swap_ctx,
            snapshot=snapshot, arm_name="A1on", interval=args.interval,
            sample_count=args.sample_count, steps=args.steps, instrument=instrument,
        )
        arms["A1_treatment_off"] = arm_summary(off, val_pool)
        arms["A1_treatment_on"] = arm_summary(on, val_pool)

        rep.check(
            TIER, "A1: no training forward carried a validation-directory image",
            not (on.train_saw_paths & val_pool) and bool(on.train_saw_paths),
            f"{len(on.train_saw_paths)} distinct training path(s), {len(val_pool)} in the pool, "
            f"overlap {len(on.train_saw_paths & val_pool)}",
        )
        rep.check(
            TIER, "A1: every training image was trained (the split held nothing out)",
            on.train_saw_paths == train_pool and train_pool == subsets["train_paths"],
            f"{len(on.train_saw_paths)} distinct training path(s) of {len(train_pool)} record(s)",
        )
        rep.check(
            TIER, "A1: both passes scored the directory's images only",
            bool(on.val_saw_paths) and on.val_saw_paths == val_pool
            and not (on.val_saw_paths & train_pool),
            f"{len(on.val_saw_paths)} distinct pass path(s), {len(on.val_saw_paths - val_pool)} "
            f"outside the directory, {len(on.val_saw_paths & train_pool)} shared with training",
        )
        expected_points = 1 + (on.steps - 1) // args.interval if on.steps >= 1 else 0
        rep.check(
            TIER, "A1: the three scalars appear at the cadence",
            len(on.val_losses) == expected_points
            and len(on.val_avg_losses) == expected_points
            and len(on.val_fixed_losses) == expected_points
            and sorted(on.val_losses) == [1 + index * args.interval for index in range(expected_points)],
            f"{len(on.val_losses)}/{len(on.val_avg_losses)}/{len(on.val_fixed_losses)} "
            f"(Val/Loss, Val/Avg_Loss, Val/Fixed_Loss) over {on.steps} step(s) at interval "
            f"{args.interval}, expected {expected_points} of each",
        )
        fixed_passes = on.val_pass_paths[1::2]
        random_passes = on.val_pass_paths[0::2]
        rep.check(
            TIER, "A1: the fixed pass scored the same directory images every cadence step",
            len(on.val_pass_paths) == 2 * expected_points
            and bool(fixed_passes)
            and len(set(fixed_passes)) == 1
            and all(paths <= val_pool for paths in fixed_passes),
            f"{len(on.val_pass_paths)} pass(es) for {expected_points} cadence step(s); "
            f"{len(set(fixed_passes))} distinct fixed sample(s)",
        )
        if len(val_pool) > args.sample_count:
            rep.check(
                TIER, "A1: the random pass moved across the directory",
                len({tuple(sorted(paths)) for paths in random_passes}) > 1,
                f"{len({tuple(sorted(paths)) for paths in random_passes})} distinct random "
                f"subset(s) over {len(random_passes)} pass(es) against {len(val_pool)} pooled "
                f"image(s) and val_sample_count {args.sample_count}",
            )
        else:
            rep.note(
                f"A1: {len(val_pool)} pooled image(s) against val_sample_count "
                f"{args.sample_count} means the random pass scores them all"
            )
        identical, first_difference = bit_identical(off.losses, on.losses)
        rep.check(
            TIER, "A1: the pass on the custom directory left the training step bit-identical",
            identical == len(off.losses) and bool(off.losses)
            and off.params_digest == on.params_digest,
            f"{identical}/{len(off.losses)} step(s) byte-identical to the pass-off run, first "
            f"difference {first_difference}; parameter digest "
            f"{'equal' if off.params_digest == on.params_digest else 'DIFFERENT'} "
            f"({off.params_digest[:16]} / {on.params_digest[:16]})",
        )
        details["isolation"] = {
            "steps_bit_identical": identical,
            "steps_compared": len(off.losses),
            "first_differing_step": first_difference,
            "params_equal": off.params_digest == on.params_digest,
        }

        # --- A2: the instrument's own control. `val_data_dir` pointed at the training folder makes
        # the two pools the same folder, so a detector that works has to report the overlap.
        cfg_leak = dataclasses.replace(cfg, val_data_dir=str(train_dir))
        leak_ds, leak_loader = build_dataloader(cfg_leak)
        leak_train, leak_val = rows(leak_ds)
        leak_artifacts = dataclasses.replace(artifacts, train_dataset=leak_ds, dataloader=leak_loader)
        leak_on = measure(
            artifacts=leak_artifacts, cfg=cfg_leak, loop_module=loop, control=control,
            swap_ctx=swap_ctx, snapshot=snapshot, arm_name="A2leak", interval=args.short_interval,
            sample_count=args.sample_count, steps=args.short_steps, instrument=instrument,
        )
        leak_overlap = leak_on.train_saw_paths & leak_val
        arms["A2_integrity_leak"] = arm_summary(leak_on, leak_val)
        rep.check(
            TIER, "A2 integrity control: the detector reports a leaking configuration",
            leak_train == leak_val and bool(leak_overlap),
            f"{len(leak_train)} training record(s) and {len(leak_val)} validation record(s) are "
            f"the same folder; {len(leak_overlap)} training forward path(s) were also in the "
            f"validation pool, so the check that read 0 in A1 does fire here",
        )

        # --- A3: the factor removed. Same training images, no custom directory: the pass pool comes
        # from the training folder and the epoch loses the held-out draws.
        cfg_split = dataclasses.replace(cfg, val_data_dir="")
        split_ds, split_loader = build_dataloader(cfg_split)
        split_train, split_val = rows(split_ds)
        split_artifacts = dataclasses.replace(artifacts, train_dataset=split_ds, dataloader=split_loader)
        split_on = measure(
            artifacts=split_artifacts, cfg=cfg_split, loop_module=loop, control=control,
            swap_ctx=swap_ctx, snapshot=snapshot, arm_name="A3split", interval=args.short_interval,
            sample_count=args.sample_count, steps=args.short_steps, instrument=instrument,
        )
        arms["A3_split_control"] = arm_summary(split_on, split_val)
        details["split_control"] = {
            "val_split_percent": float(cfg_split.val_split_percent),
            "val_records": len(split_val),
            "train_records": len(split_train),
            "total_samples": split_ds.total_samples,
            "val_pool_is_inside_the_training_folder": bool(split_val)
            and split_val <= subsets["train_paths"],
        }
        rep.check(
            TIER, "A3 control: without the key, the pool and the epoch are the split's",
            bool(split_val)
            and split_val <= subsets["train_paths"]
            and not (split_val & subsets["val_paths"])
            and split_ds.total_samples < dataset.total_samples
            and not (split_on.train_saw_paths & split_val)
            and bool(split_on.val_saw_paths)
            and split_on.val_saw_paths <= split_val,
            f"val_split_percent {float(cfg_split.val_split_percent):g} pooled "
            f"{len(split_val)} training-folder image(s) ({len(split_val & subsets['val_paths'])} of "
            f"them from the val copy) and cut the epoch to {split_ds.total_samples} draw(s) against "
            f"the custom directory's {dataset.total_samples}; {len(split_on.val_saw_paths)} pass "
            f"path(s), all inside its own pool; {len(split_on.train_saw_paths & split_val)} "
            f"held-out image(s) trained",
        )
        details["split_control"]["pass_paths"] = len(split_on.val_saw_paths)
        details["split_control"]["train_saw"] = len(split_on.train_saw_paths)
    finally:
        instrument.uninstall(loop)
        try:
            artifacts.accelerator.end_training()
        except Exception:  # noqa: BLE001 - the probe's own report is the deliverable
            pass
        try:
            control.end_run(control.STATUS_FINISHED)
        except Exception:  # noqa: BLE001
            pass

    # The events are the artifact a reader can check independently of the instrument: the three
    # validation tags have to be in the run's own TensorBoard directory.
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    logged: dict[str, int] = {}
    for run_dir in sorted(Path(cfg.logging_dir).glob("*")):
        accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
        try:
            accumulator.Reload()
        except Exception:  # noqa: BLE001 - an empty/unreadable dir is not a check failure here
            continue
        for tag in (SERIES_TAG, *VAL_TAGS):
            if tag in accumulator.Tags().get("scalars", []):
                logged[tag] = logged.get(tag, 0) + len(accumulator.Scalars(tag))
    rep.check(
        TIER, "the run's events carry the three validation scalars beside Train/Loss",
        all(logged.get(tag, 0) > 0 for tag in VAL_TAGS) and logged.get(SERIES_TAG, 0) > 0,
        " + ".join(f"{tag} {logged.get(tag, 0)}" for tag in (SERIES_TAG, *VAL_TAGS))
        + f" point(s) in {cfg.logging_dir}",
    )

    details["arms"] = arms
    details["end_to_end"] = end_to_end
    details["logged"] = logged
    details["memory_peak_drm_used_gib"] = arms["A1_treatment_on"]["memory_peak_drm_used_gib"]
    return details, {}


# --------------------------------------------------------------------------------------
# argv / report
# --------------------------------------------------------------------------------------


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=50,
                        help="optimizer steps of the treatment arm, run twice (pass off, pass on; "
                             "default 50)")
    parser.add_argument("--short-steps", type=int, default=6,
                        help="steps of the two control arms (default 6: three cadence steps at "
                             "--short-interval 2)")
    parser.add_argument("--short-interval", type=int, default=2,
                        help="val_interval of the two control arms (default 2)")
    parser.add_argument("--interval", type=int, default=5,
                        help="val_interval of the treatment arm (default 5, the shipped one)")
    parser.add_argument("--sample-count", type=int, default=3,
                        help="val_sample_count (default 3, so the random pass moves over a 4-image "
                             "directory)")
    parser.add_argument("--split-percent", type=float, default=10.0,
                        help="val_split_percent, left at its default in the treatment so it can be "
                             "seen being ignored (default 10)")
    parser.add_argument("--train-images", type=int, default=10,
                        help="images copied into the train subset (default 10)")
    parser.add_argument("--val-images", type=int, default=4,
                        help="images copied into the validation directory (default 4)")
    parser.add_argument("--batch-size", type=int, default=1, help="train_batch_size (default 1)")
    parser.add_argument("--resolution", type=int, default=512,
                        help="train_resolution with bucketing off (default 512)")
    parser.add_argument("--rank", type=int, default=4,
                        help="locon network_dim / conv_dim (default 4)")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--report-dir", type=str, default=None)
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--no-e2e", action="store_true",
                        help="skip route 2 (the mirror config + trainer/main.py child run)")
    parser.add_argument("--clean", action="store_true", help="delete the scratch dir when done")
    parser.add_argument("--no-hook", action="store_true",
                        help="do not re-exec under the [environment].amdfq allocation patch; falls "
                             "back to HSA_SVM_GUARD_PAGES=0 as the gfx1201 fault dodge. Not "
                             "recommended: a bare run can still hit the Tensile over-read "
                             "(doc/troubleshooting.md), which kills the process mid-run")
    parser.add_argument("--force", action="store_true",
                        help="run even if a live training process looks present")
    parser.add_argument("--allow-foreign-env", action="store_true")
    return parser.parse_args(argv)


def write_report(rep: Report, args: argparse.Namespace, details: dict[str, Any],
                 verdict: str) -> Path:
    out = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "probe": "val-dir",
        "prediction": PREDICTION,
        "verdict": verdict,
        "args": vars(args),
        "env": rep.env,
        "notes": rep.notes,
        "checks": [{"tier": c.tier, "name": c.name, "ok": c.ok, "detail": c.detail, "values": c.values}
                   for c in rep.checks],
        "observations": [{"tier": o.tier, "name": o.name, "detail": o.detail, "values": o.values}
                         for o in rep.observations],
        "details": details,
        "seconds": round(time.time() - rep.started, 1),
    }
    (out / "report.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Validation-directory probe",
        "",
        f"prediction: {PREDICTION}",
        f"verdict: {verdict}",
        f"seconds: {payload['seconds']}",
        "",
        "## config",
        "",
        "```json",
        json.dumps(details.get("config") or {}, indent=2, default=str),
        "```",
        "",
        "## arms (counted `image_path` sets; overlap is the leak measure)",
        "",
        "| arm | steps | train paths | pass paths | train ∩ val pool | pass − val pool | val points (loss/avg/fixed) | step s | pass s | drm peak GiB |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, arm in (details.get("arms") or {}).items():
        if not isinstance(arm, dict):
            lines.append(f"| {name} | — | — | — | — | — | — | — | — | — |")
            continue
        points = arm.get("val_points") or {}
        lines.append(
            f"| {name} | {arm.get('steps')} | {arm.get('train_paths')} | "
            f"{arm.get('pass_paths')} | {arm.get('train_overlap_with_val_pool')} | "
            f"{arm.get('pass_paths_outside_val_pool')} | "
            f"{points.get('loss')}/{points.get('avg')}/{points.get('fixed')} | "
            f"{arm.get('step_seconds_median')} | {arm.get('pass_seconds_median')} | "
            f"{arm.get('memory_peak_drm_used_gib')} |"
        )
    e2e = details.get("end_to_end") or {}
    if e2e:
        lines += [
            "",
            "## route 2: mirror config + trainer/main.py",
            "",
            f"- exit {e2e.get('exit')}, gfx1201 fault {e2e.get('gpu_fault')}, "
            f"{e2e.get('seconds')} s",
            f"- `{e2e.get('validation_line')}`",
            f"- `Validation split:` line: {e2e.get('split_line')}",
            f"- steps_per_epoch.json: {e2e.get('steps_per_epoch_recorded')}",
            f"- latents encoded: {e2e.get('latents')}",
            f"- scalars: {e2e.get('logged')}",
        ]
    lines += [
        "",
        "## checks",
        "",
    ]
    for check in rep.checks:
        lines.append(f"- {'PASS' if check.ok else 'FAIL'} [{check.tier}] {check.name} — {check.detail}")
    summary = out / "summary.md"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    os.chdir(REPO_ROOT)  # config.toml is read relative to the repo root
    argv_tail = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv_tail)
    report_dir = (Path(args.report_dir) if args.report_dir
                  else Path("/tmp/axl-probe-val-dir") / time.strftime("%Y%m%d_%H%M%S"))
    work = Path(args.work_dir) if args.work_dir else report_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    args.report_dir, args.work_dir = report_dir, work
    rep = Report(report_dir)
    print(f"== val-dir probe | report={report_dir}", flush=True)
    print(f"   prediction: {PREDICTION}", flush=True)

    maybe_reexec_under_hook(args, rep, argv_tail, script=Path(__file__).resolve())
    alloc_conf = launcher_alloc_conf()
    if alloc_conf:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", alloc_conf)
    rep.note(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')} "
             f"(start_train.sh's value, so the allocator behaves as in a real run)")
    guard_environment(args, rep)
    if not rep.env.get("cuda_available"):
        raise SystemExit("refusing to run: this probe needs the GPU (torch.cuda.is_available() false)")
    live = live_training_runs()
    if live and not args.force:
        raise SystemExit("refusing to run with a live training process (pass --force to override):\n  "
                         + "\n  ".join(live))
    if live:
        rep.note(f"--force used while these looked live: {live}")
    rep.note("Read-only on the dataset: images are copied into the scratch dir first, and the probe "
             "warms the copy's latent cache, never the configured one.")
    rep.note(f"arms: A1 treatment (val_data_dir, {args.steps} steps on and the same off), "
             f"A2 integrity (the directory pointed at the training folder, {args.short_steps} steps), "
             f"A3 split control (no directory, {args.short_steps} steps)")

    details: dict[str, Any] = {}
    verdict = "not run"
    try:
        details, _ = tier_val_dir(rep, args, work)
        arms = details.get("arms") or {}
        on = arms.get("A1_treatment_on") or {}
        clean = (
            on.get("train_overlap_with_val_pool") == 0
            and on.get("pass_paths_outside_val_pool") == 0
            and (arms.get("A2_integrity_leak") or {}).get("train_overlap_with_val_pool", 0) > 0
        )
        e2e = details.get("end_to_end") or {}
        e2e_ok = bool(e2e.get("validation_line")) and e2e.get("split_line") is None and not args.no_e2e
        verdict = (
            "the custom directory supplied both passes, no validation image reached a training "
            "forward, the same detector reported the deliberately leaking configuration, and the "
            "unmodified trainer read the key from a config file and reported the unsplit epoch "
            "with only the directory's images cached"
            if clean and e2e_ok else
            "NOT established: see the failing checks (the leak detector, the pool, the cadence or "
            "the mirror-config run)"
        )
    finally:
        report_path = write_report(rep, args, details, verdict)

    print("")
    print(f"== {len(rep.checks)} checks, {len(rep.failures)} failed | report: {report_path}", flush=True)
    print(f"== verdict: {verdict}", flush=True)
    if args.clean:
        shutil.rmtree(work, ignore_errors=True)
        print(f"== removed scratch dir {work}", flush=True)
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())
