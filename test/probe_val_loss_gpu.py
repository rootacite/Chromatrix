#!/usr/bin/env python3
"""GPU probe: does the validation pass perturb training, and what does it cost per step?

Run from the repo root in the conda env named by `environment.yml`:

    conda activate axl
    python test/probe_val_loss_gpu.py

Two questions, one experiment each, on the real trainer objects (`setup.build_train_objects` +
`prepare_artifacts` + `loop.train_one_epoch`), a small copy of the configured dataset, and a warm
latent cache:

  isolation  For every arm, the same steps are run twice from the same restored state: once with
             the validation pass off (`val_interval = 0`) and once with it on. The two runs' per-step
             `Train/Loss` values must be *bit-identical* and their LoRA parameters must hash equal.
             This is a same-process, restore-state pair on purpose. A pair of separate child runs
             cannot decide this: this repo's own `test/probe_determinism_gpu.py` established that
             the same seed does NOT reproduce a loss series across processes, so two children could
             differ for reasons that have nothing to do with the pass. Restoring one process's
             state removes that whole failure mode: the only variable between the two runs is
             whether `compute_validation_loss` runs.
             The same pair is what says the pass does not shift training's RNG stream, does not
             write a gradient, and does not touch the optimizer.

  cost       Per-step wall time for the pass-off run and the pass-on run of the same arm, from the
             instrument that watches each optimizer step. Two passes run per cadence step — the
             random subset (`Val/Loss`, whose points feed `Val/Avg_Loss`) and the fixed, mutually
             dissimilar sample (`Val/Fixed_Loss`) — so an arm's step time carries both.
             Arms: `val_interval = 1` (every step), the shipped `5`, and `10`, each with
             `val_sample_count = 8` (the shipped count), plus `val_interval = 1` with
             `val_sample_count = 1` for the per-scored-image slope. A cadence step is step 1 and
             then every `val_interval` (`loop.validation_due`), so the expected count of points per
             tag is `1 + (steps - 1) // interval`.
             Reported as mean / median / p90 ms per step and the overhead against the same steps
             with the pass off. The **mean** is the cadence figure — the median hides the few slow
             steps of a coarse arm — while the median says what a typical step costs. The pass's own
             duration is measured too. The answer to "what would validating every step cost" is the
             interval-1 row, and the report says plainly whether it is a few percent or a multiple.
             `--steps` is a floor, not a count: as in `main.py` the step budget is checked between
             epochs, so an arm runs whole epochs (one epoch is the measured unit's smallest size).

Leakage, on the same real path (two-sided): every batch of the real training dataloader is walked
for the arm's epochs and no held-out image may appear; and every path the pass hands to
`build_group_inputs` must be one of the held-out images. A caption cannot identify an image, so the
probe records the batch's own `image_path`s, not the prompts.

Nothing is written into the dataset: images are copied into the scratch directory first, and the
latent cache the probe warms belongs to that copy. Pass `--clean` to delete the scratch directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
# A script's sys.path[0] is its own directory, which is what makes `import verify_mask_pipeline`
# below work and what the trainer's package imports need.
sys.path.insert(0, str(REPO_ROOT))

from verify_mask_pipeline import (  # noqa: E402 - the path above is what makes this import work
    Report,
    base_config,
    config_sections,
    guard_environment,
    live_training_runs,
)

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
TIER = "val-loss"
SERIES_TAG = "Train/Loss"
VAL_TAG = "Val/Loss"
# Set on the re-executed process so it does not re-exec itself again.
HOOK_MARKER = "AXL_VAL_PROBE_HOOKED"
PREDICTION = (
    "Both validation passes leave the training step untouched: for every arm, the pass-on run's "
    "Train/Loss values and LoRA parameters are bit-identical to the pass-off run's same steps. The "
    "fixed pass scores the same images at every cadence step while the random one moves, and the "
    "arm's step time carries both passes."
)


# --------------------------------------------------------------------------------------
# dataset copy (read-only source), same shape the other GPU probes use
# --------------------------------------------------------------------------------------


def configured_train_dirs() -> list[Path]:
    environment = config_sections().get("environment", {})
    raw = environment.get("train_data") or []
    dirs: list[Path] = []
    if isinstance(raw, list):
        for block in raw:
            if isinstance(block, dict) and block.get("path"):
                dirs.append(Path(str(block["path"])).expanduser())
    if not dirs and environment.get("train_data_dir"):
        dirs.append(Path(str(environment["train_data_dir"])).expanduser())
    return [path for path in dirs if path.is_dir()]


def copy_images(source: Path, dest: Path, count: int) -> tuple[int, int]:
    names = sorted(
        path
        for path in source.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXT
        and not path.name.lower().endswith(".mask.png")
    )
    placed = reused = 0
    for path in names:
        if placed + reused >= count:
            break
        if (dest / path.name).exists():
            reused += 1
            continue
        target = dest / path.name
        shutil.copy2(path, target)
        sidecar = path.with_suffix(".txt")
        if sidecar.is_file():
            shutil.copy2(sidecar, target.with_suffix(".txt"))
        placed += 1
    return placed, reused


def prepare_dataset(work: Path, images: int) -> tuple[Path, dict[str, Any]]:
    sources = configured_train_dirs()
    if not sources:
        raise SystemExit(
            "config.toml names no existing training folder ([[environment.train_data]] / "
            "train_data_dir); the probe needs a dataset to copy"
        )
    dest = work / "dataset"
    dest.mkdir(parents=True, exist_ok=True)
    quota = [images // len(sources) + (1 if index < images % len(sources) else 0)
             for index in range(len(sources))]
    pairs = [copy_images(source, dest, want) for source, want in zip(sources, quota)]
    placed = sum(placed for placed, _ in pairs)
    reused = sum(kept for _, kept in pairs)
    if placed + reused < 4:
        raise SystemExit(f"only {placed + reused} image(s) placed in {dest}; four are the minimum to split")
    return dest, {
        "dir": str(dest),
        "images": placed + reused,
        "copied": placed,
        "reused": reused,
        "captions": len(list(dest.glob("*.txt"))),
        "sources": {str(source): placed_count for (placed_count, _), source in zip(pairs, sources)},
    }


# --------------------------------------------------------------------------------------
# instruments
# --------------------------------------------------------------------------------------


@dataclass
class ArmResult:
    name: str
    interval: int
    sample_count: int
    steps: int
    losses: list[float] = field(default_factory=list)
    val_losses: dict[int, float] = field(default_factory=dict)
    val_avg_losses: dict[int, float] = field(default_factory=dict)
    val_fixed_losses: dict[int, float] = field(default_factory=dict)
    val_pass_paths: list[frozenset[str]] = field(default_factory=list)
    val_chunk_sizes: list[int] = field(default_factory=list)
    step_times: list[float] = field(default_factory=list)
    val_times: list[float] = field(default_factory=list)
    val_saw_paths: set[str] = field(default_factory=set)
    train_saw_paths: set[str] = field(default_factory=set)
    start_digest: str = ""
    params_digest: str = ""
    seconds: float = 0.0
    memory: list[dict[str, float]] = field(default_factory=list)

    def memory_summary(self) -> dict[str, Any]:
        if not self.memory:
            return {}
        peak = max(sample["drm_used"] for sample in self.memory)
        last = self.memory[-1]
        return {
            "first": self.memory[0],
            "last": last,
            "drm_used_peak": peak,
            "gap_at_peak": round(peak - last["torch_allocated"], 3),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "interval": self.interval,
            "sample_count": self.sample_count,
            "steps": self.steps,
            "val_points": len(self.val_losses),
            "val_seconds": {
                "median_ms": _ms(statistics.median(self.val_times)) if self.val_times else None,
                "p90_ms": _ms(_percentile(self.val_times, 0.9)) if self.val_times else None,
            },
            "step_seconds": {
                "median_ms": _ms(statistics.median(self.step_times)) if self.step_times else None,
                "p90_ms": _ms(_percentile(self.step_times, 0.9)) if self.step_times else None,
            },
            "params_digest": self.params_digest[:16],
        }


def _ms(seconds: float) -> float:
    return round(seconds * 1000.0, 1)


def _fmt_or_none(value: float | None) -> str:
    return "n/a (no pass in this arm)" if value is None else f"{value} ms"


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no values")
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def params_digest(modules: Any) -> str:
    """One hash over every trainable LoRA parameter, in name order."""
    digest = hashlib.sha256()
    for model in [modules.denoise, *modules.text_encoders]:
        for name, param in _trainable(model):
            digest.update(name.encode("utf-8"))
            digest.update(param.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def memory_snapshot() -> dict[str, float]:
    """One memory reading from three places, in GiB.

    `drm_used` is the amdgpu driver's own counter (sysfs), which nothing in user space can re-account;
    `hip_free` is what `torch.cuda.mem_get_info()` reports (the call the allocation patch can
    re-account, so it is not a witness on its own); `torch_allocated` is what torch counts it holds.
    A gap between `drm_used` and the process's own accounting is what an OOM report needs.
    """
    import torch

    drm_bytes = 0
    for path in Path("/sys/class/drm").glob("card*/device/mem_info_vram_used"):
        try:
            drm_bytes += int(path.read_text().strip())
        except OSError:
            continue
    hip_free = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
    return {
        "drm_used": round(drm_bytes / 2**30, 3),
        "hip_free": round(hip_free / 2**30, 3),
        "torch_allocated": round(torch.cuda.memory_allocated() / 2**30, 3),
        "torch_reserved": round(torch.cuda.memory_reserved() / 2**30, 3),
    }


def _trainable(model: Any) -> list[tuple[str, Any]]:
    """The module's trainable parameters in name order (LoRA only: the base weights are frozen)."""
    return sorted((name, param) for name, param in model.named_parameters() if param.requires_grad)


class StateSnapshot:
    """Everything one measured step consumes, so the next arm starts from the identical state.

    Parameters and optimizer state decide the update; the RNG states decide the noise, timesteps
    and dropout masks; the module modes decide dropout and gradient checkpointing. Restoring all
    three is what makes "pass off" and "pass on" a one-variable comparison inside one process.
    """

    def __init__(self, modules: Any, optimizers: list[Any]) -> None:
        import torch

        self.models = [modules.denoise, *modules.text_encoders]
        self.params = [[param.detach().clone() for _, param in _trainable(model)]
                       for model in self.models]
        # The state is keyed by parameter and may be empty at step 0 (it is created on the first
        # step), so the snapshot holds an entry per parameter either way: restoring has to *clear*
        # state the measured run created, not just overwrite what was already there, or the second
        # run would inherit the first one's optimizer.
        self.optimizer = []
        for optimizer in optimizers:
            entries = []
            for param in optimizer.param_groups[0]["params"]:
                state = optimizer.state.get(param, {})
                entries.append({key: value.detach().clone() if hasattr(value, "detach") else value
                                for key, value in state.items()})
            self.optimizer.append((entries, [
                {key: (value.detach().clone() if hasattr(value, "detach") else value)
                 for key, value in group.items() if key != "params"}
                for group in optimizer.param_groups
            ]))
        self.modes = [bool(model.training) for model in self.models]
        self.cpu_rng = torch.get_rng_state().clone()
        self.device_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
        self.epoch = 0

    def restore(self, modules: Any, optimizers: list[Any], dataset: Any) -> None:
        import torch

        for model, saved in zip(self.models, self.params):
            for (_, param), value in zip(_trainable(model), saved):
                param.data.copy_(value)
        for optimizer, (entries, groups) in zip(optimizers, self.optimizer):
            for param, state in zip(optimizer.param_groups[0]["params"], entries):
                if not state:
                    optimizer.state.pop(param, None)
                    continue
                bucket = optimizer.state.setdefault(param, {})
                for key in [key for key in bucket if key not in state]:
                    del bucket[key]
                for key, value in state.items():
                    if hasattr(value, "detach") and key in bucket:
                        bucket[key].copy_(value)
                    else:
                        bucket[key] = value.clone() if hasattr(value, "detach") else value
            for group, saved in zip(optimizer.param_groups, groups):
                for key in [key for key in group if key != "params" and key not in saved]:
                    del group[key]
                for key, value in saved.items():
                    group[key] = value.clone() if hasattr(value, "detach") else value
        for model, flag in zip(self.models, self.modes):
            model.train(flag)
        torch.set_rng_state(self.cpu_rng)
        torch.cuda.set_rng_state_all(self.device_rng)
        dataset.set_epoch(self.epoch)


class StepInstrument:
    """Watches the real loop: per-step loss and timing, each pass, and who saw what."""

    def __init__(self) -> None:
        self.losses: list[float] = []
        self.val_losses: dict[int, float] = {}
        self.val_avg_losses: dict[int, float] = {}
        self.val_fixed_losses: dict[int, float] = {}
        self.step_times: list[float] = []
        self.val_times: list[float] = []
        self.val_saw: set[str] = set()
        # One entry per pass call, so the fixed sample's stability can be checked directly.
        self.val_pass_paths: list[frozenset[str]] = []
        # One entry per forward inside a pass: the images it carried (the `train_batch_size` cap).
        self.val_chunk_sizes: list[int] = []
        self.train_saw: set[str] = set()
        self._last_clock: float | None = None
        self._reading_val = False
        self._original_log = None
        self._original_val = None
        self._original_inputs = None

    def install(self, loop_module: Any) -> None:
        self._original_log = loop_module._maybe_log_and_sample
        self._original_val = loop_module.compute_validation_loss
        self._original_inputs = loop_module.build_group_inputs

        instrument = self
        original_log = self._original_log
        original_val = self._original_val
        original_inputs = self._original_inputs

        def log(*args, **kwargs):
            now = time.perf_counter()
            if instrument._last_clock is not None:
                instrument.step_times.append(now - instrument._last_clock)
            instrument._last_clock = now
            instrument.memory.append(memory_snapshot())
            instrument.losses.append(float(original_log.last_loss))
            step = int(kwargs["global_step"])
            for key, target in (
                ("val_loss", instrument.val_losses),
                ("val_avg_loss", instrument.val_avg_losses),
                ("val_fixed_loss", instrument.val_fixed_losses),
            ):
                value = kwargs.get(key)
                if value is not None:
                    target[step] = float(value)
            return original_log(*args, **kwargs)

        def val(*args, **kwargs):
            start = time.perf_counter()
            instrument._reading_val = True
            instrument._pass_paths = set()
            try:
                return original_val(*args, **kwargs)
            finally:
                instrument._reading_val = False
                instrument.val_times.append(time.perf_counter() - start)
                instrument.val_pass_paths.append(frozenset(instrument._pass_paths))

        def inputs(*args, **kwargs):
            batch = kwargs["batch"]
            indices = kwargs["indices"]
            paths = {str(batch["image_path"][index]) for index in indices}
            if instrument._reading_val:
                instrument.val_saw |= paths
                instrument._pass_paths |= paths
                instrument.val_chunk_sizes.append(len(indices))
            else:
                instrument.train_saw |= paths
            return original_inputs(*args, **kwargs)

        loop_module._maybe_log_and_sample = log
        loop_module.compute_validation_loss = val
        loop_module.build_group_inputs = inputs

    def uninstall(self, loop_module: Any) -> None:
        loop_module._maybe_log_and_sample = self._original_log
        loop_module.compute_validation_loss = self._original_val
        loop_module.build_group_inputs = self._original_inputs

    def reset(self) -> None:
        self.losses = []
        self.val_losses = {}
        self.val_avg_losses = {}
        self.val_fixed_losses = {}
        self.val_pass_paths = []
        self.val_chunk_sizes = []
        self.step_times = []
        self.val_times = []
        self.memory = []
        self.val_saw = set()
        self.train_saw = set()
        self._last_clock = None


# --------------------------------------------------------------------------------------
# the measured runs
# --------------------------------------------------------------------------------------


def run_steps(*, artifacts: Any, cfg: Any, loop_module: Any, control: Any, swap_ctx: Any,
              steps: int, instrument: StepInstrument) -> int:
    """`main.py`'s epoch loop, reduced to what the measurement needs: the real `train_one_epoch`.

    Epoch bookkeeping is copied from `main.py` (`set_epoch` on the dataset and the sampler,
    `cfg._current_epoch = epoch + 1`) so the batch order and the loss recorder behave as in a run.
    """
    global_step = 0
    guard = 0
    while global_step < steps and guard < 1000:
        guard += 1
        epoch = int(cfg._current_epoch)  # index of the epoch about to run
        artifacts.train_dataset.set_epoch(epoch)
        sampler = getattr(artifacts.dataloader, "batch_sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        cfg._current_epoch = epoch + 1
        before = global_step
        global_step = loop_module.train_one_epoch(
            artifacts=artifacts,
            cfg=cfg,
            global_step=global_step,
            progress=None,
            total_train_steps=steps,
            swap_ctx=swap_ctx,
        )
        if global_step == before or control.should_stop():
            break
    return global_step


def measure(*, artifacts: Any, cfg: Any, loop_module: Any, control: Any, swap_ctx: Any,
            snapshot: StateSnapshot, arm_name: str, interval: int, sample_count: int, steps: int,
            instrument: StepInstrument) -> ArmResult:
    snapshot.restore(artifacts.modules, [artifacts.denoise_optimizer, artifacts.te_optimizer],
                     artifacts.train_dataset)
    cfg.val_interval = interval
    cfg.val_sample_count = sample_count
    cfg._current_epoch = 0
    instrument.reset()
    started = time.perf_counter()
    start_digest = params_digest(artifacts.modules)
    done = run_steps(artifacts=artifacts, cfg=cfg, loop_module=loop_module, control=control,
                     swap_ctx=swap_ctx, steps=steps, instrument=instrument)
    elapsed = time.perf_counter() - started
    result = ArmResult(
        name=arm_name,
        interval=interval,
        sample_count=sample_count,
        steps=done,
        losses=list(instrument.losses),
        val_losses=dict(instrument.val_losses),
        val_avg_losses=dict(instrument.val_avg_losses),
        val_fixed_losses=dict(instrument.val_fixed_losses),
        val_pass_paths=list(instrument.val_pass_paths),
        val_chunk_sizes=list(instrument.val_chunk_sizes),
        step_times=list(instrument.step_times),
        val_times=list(instrument.val_times),
        val_saw_paths=set(instrument.val_saw),
        train_saw_paths=set(instrument.train_saw),
        start_digest=start_digest,
        params_digest=params_digest(artifacts.modules),
        seconds=elapsed,
        memory=list(instrument.memory),
    )
    return result


def bit_identical(left: list[float], right: list[float]) -> tuple[int, int | None]:
    """(how many steps are byte-identical, first differing step or None)."""
    import struct

    if len(left) != len(right):
        return 0, None
    identical = 0
    first: int | None = None
    for index, (a, b) in enumerate(zip(left, right)):
        if struct.pack("<f", a) == struct.pack("<f", b):
            identical += 1
        elif first is None:
            first = index + 1
    return identical, first


def arm_plan(args: argparse.Namespace) -> list[tuple[str, int, int]]:
    """`(name, interval, sample_count)` per arm, in the order they are launched."""
    plan = [(f"I{interval}C{args.sample_count}", interval, args.sample_count)
            for interval in args.intervals]
    for count in args.count_ladder:
        plan.append((f"I1C{count}", 1, count))
    names = [name for name, _, _ in plan]
    if len(set(names)) != len(names):
        raise SystemExit(f"two arms share a name: {names} (adjust --intervals / --count-ladder)")
    for name, interval, count in plan:
        if interval < 1:
            raise SystemExit(f"arm {name}: a measured arm needs val_interval >= 1 (the pass-off "
                             f"baseline is the same arm with the pass disabled)")
        if count < 1:
            raise SystemExit(f"arm {name}: val_sample_count must be >= 1")
    return plan


def prepare_artifacts(artifacts: Any) -> None:
    """`trainer/main.py::_prepare_artifacts`, mirrored.

    `trainer/main.py` uses bare imports (`from config import ...`) because `start_train.sh` runs it
    as a script with `trainer/` on `sys.path[0]`; importing it here would load a second copy of the
    trainer modules beside the package-qualified ones this probe monkeypatches. The mirrored body is
    the same call sequence, and the dataloader is deliberately not prepared (as in main.py).
    """
    n_te = len(artifacts.modules.text_encoders)
    prepared = artifacts.accelerator.prepare(
        artifacts.modules.denoise,
        *artifacts.modules.text_encoders,
        artifacts.denoise_optimizer,
        artifacts.te_optimizer,
    )
    artifacts.modules.denoise = prepared[0]
    artifacts.modules.text_encoders = list(prepared[1 : 1 + n_te])
    artifacts.denoise_optimizer = prepared[1 + n_te]
    artifacts.te_optimizer = prepared[2 + n_te]


def tier_cost(rep: Report, args: argparse.Namespace, work: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    from trainer import control, loop
    from trainer.cache import warm_latent_cache
    from trainer.config import tracker_hparams
    from trainer.device_swap import SwapContext
    from trainer.setup import build_train_objects

    data_dir, dataset = prepare_dataset(work, args.images)
    rep.note(
        f"dataset: {dataset['images']} image(s) ({dataset['copied']} copied, {dataset['reused']} "
        f"already there) + {dataset['captions']} caption(s) from "
        + ", ".join(f"{path} x{count}" for path, count in dataset["sources"].items())
        + f" into {data_dir}"
    )
    cfg = base_config(
        train_data_dir=str(data_dir),
        output_dir=str(work / "outputs"),
        logging_dir=str(work / "logs"),
        output_name="valprobe",
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
    )
    cfg.run_dir = str(work / "outputs" / "valprobe_probe")
    Path(cfg.run_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.logging_dir).mkdir(parents=True, exist_ok=True)
    if args.split_percent <= 0:
        raise SystemExit("refusing to run: --split-percent must be > 0 for this probe")

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
    artifacts.accelerator.init_trackers(project_name="valprobe", config=tracker_hparams(cfg))
    setup_memory = memory_snapshot()
    rep.note(f"memory after the pipeline is built and the cache warmed: {setup_memory}")

    dataset_info = {
        "path": str(data_dir),
        "images": len(artifacts.train_dataset),
        "val_images": artifacts.train_dataset.val_image_count,
        "val_samples": artifacts.train_dataset.val_sample_count,
        "train_samples": artifacts.train_dataset.total_samples,
    }
    rep.check(
        TIER, "the split really left the dataset",
        dataset_info["val_images"] > 0 and 0 < dataset_info["train_samples"],
        f"{dataset_info['val_images']}/{dataset_info['images']} image(s) held out, "
        f"{dataset_info['val_samples']} held-out draw(s)/epoch, {dataset_info['train_samples']} "
        f"train draw(s)/epoch",
    )
    wanted = max(count for _, _, count in arm_plan(args))
    if dataset_info["val_images"] < wanted:
        raise SystemExit(
            f"refusing to run: the split holds out only {dataset_info['val_images']} image(s), so a "
            f"pass could not score the {wanted} the arms ask for. Raise --images (or "
            f"--split-percent)."
        )

    # The fixed sample's picker, on the real held-out paths: the dataset's own answer has to be
    # reproducible from a fresh call, and its cost is a number and not a guess.
    from trainer.validation_split import select_diverse_subset

    held_paths = [Path(record["path"]) for record in artifacts.train_dataset.records
                  if record["is_val"]]
    picker_started = time.perf_counter()
    picked, picker_stats = select_diverse_subset(held_paths, int(cfg.val_sample_count))
    picker_seconds = time.perf_counter() - picker_started
    held_indices = sorted(index for index, record in enumerate(artifacts.train_dataset.records)
                          if record["is_val"])
    expected_fixed = [held_indices[position] for position in picked]
    rep.check(
        TIER, "the fixed sample is reproducible and inside the held-out set",
        expected_fixed == artifacts.train_dataset.fixed_validation_indices()
        and 0 < len(picked) <= min(int(cfg.val_sample_count), len(held_paths)),
        f"picked {picked} ({len(held_paths)} held out, count {int(cfg.val_sample_count)}), "
        f"max pairwise similarity {picker_stats['max_similarity']} %, median "
        f"{picker_stats['median_similarity']} %",
    )
    rep.observe(
        TIER, "cost of picking the fixed sample",
        f"{len(held_paths)} held-out image(s) decoded and compared in {picker_seconds:.2f} s "
        f"(ranks a subset by thumbnail similarity, `tools/cmp_img.py`'s metric on 32x32 signatures)",
        images=len(held_paths),
        seconds=round(picker_seconds, 3),
        max_similarity=picker_stats["max_similarity"],
        median_similarity=picker_stats["median_similarity"],
    )

    held_out = {
        str(record["path"])
        for record in artifacts.train_dataset.records
        if record["is_val"]
    }
    trained = {
        str(record["path"])
        for record in artifacts.train_dataset.records
        if not record["is_val"]
    }
    # Leakage, side A: the real training batches, over two epochs (the sampler reshuffles per epoch).
    walked: set[str] = set()
    sampler = artifacts.dataloader.batch_sampler
    for epoch in (0, 1):
        artifacts.train_dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        for batch in artifacts.dataloader:
            walked |= set(batch["image_path"])
    rep.check(
        TIER, "no held-out image appears in a training batch (two epochs)",
        not (walked & held_out) and bool(walked) and walked <= trained,
        f"{len(walked)} distinct path(s) drawn, {len(held_out)} held out, overlap "
        f"{len(walked & held_out)}",
    )

    instrument = StepInstrument()
    instrument.install(loop)
    snapshot = StateSnapshot(artifacts.modules, [artifacts.denoise_optimizer, artifacts.te_optimizer])
    rep.note(f"state snapshot taken at global step 0; {len(snapshot.params)} parameter group(s)")

    arms = arm_plan(args)
    baseline: ArmResult | None = None
    per_arm: dict[str, Any] = {}
    isolation: dict[str, Any] = {}
    pairs: dict[str, list[ArmResult]] = {}
    try:
        for repeat in range(1, args.repeats + 1):
            for name, interval, count in arms:
                off = measure(
                    artifacts=artifacts, cfg=cfg, loop_module=loop, control=control,
                    swap_ctx=swap_ctx, snapshot=snapshot, arm_name=f"{name}R{repeat}off",
                    interval=0, sample_count=count, steps=args.steps, instrument=instrument,
                )
                on = measure(
                    artifacts=artifacts, cfg=cfg, loop_module=loop, control=control,
                    swap_ctx=swap_ctx, snapshot=snapshot, arm_name=f"{name}R{repeat}on",
                    interval=interval, sample_count=count, steps=args.steps, instrument=instrument,
                )
                pairs.setdefault(name, []).extend([off, on])
                if baseline is None:
                    baseline = off
                identical, first = bit_identical(off.losses, on.losses)
                rep.check(
                    TIER, f"isolation {name} R{repeat}: both arms start from one state",
                    off.start_digest == on.start_digest,
                    f"off={off.start_digest[:16]} on={on.start_digest[:16]}",
                )
                rep.check(
                    TIER, f"isolation {name} R{repeat}: pass on logs the pass-off losses bit for bit",
                    identical == len(off.losses) and bool(off.losses),
                    f"{identical}/{len(off.losses)} step(s) byte-identical, first difference "
                    f"{first}",
                )
                rep.check(
                    TIER, f"isolation {name} R{repeat}: the parameters after the same steps hash equal",
                    off.params_digest == on.params_digest,
                    f"off={off.params_digest[:16]} on={on.params_digest[:16]}",
                )
                expected_points = 1 + (on.steps - 1) // interval if on.steps >= 1 else 0
                if expected_points <= 0:
                    rep.note(
                        f"{name} R{repeat}: interval {interval} exceeds the arm's {on.steps} step(s), "
                        f"so no pass runs in this arm; the pass-side checks are skipped"
                    )
                else:
                    rep.check(
                        TIER, f"leakage {name} R{repeat}: the pass saw held-out images only",
                        bool(on.val_saw_paths) and on.val_saw_paths <= held_out
                        and not (on.val_saw_paths & trained),
                        f"{len(on.val_saw_paths)} distinct pass path(s), "
                        f"{len(on.val_saw_paths - held_out)} outside the held-out set",
                    )
                    rep.check(
                        TIER, f"{name} R{repeat}: no validation forward exceeds train_batch_size",
                        bool(on.val_chunk_sizes)
                        and max(on.val_chunk_sizes) <= int(cfg.train_batch_size),
                        f"{len(on.val_chunk_sizes)} forward(s) inside the passes, largest "
                        f"{max(on.val_chunk_sizes) if on.val_chunk_sizes else 0} image(s) against "
                        f"train_batch_size {int(cfg.train_batch_size)}",
                    )
                    rep.check(
                        TIER, f"{name} R{repeat}: three validation tags, one per cadence step",
                        len(on.val_losses) == expected_points
                        and len(on.val_avg_losses) == expected_points
                        and len(on.val_fixed_losses) == expected_points,
                        f"{len(on.val_losses)} Val/Loss, {len(on.val_avg_losses)} Val/Avg_Loss, "
                        f"{len(on.val_fixed_losses)} Val/Fixed_Loss point(s), expected "
                        f"{expected_points} of each in {on.steps} step(s) at interval {interval}",
                    )
                    # Two passes per cadence step, and the second one is the fixed sample: its path
                    # set has to be byte-for-byte the same at every pass, which is what makes that
                    # curve comparable while the random one covers the held-out set.
                    fixed_passes = on.val_pass_paths[1::2]
                    random_passes = on.val_pass_paths[0::2]
                    rep.check(
                        TIER, f"{name} R{repeat}: the fixed pass scores the same images every time",
                        len(on.val_pass_paths) == 2 * expected_points
                        and bool(fixed_passes)
                        and len(set(fixed_passes)) == 1
                        and all(paths <= held_out for paths in fixed_passes),
                        f"{len(on.val_pass_paths)} pass(es) for {expected_points} cadence step(s); "
                        f"{len(set(fixed_passes))} distinct fixed sample(s)",
                    )
                    # The random pass only has room to move when the held-out set is bigger than the
                    # count; at `count >= pool` it scores every held-out image, exactly like the
                    # fixed pass, and the two tags coincide by construction.
                    if len(held_out) > int(cfg.val_sample_count):
                        rep.check(
                            TIER, f"{name} R{repeat}: the random pass moves between steps",
                            len({tuple(sorted(paths)) for paths in random_passes}) > 1,
                            f"{len({tuple(sorted(paths)) for paths in random_passes})} distinct "
                            f"random subset(s) over {len(random_passes)} pass(es)",
                        )
                    else:
                        rep.note(
                            f"{name} R{repeat}: {len(held_out)} held-out image(s) and "
                            f"val_sample_count {int(cfg.val_sample_count)} mean the random pass "
                            f"scores them all, so `Val/Loss` and `Val/Fixed_Loss` are the same work "
                            f"(one distinct subset, as expected)"
                        )
                rep.check(
                    TIER, f"no held-out image appears in a training batch during the arm ({name} R{repeat})",
                    not (on.train_saw_paths & held_out) and bool(on.train_saw_paths),
                    f"{len(on.train_saw_paths)} distinct training path(s), overlap "
                    f"{len(on.train_saw_paths & held_out)}",
                )
                isolation[f"{name}R{repeat}"] = {
                    "steps_bit_identical": identical,
                    "steps_compared": len(off.losses),
                    "first_differing_step": first,
                    "params_equal": off.params_digest == on.params_digest,
                    "val_points": len(on.val_losses),
                    "val_avg_points": len(on.val_avg_losses),
                    "val_fixed_points": len(on.val_fixed_losses),
                    "passes": len(on.val_pass_paths),
                    "fixed_samples": len({tuple(sorted(paths)) for paths in on.val_pass_paths[1::2]}),
                    "val_points_expected": (1 + (on.steps - 1) // interval) if interval > 0 else 0,
                }
                per_arm.setdefault(name, {"off": [], "on": []})
                per_arm[name]["off"].append(off)
                per_arm[name]["on"].append(on)
                print(
                    f"      [{TIER}] {name} R{repeat}: off "
                    f"{_ms(statistics.median(off.step_times)) if off.step_times else float('nan'):.1f} "
                    f"ms/step, on "
                    f"{_ms(statistics.median(on.step_times)) if on.step_times else float('nan'):.1f} "
                    f"ms/step, pass "
                    f"{_ms(statistics.median(on.val_times)) if on.val_times else float('nan'):.1f} "
                    f"ms, {len(on.val_losses)} val point(s)",
                    flush=True,
                )
            # Second repeat of an arm restores the state again: the pair for each repeat is what is
            # reported, and the medians below pool both.
        for name, arm_interval, arm_count in arms:
            runs = per_arm[name]
            off_times = [value for run in runs["off"] for value in run.step_times]
            on_times = [value for run in runs["on"] for value in run.step_times]
            val_times = [value for run in runs["on"] for value in run.val_times]
            off_median = statistics.median(off_times) if off_times else 0.0
            on_median = statistics.median(on_times) if on_times else 0.0
            # The median answers "what a typical step costs" and hides a coarse cadence's few slow
            # steps, so the cadence overhead is reported off the mean; both are in the report.
            off_mean = statistics.fmean(off_times) if off_times else 0.0
            on_mean = statistics.fmean(on_times) if on_times else 0.0
            first_val = None
            if runs["on"] and runs["on"][0].val_losses:
                first_step = min(runs["on"][0].val_losses)
                first_val = runs["on"][0].val_losses[first_step]
            per_arm[name] = {
                "interval": arm_interval,
                "sample_count": arm_count,
                "pairs": len(runs["on"]),
                "steps_per_run": args.steps,
                "off": {
                    "mean_ms": _ms(off_mean),
                    "median_ms": _ms(off_median),
                    "p90_ms": _ms(_percentile(off_times, 0.9)) if off_times else None,
                },
                "on": {
                    "mean_ms": _ms(on_mean),
                    "median_ms": _ms(on_median),
                    "p90_ms": _ms(_percentile(on_times, 0.9)) if on_times else None,
                },
                "val_pass": {
                    "mean_ms": _ms(statistics.fmean(val_times)) if val_times else None,
                    "median_ms": _ms(statistics.median(val_times)) if val_times else None,
                    "p90_ms": _ms(_percentile(val_times, 0.9)) if val_times else None,
                },
                "mean_overhead_percent": (round(100.0 * (on_mean - off_mean) / off_mean, 1)
                                          if off_mean > 0 else None),
                "median_overhead_percent": (round(100.0 * (on_median - off_median) / off_median, 1)
                                            if off_median > 0 else None),
                "val_loss_first": first_val,
                "memory_off": runs["off"][0].memory_summary() if runs["off"] else {},
                "memory_on": runs["on"][0].memory_summary() if runs["on"] else {},
            }
            rep.observe(
                TIER, f"cost {name}",
                f"interval {arm_interval}, {arm_count} image(s)/pass: off "
                f"{per_arm[name]['off']['mean_ms']:.1f} ms/step mean -> on "
                f"{per_arm[name]['on']['mean_ms']:.1f} ms/step mean (median "
                f"{per_arm[name]['on']['median_ms']:.1f}), pass itself "
                f"{_fmt_or_none(per_arm[name]['val_pass']['mean_ms'])}",
                interval=arm_interval,
                sample_count=arm_count,
                mean_overhead_percent=per_arm[name]["mean_overhead_percent"],
                median_overhead_percent=per_arm[name]["median_overhead_percent"],
            )
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

    # The events are the artifact a reader can check: Val/Loss has to be in them, at the arm's cadence.
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    logged: dict[str, int] = {}
    for run_dir in sorted(Path(cfg.logging_dir).glob("*")):
        accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
        try:
            accumulator.Reload()
        except Exception:  # noqa: BLE001 - an empty/unreadable dir is not a check failure here
            continue
        for tag in (SERIES_TAG, VAL_TAG):
            if tag in accumulator.Tags().get("scalars", []):
                logged[tag] = logged.get(tag, 0) + len(accumulator.Scalars(tag))
    rep.check(
        TIER, "the events carry Val/Loss beside Train/Loss",
        logged.get(VAL_TAG, 0) > 0 and logged.get(SERIES_TAG, 0) > 0,
        f"{logged.get(SERIES_TAG, 0)} Train/Loss point(s), {logged.get(VAL_TAG, 0)} Val/Loss point(s) "
        f"in {cfg.logging_dir}",
    )
    return (
        {
            "dataset": dataset_info,
            "arms": per_arm,
            "isolation": isolation,
            "logged": logged,
            "memory_after_setup": setup_memory,
            "baseline_digest": baseline.params_digest[:16] if baseline else None,
        },
        {},
    )


# --------------------------------------------------------------------------------------
# argv / report
# --------------------------------------------------------------------------------------


def launcher_alloc_conf() -> str:
    """`PYTORCH_CUDA_ALLOC_CONF` exactly as `start_train.sh` sets it for a real run.

    Parity, not a fix: the probe should allocate the way a training process does, so a cost number
    measured here is a number that run would see. Read out of the launcher rather than copied, so
    the two cannot drift.
    """
    try:
        text = (REPO_ROOT / "start_train.sh").read_text(encoding="utf-8")
    except OSError:
        return ""
    match = re.search(r'PYTORCH_CUDA_ALLOC_CONF="?([^"\n]+)"?', text)
    return match.group(1) if match else ""


def maybe_reexec_under_hook(args: argparse.Namespace, rep: Report, argv_tail: list[str]) -> None:
    """Run this process under the allocation patch `[environment].amdfq` selects.

    The gfx1201 Tensile over-read described in `doc/troubleshooting.md` kills a bare process
    outright (`HSA_STATUS_ERROR_MEMORY_FAULT`, no traceback). The child-run probes retry after such
    a death; an in-process measurement cannot, so it starts under the patch instead — the same
    `start_hook.sh` line `start_train.sh` preloads on this machine, which leaves the measured step
    times on the path the repo actually trains with. `--no-hook` skips the re-exec and falls back to
    `HSA_SVM_GUARD_PAGES=0` (the documented dodge that removes the page the over-read lands on); the
    report says which of the two was in effect.
    """
    if args.no_hook:
        os.environ.setdefault("HSA_SVM_GUARD_PAGES", "0")
        rep.note("--no-hook: no allocation patch; HSA_SVM_GUARD_PAGES=0 is this run's fault dodge")
        return
    if os.environ.get(HOOK_MARKER) == "1":
        rep.note(f"running under the allocation patch: LD_PRELOAD={os.environ.get('LD_PRELOAD', '')}")
        return
    from trainer.amdfq_patch import launch_env_line

    try:
        line = launch_env_line()
    except FileNotFoundError as exc:
        os.environ.setdefault("HSA_SVM_GUARD_PAGES", "0")
        rep.note(f"no allocation patch to preload ({exc}); HSA_SVM_GUARD_PAGES=0 is the fault dodge")
        return
    if line.split("|", 1)[0] not in ("tail", "vmm"):
        os.environ.setdefault("HSA_SVM_GUARD_PAGES", "0")
        rep.note("config.toml selects no allocation patch (amdfq = none); HSA_SVM_GUARD_PAGES=0 is "
                 "the fault dodge")
        return
    hook = REPO_ROOT / "start_hook.sh"
    print(f"== re-executing under {hook} ({line.split('|', 1)[0]})", flush=True)
    env = dict(os.environ)
    env[HOOK_MARKER] = "1"
    os.execve(str(hook), [str(hook), "--", sys.executable, str(Path(__file__).resolve()), *argv_tail],
              env)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=20,
                        help="optimizer steps per measured run (default 20)")
    parser.add_argument("--repeats", type=int, default=2,
                        help="how many (pass-off, pass-on) pairs per arm (default 2)")
    parser.add_argument("--images", type=int, default=80,
                        help="images copied from the configured folders (default 80, so a 10%% "
                             "split holds out enough to score --sample-count of them)")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--split-percent", type=float, default=10.0)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--intervals", type=int, nargs="+", default=[1, 5, 10],
                        help="val_interval per arm (default 1 5 10; 5 is the shipped one)")
    parser.add_argument("--count-ladder", type=int, nargs="*", default=[1],
                        help="extra val_sample_count arms at interval 1 (default 1)")
    parser.add_argument("--report-dir", type=str, default=None)
    parser.add_argument("--work-dir", type=str, default=None)
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
        "probe": "val-loss",
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
    path = out / "report.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Val-loss probe",
        "",
        f"prediction: {PREDICTION}",
        f"verdict: {verdict}",
        f"seconds: {payload['seconds']}",
        "",
        f"isolation: {json.dumps(details.get('isolation') or {}, indent=2, default=str)}",
        "",
        "## cost (median ms per optimizer step; pass = the validation pass itself)",
        "",
        "| arm | interval | images/pass | off mean ms/step | on mean ms/step | pass mean ms | mean overhead | median overhead |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, arm in (details.get("arms") or {}).items():
        lines.append(
            f"| {name} | {arm.get('interval')} | {arm.get('sample_count')} | "
            f"{arm['off']['mean_ms']} | {arm['on']['mean_ms']} | "
            f"{arm['val_pass']['mean_ms']} | {arm.get('mean_overhead_percent')}% | "
            f"{arm.get('median_overhead_percent')}% |"
        )
    lines += ["", "## checks", ""]
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
                  else Path("/tmp/axl-probe-val-loss") / time.strftime("%Y%m%d_%H%M%S"))
    work = Path(args.work_dir) if args.work_dir else report_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    args.report_dir, args.work_dir = report_dir, work
    rep = Report(report_dir)
    print(f"== val-loss probe | report={report_dir}", flush=True)
    print(f"   prediction: {PREDICTION}", flush=True)

    maybe_reexec_under_hook(args, rep, argv_tail)
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
    rep.note(f"arms: {arm_plan(args)}; {args.repeats} (pass-off, pass-on) pair(s) each, "
             f"{args.steps} steps per run")

    details: dict[str, Any] = {}
    verdict = "not run"
    try:
        details, _ = tier_cost(rep, args, work)
        isolation = details.get("isolation") or {}
        pairs = [value for value in isolation.values()]
        clean = bool(pairs) and all(
            value["steps_bit_identical"] == value["steps_compared"] and value["params_equal"]
            for value in pairs
        )
        verdict = (
            "the validation pass left every compared training step bit-identical and all parameters "
            "hash-equal" if clean else
            "the validation pass CHANGED training: see the isolation checks for the first differing step"
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
