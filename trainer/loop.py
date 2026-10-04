from __future__ import annotations

import contextlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

try:
    from config import TrainConfig
    from loss_log import LossRecorder
    from env import flush_memory
    import control
    from control import LiveSettings
    from dataset import collate_fn
    from device_swap import SwapContext, at_safe_point
    from family import FamilyModules, ModelFamily
    from models import artifact_root, lora_checkpoint_file
    from setup import TrainArtifacts
except ImportError:
    from trainer.config import TrainConfig
    from trainer.loss_log import LossRecorder
    from trainer.env import flush_memory
    from trainer import control
    from trainer.control import LiveSettings
    from trainer.dataset import collate_fn
    from trainer.device_swap import SwapContext, at_safe_point
    from trainer.family import FamilyModules, ModelFamily
    from trainer.models import artifact_root, lora_checkpoint_file
    from trainer.setup import TrainArtifacts

_loss_recorder = LossRecorder()
# `Val/Avg_Loss`: the same epoch-window rule as `Train/Avg_Loss`, fed with the validation points
# instead of the training steps. Module-level like `_loss_recorder`; a process that runs two runs
# in a row (or a test that drives `train_one_epoch` twice) resets it the same way.
_val_loss_recorder = LossRecorder()


def _scheduled_lr(optimizer: Any) -> float:
    """The LR Schedule-Free last applied (`scheduled_lr`, its warmup ramp included)."""
    group = optimizer.param_groups[0]
    return group.get("scheduled_lr", group["lr"])


@contextlib.contextmanager
def optimizers_eval(optimizers: Iterable[Any]):
    """Run the body with every optimizer in eval mode, i.e. with the parameters at the averaged `x`.

    Schedule-Free keeps them at the training iterate `y`; `x` is only there after `eval()`, so
    anything that reads the weights rather than stepping them — a checkpoint, and the samples drawn
    from it — has to be written inside this. Both are put back in the `finally`, whatever the save
    or the sample does. `optimizer_eval` is the one implementation of that: the step cadence save in
    `_maybe_log_and_sample` and `main.py`'s final save both use it.
    """
    switched = [opt for opt in optimizers if hasattr(opt, "eval") and hasattr(opt, "train")]
    for optimizer in switched:
        optimizer.eval()
    try:
        yield
    finally:
        for optimizer in switched:
            optimizer.train()


def save_stopped_lora(artifacts: TrainArtifacts, cfg: TrainConfig, global_step: int) -> bool:
    """Write the checkpoint a stop during training owes, if this step has none.

    Same rule as the cadence save: the weights are read rather than stepped, so it runs inside
    `optimizers_eval`. Without that the file would hold the training iterate `y` — the mode the loop
    leaves behind — instead of the averaged `x` every other checkpoint is written from.
    """
    if global_step <= 0 or lora_checkpoint_file(cfg, global_step).is_file():
        return False
    with optimizers_eval((artifacts.denoise_optimizer, artifacts.te_optimizer)):
        artifacts.family.save_lora(artifacts.accelerator, artifacts.modules, cfg, global_step)
    return True


def _batch_int(values: Any, idx: int) -> int:
    item = values[idx]
    if torch.is_tensor(item):
        return int(item.item())
    return int(item)


def group_indices_by_bucket(batch: dict[str, Any]) -> dict[tuple[int, int], list[int]]:
    """Group batch items by spatial bucket to keep tensor shapes consistent."""
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx in range(len(batch["caption"])):
        bw = _batch_int(batch["bucket_w"], idx)
        bh = _batch_int(batch["bucket_h"], idx)
        groups[(bw, bh)].append(idx)
    return groups


def encode_latent_for_item(
    *,
    item_index: int,
    batch: dict[str, Any],
    vae: torch.nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    """Load a latent directly or encode a pixel image on demand."""
    img_type = batch["img_type"][item_index]
    cache_path = Path(batch["cache_path"][item_index])
    img_data = batch["img_data"][item_index]
    if img_type == "latent":
        return img_data.to(device=device, dtype=weight_dtype, non_blocking=True)

    pixel_values = img_data.unsqueeze(0).to(device=device, dtype=weight_dtype)
    with torch.no_grad():
        latent = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
    latent = latent.squeeze(0)

    if cfg.cache_latents and cfg.cache_latents_to_disk:
        torch.save(latent.detach().cpu(), cache_path)

    return latent


def _stack_extra(extras: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not extras or not extras[0]:
        return {}
    return {key: torch.stack([item[key] for item in extras], dim=0) for key in extras[0]}


def build_group_inputs(
    *,
    indices: list[int],
    batch: dict[str, Any],
    family: ModelFamily,
    vae: torch.nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> tuple[list[str], torch.Tensor, dict[str, torch.Tensor]]:
    """Build prompts, latents, and family-specific extra cond for one bucket group."""
    prompts = [batch["caption"][i] for i in indices]
    extras: list[dict[str, torch.Tensor]] = []
    for i in indices:
        extras.append(
            family.extra_cond(
                src_wh=(_batch_int(batch["src_w"], i), _batch_int(batch["src_h"], i)),
                bucket_wh=(_batch_int(batch["bucket_w"], i), _batch_int(batch["bucket_h"], i)),
                device=device,
                dtype=weight_dtype,
            )
        )

    stacked = _stack_extra(extras)
    batch_masks = batch.get("loss_mask")
    if batch_masks is not None:
        if torch.is_tensor(batch_masks):
            stacked["loss_mask"] = batch_masks[indices].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
        else:
            stacked["loss_mask"] = torch.stack(
                [batch_masks[i] for i in indices], dim=0
            ).to(device=device, dtype=torch.float32)

    img_data = batch["img_data"]
    if torch.is_tensor(img_data):
        latents = img_data[indices].to(device=device, dtype=weight_dtype, non_blocking=True)
        return prompts, latents, stacked

    latents_list: list[torch.Tensor] = []
    for i in indices:
        latents_list.append(
            encode_latent_for_item(
                item_index=i,
                batch=batch,
                vae=vae,
                cfg=cfg,
                device=device,
                weight_dtype=weight_dtype,
            )
        )
    latents = torch.stack(latents_list, dim=0).to(device=device, dtype=weight_dtype)
    return prompts, latents, stacked


def validation_due(interval: int, global_step: int) -> bool:
    """Whether `global_step` runs a validation pass.

    The first pass lands on **step 1** and then every `interval`: 1, 1+N, 1+2N, … A modulo rule
    would put the first point at step N, which is late for a short run and hides the step where the
    loss is most comparable with the start. `interval = 0` never validates.
    """
    steps = int(interval)
    step = int(global_step)
    return steps > 0 and step >= 1 and (step - 1) % steps == 0


def compute_validation_loss(
    *,
    artifacts: TrainArtifacts,
    cfg: TrainConfig,
    global_step: int,
    swap_ctx: SwapContext | None = None,
    indices: list[int] | None = None,
) -> float | None:
    """Mean held-out loss for one validation pass, or `None` when nothing was drawn or a stop came.

    Reuses the training loss path (`build_group_inputs` + `family.compute_loss`) on the held-out
    images named by `indices` — `None` means this step's random subset
    (`dataset.sample_validation_indices`), the loop passes the fixed sample for the second pass —
    averaged over the images actually scored (never over the requested count, so a small held-out
    set is not diluted with zeros).

    `train_batch_size` is also the cap on one forward: a bucket group bigger than that is scored in
    chunks, so a pass on a 16 GB card can never peak above a training step of the same run. The
    chunking changes the aggregation not at all (each chunk weighs by its image count) and the
    per-image noise/timestep draws only in the sense that they come from the chunk's own forward.

    The pass must leave training untouched, and three things do that: `torch.no_grad()` (no
    gradient, no `.grad`), a forked RNG around the *whole* pass including `dataset.__getitem__`
    (a cold latent cache draws its VAE posterior sample from the global generator, and the
    timestep/noise draws of `denoise_loss` do too, so without the fork a validation point would
    shift the training run's noise sequence), and `eval()` on the modules for the pass, restored in
    the `finally` (dropout off, gradient checkpointing off). The optimizers are deliberately not
    touched: Schedule-Free holds the parameters at the training iterate while training, and the
    validation loss is meant to be comparable with the same step's `Train/Loss`.
    """
    dataset = artifacts.train_dataset
    if indices is None:
        indices = dataset.sample_validation_indices(cfg.val_sample_count, global_step)
    indices = list(indices)
    if not indices:
        return None
    device = artifacts.device
    modules = artifacts.modules
    switched = [module for module in [modules.denoise, *modules.text_encoders] if hasattr(module, "training")]
    modes = [bool(module.training) for module in switched]
    rng_devices: list[int] = [] if str(getattr(device, "type", "cpu")) == "cpu" else [device.index or 0]
    loss_sum = 0.0
    item_count = 0
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=rng_devices):
            for module in switched:
                module.eval()
            batch = collate_fn([dataset[index] for index in indices])
            # One forward per bucket group, and never more images in it than a training step's own
            # forward: a group is chunked at `train_batch_size`, so a pass cannot peak above the
            # step it rides on (8 held-out images in one bucket used to go through as one batch 8).
            # The aggregate stays the mean over the scored images: each chunk weighs by its size.
            chunk_size = max(1, int(cfg.train_batch_size))
            for group in group_indices_by_bucket(batch).values():
                for start in range(0, len(group), chunk_size):
                    chunk = group[start : start + chunk_size]
                    # A pass is short, but a pause or stop between chunks should not wait for the rest.
                    if not at_safe_point("training", swap_ctx):
                        return None
                    prompts, latents, extra = build_group_inputs(
                        indices=chunk,
                        batch=batch,
                        family=artifacts.family,
                        vae=modules.vae,
                        cfg=cfg,
                        device=device,
                        weight_dtype=artifacts.weight_dtype,
                    )
                    loss = artifacts.family.compute_loss(
                        prompts=prompts,
                        latents=latents,
                        extra=extra,
                        modules=modules,
                        cfg=cfg,
                        device=device,
                        dtype=artifacts.weight_dtype,
                    )
                    loss_sum += float(loss.detach().item()) * len(chunk)
                    item_count += len(chunk)
    finally:
        for module, was_training in zip(switched, modes):
            module.train(was_training)
    if item_count == 0:
        return None
    return loss_sum / item_count


def run_validation_passes(
    *,
    artifacts: TrainArtifacts,
    cfg: TrainConfig,
    global_step: int,
    epoch_index: int,
    val_point_in_epoch: int,
    swap_ctx: SwapContext | None = None,
) -> tuple[float | None, float | None, float | None]:
    """Both validation passes for one cadence step: `(Val/Loss, Val/Avg_Loss, Val/Fixed_Loss)`.

    Pass 1 scores this step's random subset — coverage over the run — and its mean both goes to
    `Val/Loss` and feeds the epoch-window recorder behind `Val/Avg_Loss` (the same rule
    `Train/Avg_Loss` uses). Pass 2 scores the fixed, mutually dissimilar sample
    (`dataset.fixed_validation_indices`), the same images every pass, so that curve is comparable
    point to point. Each pass runs its own `compute_validation_loss`, hence its own no-grad /
    forked-RNG / eval-mode sandwich: the isolation the probe asserts holds per pass, not only for
    the pair.
    """
    val_loss = compute_validation_loss(
        artifacts=artifacts,
        cfg=cfg,
        global_step=global_step,
        swap_ctx=swap_ctx,
    )
    val_avg_loss: float | None = None
    if val_loss is not None:
        _val_loss_recorder.add(epoch=epoch_index, step=val_point_in_epoch, loss=val_loss)
        val_avg_loss = _val_loss_recorder.moving_average
    val_fixed_loss = compute_validation_loss(
        artifacts=artifacts,
        cfg=cfg,
        global_step=global_step,
        swap_ctx=swap_ctx,
        indices=artifacts.train_dataset.fixed_validation_indices(),
    )
    return val_loss, val_avg_loss, val_fixed_loss


def _maybe_log_and_sample(
    *,
    artifacts: TrainArtifacts,
    cfg: TrainConfig,
    global_step: int,
    swap_ctx: SwapContext | None = None,
    val_loss: float | None = None,
    val_avg_loss: float | None = None,
    val_fixed_loss: float | None = None,
) -> None:
    """Log the step, then save a checkpoint (and, when the switch is on, its samples).

    Sampling has no cadence of its own: a sample always belongs to the checkpoint written in the
    same step, so turning sampling off makes a save point checkpoint-only. The three `val_*` scalars
    are logged in the same event as the training scalars, so every curve shares one x in the chart.
    """
    accelerator = artifacts.accelerator
    if accelerator.is_main_process:
        denoise_optimizer = artifacts.denoise_optimizer
        te_optimizer = artifacts.te_optimizer
        unet_effective_lr = _scheduled_lr(denoise_optimizer)
        te_effective_lr = _scheduled_lr(te_optimizer)

        scalars = {
            "Train/Loss": _maybe_log_and_sample.last_loss,
            "Train/Avg_Loss": _maybe_log_and_sample.last_avg_loss,
            "UNet/LR/Effective_Actual_LR": unet_effective_lr,
            "TE/LR/Effective_Actual_LR": te_effective_lr,
        }
        # Three validation scalars ride the same event, so every curve shares one x: the random
        # subset's raw mean (`Val/Loss`), its epoch-window average (`Val/Avg_Loss`, the same
        # `LossRecorder` rule as `Train/Avg_Loss`), and the fixed sample's raw mean
        # (`Val/Fixed_Loss`).
        if val_loss is not None:
            scalars["Val/Loss"] = float(val_loss)
        if val_avg_loss is not None:
            scalars["Val/Avg_Loss"] = float(val_avg_loss)
        if val_fixed_loss is not None:
            scalars["Val/Fixed_Loss"] = float(val_fixed_loss)
        accelerator.log(scalars, step=global_step)

        settings = artifacts.settings
        if settings.due(global_step):
            # The checkpoint and the samples drawn from it have to be written with both optimizers
            # in eval mode (see `optimizers_eval`): they hold y while training, x is the averaged one.
            with optimizers_eval((denoise_optimizer, te_optimizer)):
                artifacts.family.save_lora(
                    accelerator, artifacts.modules, cfg, global_step
                )
                if settings.sampling_enabled:
                    artifacts.family.generate_sample(
                        accelerator=accelerator,
                        modules=artifacts.modules,
                        cfg=cfg,
                        device=artifacts.device,
                        dtype=artifacts.weight_dtype,
                        global_step=global_step,
                        output_dir_base=artifact_root(cfg),
                        swap_ctx=swap_ctx,
                    )
            settings.mark_saved(global_step)
            control.publish_settings(settings)

_maybe_log_and_sample.last_loss = 0.0
_maybe_log_and_sample.last_avg_loss = 0.0


def adopt_live_settings(artifacts: TrainArtifacts, global_step: int) -> None:
    """Take over a cadence / sampling switch requested while the run was going.

    Called once per optimizer step, after `at_safe_point` so a change made while paused applies
    to the step the run resumes with.
    """
    adopted = artifacts.settings.adopt(control.read_settings(), global_step)
    if adopted is artifacts.settings:
        return
    artifacts.settings = adopted
    control.publish_settings(adopted)


def train_one_epoch(
    *,
    artifacts: TrainArtifacts,
    cfg: TrainConfig,
    global_step: int,
    progress,
    total_train_steps: int,
    swap_ctx: SwapContext | None = None,
) -> int:
    """Train one epoch and keep all step-based actions aligned with optimizer steps."""
    accelerator = artifacts.accelerator
    modules: FamilyModules = artifacts.modules
    family = artifacts.family
    denoise = modules.denoise
    text_encoders = modules.text_encoders
    denoise_optimizer = artifacts.denoise_optimizer
    te_optimizer = artifacts.te_optimizer
    device = artifacts.device
    weight_dtype = artifacts.weight_dtype

    denoise.train()
    if hasattr(denoise_optimizer, "train"):
        denoise_optimizer.train()
    if hasattr(te_optimizer, "train"):
        te_optimizer.train()
    for te in text_encoders:
        te.train()

    denoise_clip_params = [p for p in denoise.parameters() if p.requires_grad]
    te_clip_params = [
        p
        for te in text_encoders
        for p in te.parameters()
        if p.requires_grad
    ]

    epoch_step = 0
    epoch_index = max(0, int(cfg._current_epoch) - 1)
    # Validation points already emitted inside this epoch: the index `_val_loss_recorder` overwrites
    # next. Counting points rather than dividing the step by the cadence keeps the window a full
    # epoch's worth when the epoch length is not a multiple of `val_interval`.
    val_point_in_epoch = 0

    for batch in artifacts.dataloader:
        with accelerator.accumulate(denoise, *text_encoders):
            groups = group_indices_by_bucket(batch)

            batch_loss_sum: torch.Tensor | None = None
            batch_item_count = 0
            n_caption = len(batch["caption"])

            for _, indices in groups.items():
                prompts, latents, extra = build_group_inputs(
                    indices=indices,
                    batch=batch,
                    family=family,
                    vae=modules.vae,
                    cfg=cfg,
                    device=device,
                    weight_dtype=weight_dtype,
                )

                loss = family.compute_loss(
                    prompts=prompts,
                    latents=latents,
                    extra=extra,
                    modules=modules,
                    cfg=cfg,
                    device=device,
                    dtype=weight_dtype,
                )

                scaled_loss = loss * (len(indices) / n_caption)
                accelerator.backward(scaled_loss)
                weighted = loss.detach() * len(indices)
                batch_loss_sum = weighted if batch_loss_sum is None else batch_loss_sum + weighted
                batch_item_count += len(indices)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(denoise_clip_params, cfg.unet_max_grad_norm)
                accelerator.clip_grad_norm_(te_clip_params, cfg.te_max_grad_norm)
                denoise_optimizer.step()
                te_optimizer.step()
                denoise_optimizer.zero_grad(set_to_none=True)
                te_optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            global_step += 1
            if batch_loss_sum is None:
                avg_loss = 0.0
            else:
                avg_loss = float((batch_loss_sum / max(1, batch_item_count)).item())
            _loss_recorder.add(epoch=epoch_index, step=epoch_step, loss=avg_loss)
            _maybe_log_and_sample.last_loss = avg_loss
            _maybe_log_and_sample.last_avg_loss = _loss_recorder.moving_average
            epoch_step += 1

            if progress is not None:
                progress.update(1)
                progress.set_description(
                    f"epoch={cfg._current_epoch}/{cfg.epoch} step={global_step} loss={avg_loss:.4f}"
                )

            control.set_training(
                step=global_step,
                total_steps=total_train_steps,
                epoch=int(cfg._current_epoch),
                epochs=cfg.epoch,
                loss=float(avg_loss),
                avg_loss=float(_loss_recorder.moving_average),
            )
            if not at_safe_point("training", swap_ctx):
                return global_step

            adopt_live_settings(artifacts, global_step)
            val_loss: float | None = None
            val_avg_loss: float | None = None
            val_fixed_loss: float | None = None
            if validation_due(cfg.val_interval, global_step) and artifacts.train_dataset.val_image_count > 0:
                val_loss, val_avg_loss, val_fixed_loss = run_validation_passes(
                    artifacts=artifacts,
                    cfg=cfg,
                    global_step=global_step,
                    epoch_index=epoch_index,
                    val_point_in_epoch=val_point_in_epoch,
                    swap_ctx=swap_ctx,
                )
                if val_loss is not None:
                    val_point_in_epoch += 1
            _maybe_log_and_sample(
                artifacts=artifacts,
                cfg=cfg,
                global_step=global_step,
                swap_ctx=swap_ctx,
                val_loss=val_loss,
                val_avg_loss=val_avg_loss,
                val_fixed_loss=val_fixed_loss,
            )
            if control.should_stop():
                return global_step

        if cfg.flush_memory_every_step:
            flush_memory(device)

    return global_step
