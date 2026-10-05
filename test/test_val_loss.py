"""`loop.compute_validation_loss`: what one validation pass is allowed to touch.

The pass is extra GPU work inside the training step, so its claims are checked against a stub run
rather than read out of the source: it draws only held-out images, it runs without gradients, it
leaves the parameters, the optimizer state and the global RNG exactly as it found them, it restores
the module modes it switched, and `_maybe_log_and_sample` writes `Val/Loss` only when a pass
actually produced one.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from PIL import Image

# `python test/test_val_loss.py` has to import the repo's own packages, exactly like
# `unittest discover -s test` does from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer import control, loop
from trainer.config import TrainConfig
from trainer.control import LiveSettings
from trainer.dataset import LoraImageDataset
from trainer.loss_log import LossRecorder, synthesize_avg_loss


class _MockLatentDist:
    def __init__(self, pooled):
        self._pooled = pooled

    def sample(self, generator=None):
        return self._pooled


class MockVAE(torch.nn.Module):
    """Deterministic stand-in: latent = pooled pixels, shaped like a real VAE's output."""

    def __init__(self):
        super().__init__()
        self.config = type("Cfg", (), {"scaling_factor": 0.5})()

    def encode(self, x):
        pooled = x[:, :1, ::8, ::8].expand(-1, 4, -1, -1)
        return type("Enc", (), {"latent_dist": _MockLatentDist(pooled * 2.0)})()


class _RecordingFamily:
    """Counts and records each `compute_loss` call; the loss is a function of the group size."""

    def __init__(self, per_group=None):
        self.per_group = per_group if per_group is not None else (lambda size: 2.0)
        self.calls: list[list[str]] = []
        self.grad_enabled: list[bool] = []

    def extra_cond(self, *, src_wh, bucket_wh, device, dtype):
        return {}

    def compute_loss(self, *, prompts, latents, extra, modules, cfg, device, dtype):
        self.calls.append(list(prompts))
        self.grad_enabled.append(torch.is_grad_enabled())
        return torch.tensor(self.per_group(len(prompts)), dtype=torch.float32)


class _StubAccelerator:
    is_main_process = True

    def __init__(self):
        self.logged: list[tuple[dict, int]] = []

    def log(self, scalars, step=None):
        self.logged.append((dict(scalars), step))


def _optimizer_state(optimizer) -> dict:
    return {
        key: {name: value.detach().clone() for name, value in bucket.items() if torch.is_tensor(value)}
        for key, bucket in optimizer.state.items()
    }


class ValidationCadenceTest(unittest.TestCase):
    """`loop.validation_due`: the pass starts at step 1 and repeats every `val_interval`."""

    def test_the_first_pass_is_step_one_then_every_interval(self):
        self.assertEqual([1, 6, 11, 16, 21], [step for step in range(1, 25) if loop.validation_due(5, step)])
        self.assertEqual([1, 11, 21, 31], [step for step in range(1, 40) if loop.validation_due(10, step)])

    def test_interval_one_validates_every_step(self):
        self.assertEqual(list(range(1, 8)), [step for step in range(1, 8) if loop.validation_due(1, step)])

    def test_zero_never_validates(self):
        self.assertEqual([], [step for step in range(0, 30) if loop.validation_due(0, step)])

    def test_step_zero_is_not_a_step(self):
        # `global_step` is incremented before the check, so 0 is only a guard: it must not read as a
        # pass for interval 1 either.
        self.assertFalse(loop.validation_due(1, 0))
        self.assertFalse(loop.validation_due(5, 0))


class ValidationPassTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["AXL_RUNTIME_DIR"] = self.tmp.name
        control._state = {}
        control._last_cmd_seq = 0
        control._ended = False
        control.release_lock()
        self.folder = Path(self.tmp.name) / "data"
        self.folder.mkdir()
        for i in range(8):
            Image.new("RGB", (512, 512), (i * 20, 40, 80)).save(self.folder / f"img_{i:03d}.png")
            (self.folder / f"img_{i:03d}.txt").write_text(f"cap_{i:03d}", encoding="utf-8")
        self.cfg = TrainConfig()
        self.cfg.train_data = [{"path": str(self.folder), "repeat": 1}]
        self.cfg.train_data_dir = str(self.folder)
        # The repo's own validation folder would replace the split these tests are about.
        self.cfg.val_data_dir = ""
        self.cfg.enable_bucket = False
        self.cfg.train_resolution = 512
        self.cfg.cache_latents = False
        self.cfg.cache_latents_to_disk = False
        self.cfg.shuffle_caption = False
        self.cfg.max_data_loader_n_workers = 0
        self.cfg.val_split_percent = 25.0  # 8 images -> ceil(2) = 2 held out
        self.cfg.val_sample_count = 5
        self.cfg.val_interval = 1

    def tearDown(self):
        control.release_lock()
        self.tmp.cleanup()
        os.environ.pop("AXL_RUNTIME_DIR", None)

    def _artifacts(self, family, **cfg_overrides):
        for key, value in cfg_overrides.items():
            setattr(self.cfg, key, value)
        dataset = LoraImageDataset(self.cfg)
        denoise = torch.nn.Linear(4, 4)
        te = torch.nn.Linear(4, 4)
        # A real optimizer with real state, so "left untouched" is a comparison and not a no-op.
        optimizer = torch.optim.Adam(denoise.parameters(), lr=0.1)
        denoise(torch.zeros(1, 4)).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        te.train(False)  # an eval module must stay eval after the pass
        return SimpleNamespace(
            accelerator=_StubAccelerator(),
            denoise_optimizer=optimizer,
            te_optimizer=optimizer,
            settings=LiveSettings(save_every_n_steps=0, sampling_enabled=False),
            family=family,
            modules=SimpleNamespace(denoise=denoise, text_encoders=[te], vae=MockVAE()),
            device=torch.device("cpu"),
            weight_dtype=torch.float32,
            train_dataset=dataset,
        )

    def _held_out_captions(self, dataset: LoraImageDataset) -> set[str]:
        return {
            dataset.records[index]["path"].stem.replace("img", "cap")
            for index, record in enumerate(dataset.records)
            if record["is_val"]
        }

    def test_only_held_out_images_reach_the_pass(self):
        family = _RecordingFamily()
        artifacts = self._artifacts(family)

        loss = loop.compute_validation_loss(artifacts=artifacts, cfg=self.cfg, global_step=1)

        self.assertEqual(loss, 2.0)
        held = self._held_out_captions(artifacts.train_dataset)
        self.assertEqual(len(held), 2)
        recorded = {prompt for call in family.calls for prompt in call}
        self.assertEqual(recorded, held)

    def test_the_pass_is_capped_at_the_requested_count(self):
        family = _RecordingFamily()
        self.cfg.val_sample_count = 1
        artifacts = self._artifacts(family)

        loop.compute_validation_loss(artifacts=artifacts, cfg=self.cfg, global_step=1)

        self.assertEqual(sum(len(call) for call in family.calls), 1)
        held = self._held_out_captions(artifacts.train_dataset)
        self.assertTrue({prompt for call in family.calls for prompt in call} <= held)

    def test_the_average_is_weighted_by_group_size(self):
        family = _RecordingFamily(per_group=lambda size: float(size))
        artifacts = self._artifacts(family)

        loss = loop.compute_validation_loss(artifacts=artifacts, cfg=self.cfg, global_step=1)

        sizes = [len(call) for call in family.calls]
        self.assertEqual(sizes, [2])  # one bucket, both held-out images, train_batch_size fits both
        self.assertAlmostEqual(loss, sum(size * size for size in sizes) / sum(sizes))

    def test_one_forward_never_carries_more_than_the_training_batch(self):
        """A bucket group bigger than `train_batch_size` is scored in chunks."""
        family = _RecordingFamily(per_group=lambda size: float(size))
        self.cfg.train_batch_size = 1
        artifacts = self._artifacts(family)
        self.assertEqual(2, artifacts.train_dataset.val_image_count)

        loss = loop.compute_validation_loss(artifacts=artifacts, cfg=self.cfg, global_step=1)

        sizes = [len(call) for call in family.calls]
        self.assertEqual(sizes, [1, 1], "a group of 2 must go through as two forwards of 1")
        self.assertLessEqual(max(sizes), self.cfg.train_batch_size)
        # The aggregate is still the image-count-weighted mean, not the mean of the chunk means.
        self.assertAlmostEqual(loss, (1.0 * 1 + 1.0 * 1) / 2)

    def test_chunking_does_not_change_how_the_passes_are_averaged(self):
        """Same images, different cap: the aggregate formula is the one the doc states."""
        family = _RecordingFamily(per_group=lambda size: float(size))
        self.cfg.train_batch_size = 2
        artifacts = self._artifacts(family)

        loss = loop.compute_validation_loss(artifacts=artifacts, cfg=self.cfg, global_step=1)

        self.assertEqual([2], [len(call) for call in family.calls])
        self.assertAlmostEqual(loss, 2.0)  # (2 * 2) / 2, the same weighted mean with one chunk

    def test_the_pass_runs_without_gradients(self):
        family = _RecordingFamily()
        artifacts = self._artifacts(family)

        loop.compute_validation_loss(artifacts=artifacts, cfg=self.cfg, global_step=1)

        self.assertTrue(family.calls)
        self.assertTrue(all(flag is False for flag in family.grad_enabled))
        self.assertTrue(all(p.grad is None for p in artifacts.modules.denoise.parameters()))

    def test_parameters_optimizer_state_module_modes_and_the_rng_are_unchanged(self):
        family = _RecordingFamily()
        artifacts = self._artifacts(family)
        before_params = {name: p.detach().clone() for name, p in artifacts.modules.denoise.named_parameters()}
        before_state = _optimizer_state(artifacts.denoise_optimizer)
        before_rng = torch.get_rng_state().clone()

        loop.compute_validation_loss(artifacts=artifacts, cfg=self.cfg, global_step=1)

        for name, param in artifacts.modules.denoise.named_parameters():
            self.assertTrue(torch.equal(param.detach(), before_params[name]), f"{name} moved")
        after_state = _optimizer_state(artifacts.denoise_optimizer)
        self.assertEqual(set(after_state), set(before_state))
        for key, bucket in before_state.items():
            for name, value in bucket.items():
                self.assertTrue(torch.equal(after_state[key][name], value), f"optimizer state {key}.{name} moved")
        self.assertTrue(torch.equal(torch.get_rng_state(), before_rng), "the pass consumed global RNG")
        self.assertTrue(artifacts.modules.denoise.training)
        self.assertFalse(artifacts.modules.text_encoders[0].training)

    def test_nothing_is_drawn_without_a_split(self):
        family = _RecordingFamily()
        self.cfg.val_split_percent = 0.0
        artifacts = self._artifacts(family)

        loss = loop.compute_validation_loss(artifacts=artifacts, cfg=self.cfg, global_step=1)

        self.assertIsNone(loss)
        self.assertEqual(family.calls, [])

    def test_a_stop_during_the_pass_reports_no_loss(self):
        family = _RecordingFamily()
        artifacts = self._artifacts(family)
        control.request("stop")

        loss = loop.compute_validation_loss(artifacts=artifacts, cfg=self.cfg, global_step=1)

        self.assertIsNone(loss)
        self.assertTrue(artifacts.modules.denoise.training)


class RunValidationPassesTest(unittest.TestCase):
    """The two passes one cadence step runs, and the epoch-window average behind `Val/Avg_Loss`."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["AXL_RUNTIME_DIR"] = self.tmp.name
        control._state = {}
        control._last_cmd_seq = 0
        control._ended = False
        control.release_lock()
        loop._val_loss_recorder = LossRecorder()
        self.folder = Path(self.tmp.name) / "data"
        self.folder.mkdir()
        for i in range(20):
            Image.new("RGB", (512, 512), (i * 11, 40, 200 - i * 7)).save(
                self.folder / f"img_{i:03d}.png")
            (self.folder / f"img_{i:03d}.txt").write_text(f"cap_{i:03d}", encoding="utf-8")
        self.cfg = TrainConfig()
        self.cfg.train_data = [{"path": str(self.folder), "repeat": 1}]
        self.cfg.train_data_dir = str(self.folder)
        # The repo's own validation folder would replace the split these tests are about.
        self.cfg.val_data_dir = ""
        self.cfg.enable_bucket = False
        self.cfg.train_resolution = 512
        self.cfg.cache_latents = False
        self.cfg.cache_latents_to_disk = False
        self.cfg.shuffle_caption = False
        self.cfg.max_data_loader_n_workers = 0
        self.cfg.val_split_percent = 50.0  # 20 images -> 10 held out, more than the count
        self.cfg.val_sample_count = 3
        self.cfg.val_interval = 1

    def tearDown(self):
        control.release_lock()
        self.tmp.cleanup()
        os.environ.pop("AXL_RUNTIME_DIR", None)

    def _dataset(self) -> LoraImageDataset:
        return LoraImageDataset(self.cfg)

    def _artifacts(self, dataset: LoraImageDataset):
        return SimpleNamespace(train_dataset=dataset)

    def test_one_call_runs_a_random_pass_then_the_fixed_one(self):
        dataset = self._dataset()
        calls: list[tuple[list[int] | None, int]] = []

        def fake_loss(*, artifacts, cfg, global_step, swap_ctx=None, indices=None):
            calls.append((None if indices is None else list(indices), int(global_step)))
            return 1.0 + 0.1 * len(calls)

        fixed = dataset.fixed_validation_indices()
        with mock.patch.object(loop, "compute_validation_loss", side_effect=fake_loss):
            first = loop.run_validation_passes(
                artifacts=self._artifacts(dataset), cfg=self.cfg, global_step=1,
                epoch_index=0, val_point_in_epoch=0,
            )
            second = loop.run_validation_passes(
                artifacts=self._artifacts(dataset), cfg=self.cfg, global_step=6,
                epoch_index=0, val_point_in_epoch=1,
            )

        # Pass order and arguments: `None` means "this step's random subset", then the fixed list —
        # the same list at both steps, which is what makes the fixed curve comparable.
        self.assertEqual([None, fixed, None, fixed], [entry[0] for entry in calls])
        self.assertEqual([1, 1, 6, 6], [entry[1] for entry in calls])
        self.assertEqual(3, len(fixed))
        for got, want in zip(first, (1.1, 1.1, 1.2)):
            self.assertAlmostEqual(got, want)
        # Second step: pass 1 brings 1.3, so the window holds 1.1 and 1.3.
        for got, want in zip(second, (1.3, 1.2, 1.4)):
            self.assertAlmostEqual(got, want)

    def test_the_random_pass_redraws_and_the_fixed_one_does_not(self):
        dataset = self._dataset()
        draws = [dataset.sample_validation_indices(3, step) for step in (1, 6, 11)]
        self.assertGreater(len({tuple(draw) for draw in draws}), 1)
        self.assertEqual(dataset.fixed_validation_indices(), dataset.fixed_validation_indices())
        for draw in draws:
            self.assertTrue(set(draw) <= {index for index, record in enumerate(dataset.records)
                                          if record["is_val"]})

    def test_the_average_is_the_epoch_window_the_train_curve_uses(self):
        """`Val/Avg_Loss` against `synthesize_avg_loss`: one rule, two series."""
        dataset = self._dataset()
        values = [0.5, 0.7, 0.6, 0.9, 0.4]
        averages: list[float] = []
        # Each step asks twice (the random pass, then the fixed one); the pair carries the step's
        # value, and only the first of the two feeds the average.
        calls = {"n": 0}

        def fake_loss(**_kwargs):
            value = values[min(calls["n"] // 2, len(values) - 1)]
            calls["n"] += 1
            return value

        with mock.patch.object(loop, "compute_validation_loss", side_effect=fake_loss):
            for index, _ in enumerate(values):
                _, average, _ = loop.run_validation_passes(
                    artifacts=self._artifacts(dataset), cfg=self.cfg, global_step=1 + index,
                    epoch_index=0, val_point_in_epoch=index,
                )
                averages.append(average)

        points = [{"step": index + 1, "value": value} for index, value in enumerate(values)]
        expected = [point["value"] for point in synthesize_avg_loss(points)]
        for got, want in zip(averages, expected):
            self.assertAlmostEqual(got, want)

    def test_the_window_wraps_when_the_next_epoch_starts(self):
        """A later epoch overwrites the same intra-epoch index, exactly like `LossRecorder`."""
        dataset = self._dataset()
        values = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
        averages: list[float] = []
        plan = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        calls = {"n": 0}

        def fake_loss(**_kwargs):
            value = values[min(calls["n"] // 2, len(values) - 1)]
            calls["n"] += 1
            return value

        with mock.patch.object(loop, "compute_validation_loss", side_effect=fake_loss):
            for step, (epoch_index, point_index) in enumerate(plan):
                _, average, _ = loop.run_validation_passes(
                    artifacts=self._artifacts(dataset), cfg=self.cfg, global_step=1 + step,
                    epoch_index=epoch_index, val_point_in_epoch=point_index,
                )
                averages.append(average)

        recorder = LossRecorder()
        recorder.add(epoch=0, step=0, loss=1.0)
        self.assertAlmostEqual(averages[0], recorder.moving_average)
        # After the first point of the second epoch the window holds 2.0, 3.0, 5.0: the overwritten
        # value replaces its own slot rather than starting a fresh list.
        self.assertAlmostEqual(averages[3], (2.0 + 3.0 + 5.0) / 3)
        self.assertAlmostEqual(averages[4], (2.0 + 4.0 + 5.0) / 3)
        self.assertAlmostEqual(averages[5], (2.0 + 4.0 + 6.0) / 3)


class ValLossLoggingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["AXL_RUNTIME_DIR"] = self.tmp.name
        control._state = {}
        control._last_cmd_seq = 0
        control._ended = False
        control.release_lock()

    def tearDown(self):
        control.release_lock()
        self.tmp.cleanup()
        os.environ.pop("AXL_RUNTIME_DIR", None)

    def _artifacts(self):
        denoise = SimpleNamespace(parameters=lambda: [])
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(2))], lr=0.1)
        return SimpleNamespace(
            accelerator=_StubAccelerator(),
            denoise_optimizer=optimizer,
            te_optimizer=optimizer,
            settings=LiveSettings(save_every_n_steps=0, sampling_enabled=False),
            family=SimpleNamespace(),
            modules=SimpleNamespace(denoise=denoise, text_encoders=[], vae=None),
            device=torch.device("cpu"),
            weight_dtype=torch.float32,
            train_dataset=SimpleNamespace(),
        )

    def test_val_loss_is_logged_in_the_same_event_as_the_training_scalars(self):
        artifacts = self._artifacts()
        loop._maybe_log_and_sample.last_loss = 0.5
        loop._maybe_log_and_sample.last_avg_loss = 0.4

        loop._maybe_log_and_sample(
            artifacts=artifacts, cfg=TrainConfig(), global_step=7, val_loss=1.25
        )

        self.assertEqual(len(artifacts.accelerator.logged), 1)
        scalars, step = artifacts.accelerator.logged[0]
        self.assertEqual(step, 7)
        self.assertEqual(scalars["Train/Loss"], 0.5)
        self.assertEqual(scalars["Train/Avg_Loss"], 0.4)
        self.assertEqual(scalars["Val/Loss"], 1.25)

    def test_all_three_validation_scalars_ride_one_event(self):
        artifacts = self._artifacts()
        loop._maybe_log_and_sample.last_loss = 0.5
        loop._maybe_log_and_sample.last_avg_loss = 0.4

        loop._maybe_log_and_sample(
            artifacts=artifacts, cfg=TrainConfig(), global_step=7,
            val_loss=1.25, val_avg_loss=1.1, val_fixed_loss=1.4,
        )

        self.assertEqual(len(artifacts.accelerator.logged), 1, "the curves must share one x")
        scalars, _ = artifacts.accelerator.logged[0]
        self.assertEqual(scalars["Val/Loss"], 1.25)
        self.assertEqual(scalars["Val/Avg_Loss"], 1.1)
        self.assertEqual(scalars["Val/Fixed_Loss"], 1.4)

    def test_no_val_loss_key_when_the_step_did_not_validate(self):
        artifacts = self._artifacts()
        loop._maybe_log_and_sample.last_loss = 0.5
        loop._maybe_log_and_sample.last_avg_loss = 0.4

        loop._maybe_log_and_sample(artifacts=artifacts, cfg=TrainConfig(), global_step=7)

        scalars, _ = artifacts.accelerator.logged[0]
        for tag in ("Val/Loss", "Val/Avg_Loss", "Val/Fixed_Loss"):
            self.assertNotIn(tag, scalars)


if __name__ == "__main__":
    unittest.main()
