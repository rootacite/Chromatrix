"""The validation-set split: which images leave training, and that they really do.

Three layers, deliberately including a probe rather than a reading of the source:

1. `validation_split.held_out_count` / `select_validation`: torch-free arithmetic, `ceil` and
   determinism.
2. `LoraImageDataset`: the held-out images are out of the training buckets, in `val_buckets`, and
   still carry records (so the warm cache encodes them).
3. The two-sided leakage probe: no held-out image ever appears in a training batch — three epochs,
   and the sampler reshuffles per epoch — and the validation pass is only ever handed held-out
   images. Both directions are checked on real batches from a real dataloader.
"""

import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

# `python test/test_validation_split.py` has to import the repo's own packages, exactly like
# `unittest discover -s test` does from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer import control
from trainer.config import TrainConfig
from trainer.dataset import LoraImageDataset
from trainer.setup import build_dataloader
from trainer.validation_split import (
    held_out_count,
    select_diverse_subset,
    select_validation,
    signature_similarity,
)


class _MockLatentDist:
    def __init__(self, pooled):
        self._pooled = pooled

    def sample(self, generator=None):
        return self._pooled


class MockVAE(torch.nn.Module):
    """Deterministic stand-in: latent = pooled pixels, the shape a real cache entry holds."""

    def __init__(self):
        super().__init__()
        self.config = type("Cfg", (), {"scaling_factor": 0.5})()

    def encode(self, x):
        pooled = x[:, :1, ::8, ::8].expand(-1, 4, -1, -1)
        return type("Enc", (), {"latent_dist": _MockLatentDist(pooled * 2.0)})()


def _write_images(root: Path, count: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        Image.new("RGB", (512, 512), (i * 20, 40, 80)).save(root / f"img_{i:03d}.png")
        (root / f"img_{i:03d}.txt").write_text("tag_a, tag_b", encoding="utf-8")


def _cfg(folders: list[tuple[Path, int]], **overrides) -> TrainConfig:
    cfg = TrainConfig()
    cfg.train_data = [{"path": str(path), "repeat": repeat} for path, repeat in folders]
    cfg.train_data_dir = str(folders[0][0])
    cfg.enable_bucket = False
    cfg.train_resolution = 512
    cfg.cache_latents = True
    cfg.cache_latents_to_disk = True
    cfg.train_batch_size = 2
    cfg.max_data_loader_n_workers = 0
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


class ValSplitConfigTest(unittest.TestCase):
    """The three `[training]` keys: read from a mapping, and each range refused by name."""

    def test_the_keys_are_read_and_zero_interval_is_a_valid_off(self):
        cfg = TrainConfig.from_mapping(
            {"val_split_percent": 25.5, "val_sample_count": 3, "val_interval": 0}
        )
        self.assertEqual(cfg.val_split_percent, 25.5)
        self.assertEqual(cfg.val_sample_count, 3)
        self.assertEqual(cfg.val_interval, 0)

    def test_out_of_range_values_name_the_key(self):
        cases = [
            ({"val_split_percent": 91.0}, "val_split_percent"),
            ({"val_split_percent": -1.0}, "val_split_percent"),
            ({"val_sample_count": 0}, "val_sample_count"),
            ({"val_sample_count": 65}, "val_sample_count"),
            ({"val_interval": -1}, "val_interval"),
            # A hand-edited non-integer is refused rather than truncated to a whole number.
            ({"val_sample_count": 5.5}, "val_sample_count"),
            ({"val_interval": "ten"}, "val_interval"),
        ]
        for mapping, key in cases:
            with self.subTest(mapping=mapping):
                with self.assertRaises(ValueError) as ctx:
                    TrainConfig.from_mapping(mapping)
                self.assertIn(key, str(ctx.exception))

    def test_the_bounds_are_accepted(self):
        cfg = TrainConfig.from_mapping(
            {"val_split_percent": 0.0, "val_sample_count": 1, "val_interval": 100000}
        )
        self.assertEqual(cfg.val_split_percent, 0.0)
        cfg = TrainConfig.from_mapping({"val_split_percent": 90.0, "val_sample_count": 64})
        self.assertEqual(cfg.val_split_percent, 90.0)


class DiverseSubsetTest(unittest.TestCase):
    """The fixed sample's picker: mutually dissimilar, deterministic, and hard to fool."""

    def _structured(self, index: int, size=(256, 256)) -> Image.Image:
        """A gradient plus `index + 1` discs, so distinct indices differ like photos do."""
        width, height = size
        array = np.zeros((height, width, 3), dtype=np.uint8)
        array[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
        array[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
        array[:, :, 2] = (index * 37) % 256
        image = Image.fromarray(array)
        draw = ImageDraw.Draw(image)
        for step in range(index + 1):
            draw.ellipse(
                [20 + step * 9, 30 + step * 7, 90 + step * 9, 120 + step * 7],
                fill=(255 - index * 20, 40, 200 - index * 15),
            )
        return image

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for index in range(6):
            self._structured(index).save(self.root / f"img_{index:02d}.png")
        for copy in range(3):  # exact copies of img_00
            self._structured(0).save(self.root / f"dup_{copy:02d}.png")
        self.paths = sorted(self.root.glob("*.png"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_it_never_picks_two_copies_of_one_image(self):
        chosen, stats = select_diverse_subset(self.paths, 4)
        names = [self.paths[index].name for index in chosen]
        copies = [name for name in names if name.startswith("dup")]
        self.assertLessEqual(len(copies), 1, f"picked copies {names}")
        self.assertEqual(4, len(chosen))
        # The stat is what the run's log line reports: a set holding two copies would read 100.
        self.assertLess(stats["max_similarity"], 100.0)

    def test_the_same_folder_contents_always_pick_the_same_images(self):
        self.assertEqual(
            select_diverse_subset(self.paths, 4)[0],
            select_diverse_subset(self.paths, 4)[0],
        )

    def test_the_pick_is_spread_out_not_just_the_first_ones(self):
        """Every candidate is measured against what is already chosen, not taken in name order."""
        chosen, _ = select_diverse_subset(self.paths, 4)
        # img_00 is byte-identical to the picked dup_*; both must not be in the set, and the pick
        # must not simply be the four lowest indices (which would be the three dups + img_00).
        self.assertNotEqual(chosen, [0, 1, 2, 3])

    def test_a_count_at_or_above_the_pool_takes_everything(self):
        chosen, stats = select_diverse_subset(self.paths, 99)
        self.assertEqual(list(range(len(self.paths))), chosen)
        self.assertEqual(len(self.paths), stats["count"])
        self.assertEqual(100.0, stats["max_similarity"])  # the copies are in there by definition

    def test_nothing_to_pick_is_an_empty_answer(self):
        self.assertEqual(([], {"count": 0, "pool": 0, "max_similarity": None,
                               "median_similarity": None, "unreadable": 0}),
                         select_diverse_subset([], 5))
        self.assertEqual([], select_diverse_subset(self.paths, 0)[0])

    def test_differently_sized_images_compare(self):
        """`tools/cmp_img.py` refuses pairs whose shapes differ; the signatures do not."""
        Image.new("RGB", (800, 300), (10, 200, 30)).save(self.root / "wide.png")
        chosen, stats = select_diverse_subset(sorted(self.root.glob("*.png")), 3)
        self.assertEqual(3, len(chosen))
        self.assertEqual(0, stats["unreadable"])
        left = np.asarray(Image.open(self.paths[0]).convert("RGB").resize((32, 32)),
                          dtype=np.float32).reshape(-1)
        right = np.asarray(Image.open(self.root / "wide.png").convert("RGB").resize((32, 32)),
                           dtype=np.float32).reshape(-1)
        self.assertGreaterEqual(signature_similarity(left, right), 0.0)

    def test_an_unreadable_image_is_counted_and_kept_out_of_the_candidates(self):
        broken = self.root / "broken.png"
        broken.write_bytes(b"not a png")
        paths = sorted(self.root.glob("*.png"))
        chosen, stats = select_diverse_subset(paths, 4)
        self.assertEqual(1, stats["unreadable"])
        self.assertNotIn(paths.index(broken), chosen)


class DatasetFixedSubsetTest(unittest.TestCase):
    """`LoraImageDataset.fixed_validation_indices`: same images every pass, inside the split."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["AXL_RUNTIME_DIR"] = self.tmp.name
        control._state = {}
        control._last_cmd_seq = 0
        control._ended = False
        control.release_lock()
        self.root = Path(self.tmp.name) / "data"
        self.root.mkdir()
        for index in range(20):
            Image.new("RGB", (512, 512), (index * 11, 60, 200 - index * 7)).save(
                self.root / f"img_{index:02d}.png")
            (self.root / f"img_{index:02d}.txt").write_text("tag", encoding="utf-8")
        for copy in range(10):  # copies of img_00, so a held-out pair can be near-identical
            Image.new("RGB", (512, 512), (0, 60, 200)).save(self.root / f"copy_{copy:02d}.png")

    def tearDown(self):
        control.release_lock()
        self.tmp.cleanup()
        os.environ.pop("AXL_RUNTIME_DIR", None)

    def _dataset(self, **overrides):
        options = {"val_split_percent": 40.0, "val_sample_count": 8, "seed": 31}
        options.update(overrides)
        cfg = TrainConfig()
        cfg.train_data = [{"path": str(self.root), "repeat": 1}]
        cfg.train_data_dir = str(self.root)
        cfg.enable_bucket = False
        cfg.cache_latents = False
        cfg.cache_latents_to_disk = False
        for key, value in options.items():
            setattr(cfg, key, value)
        return LoraImageDataset(cfg)

    def _held_out(self, ds):
        return {index for index, record in enumerate(ds.records) if record["is_val"]}

    def test_the_fixed_sample_is_inside_the_split_and_sized_by_the_count(self):
        ds = self._dataset()
        held = self._held_out(ds)
        fixed = ds.fixed_validation_indices()
        self.assertEqual(min(8, len(held)), len(fixed))
        self.assertTrue(set(fixed) <= held)
        self.assertEqual(len(set(fixed)), len(fixed))
        self.assertEqual(sorted(fixed), fixed)

    def test_two_datasets_over_the_same_folders_pick_the_same_samples(self):
        self.assertEqual(
            self._dataset().fixed_validation_indices(),
            self._dataset().fixed_validation_indices(),
        )

    def test_the_fixed_sample_avoids_two_copies_of_one_image(self):
        ds = self._dataset()
        held = self._held_out(ds)
        copies = {index for index in held if ds.records[index]["path"].name.startswith("copy_")}
        self.assertGreaterEqual(len(copies), 2, "the fixture's split should hold out two copies")
        picked_copies = copies & set(ds.fixed_validation_indices())
        self.assertLessEqual(len(picked_copies), 1, "the fixed sample kept two copies")

    def test_the_random_series_still_redraws_while_the_fixed_one_does_not(self):
        ds = self._dataset()
        fixed = ds.fixed_validation_indices()
        draws = [ds.sample_validation_indices(8, step) for step in (1, 6, 11, 16)]
        self.assertEqual([fixed] * 4, [ds.fixed_validation_indices() for _ in range(4)])
        self.assertGreater(len({tuple(draw) for draw in draws}), 1, "the random draws never moved")
        for draw in draws:
            self.assertTrue(set(draw) <= self._held_out(ds))


class HeldOutCountTest(unittest.TestCase):
    def test_rounds_up_the_share_of_unique_images(self):
        self.assertEqual(held_out_count(20, 10.0), 2)
        self.assertEqual(held_out_count(15, 10.0), 2)
        self.assertEqual(held_out_count(10, 10.0), 1)
        self.assertEqual(held_out_count(3, 10.0), 1)

    def test_never_takes_every_image(self):
        self.assertEqual(held_out_count(5, 90.0), 4)
        self.assertEqual(held_out_count(2, 90.0), 1)
        self.assertEqual(held_out_count(1, 90.0), 0)
        self.assertEqual(held_out_count(0, 50.0), 0)

    def test_zero_or_negative_is_off(self):
        self.assertEqual(held_out_count(100, 0.0), 0)
        self.assertEqual(held_out_count(100, -5.0), 0)


class SelectValidationTest(unittest.TestCase):
    def _folders(self):
        return [
            [Path(f"/data/a/{i:03d}.png") for i in range(12)],
            [Path(f"/data/b/{i:03d}.png") for i in range(8)],
        ]

    def test_names_the_rounded_up_count_of_positions(self):
        folders = self._folders()
        held = select_validation(folders, 10.0, 7)
        self.assertEqual(len(held), held_out_count(20, 10.0))
        for folder, position in held:
            self.assertLess(position, len(folders[folder]))

    def test_the_same_seed_and_percent_pick_the_same_images(self):
        self.assertEqual(
            select_validation(self._folders(), 25.0, 11),
            select_validation(self._folders(), 25.0, 11),
        )

    def test_a_different_seed_picks_a_different_set(self):
        self.assertNotEqual(
            select_validation(self._folders(), 10.0, 1),
            select_validation(self._folders(), 10.0, 2),
        )

    def test_a_toml_integer_and_a_parsed_float_seed_match(self):
        self.assertEqual(
            select_validation(self._folders(), 10, 3),
            select_validation(self._folders(), 10.0, 3.0),
        )

    def test_zero_percent_holds_nothing_out(self):
        self.assertEqual(select_validation(self._folders(), 0.0, 5), set())


class DatasetValidationSplitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["AXL_RUNTIME_DIR"] = self.tmp.name
        control._state = {}
        control._last_cmd_seq = 0
        control._ended = False
        control.release_lock()
        root = Path(self.tmp.name)
        self.first = root / "first"  # 12 images, drawn 3x each
        self.second = root / "second"  # 6 images, drawn once
        _write_images(self.first, 12)
        _write_images(self.second, 6)

    def tearDown(self):
        control.release_lock()
        self.tmp.cleanup()
        os.environ.pop("AXL_RUNTIME_DIR", None)

    def _dataset(self, **overrides) -> LoraImageDataset:
        defaults = {"val_split_percent": 10.0, "seed": 4242}
        defaults.update(overrides)
        cfg = _cfg([(self.first, 3), (self.second, 1)], **defaults)
        return LoraImageDataset(cfg)

    def _held_out_indices(self, ds: LoraImageDataset) -> set[int]:
        return {index for index, record in enumerate(ds.records) if record["is_val"]}

    def test_the_split_is_the_complement_and_the_total_shrinks_by_its_draws(self):
        ds = self._dataset()
        held = self._held_out_indices(ds)
        # 18 unique images, 10% -> ceil(1.8) = 2 held out.
        self.assertEqual(ds.val_image_count, 2)
        self.assertEqual(len(held), 2)
        self.assertEqual(ds.val_sample_count, sum(ds.records[i]["repeat"] for i in held))

        trained = Counter({i: ds.records[i]["repeat"] for i in range(len(ds.records)) if i not in held})
        drawn = Counter({i: ds.records[i]["repeat"] for i in range(len(ds.records))})
        self.assertEqual(Counter(self._bucket_indices(ds.buckets)), trained)
        self.assertEqual(Counter(self._bucket_indices(ds.val_buckets)), {i: 1 for i in held})
        self.assertEqual(ds.total_samples, sum(drawn.values()) - ds.val_sample_count)

    def _bucket_indices(self, buckets) -> list[int]:
        return [index for indices in buckets.values() for index in indices]

    def test_len_and_records_keep_every_image(self):
        ds = self._dataset()
        self.assertEqual(len(ds), 18)
        self.assertEqual(len(ds.records), 18)
        self.assertEqual([record["path"] for record in ds.records], sorted(ds.images))

    def test_three_epochs_never_draw_a_held_out_image(self):
        """The probe: real batches from the real loader, every epoch, both directions."""
        cfg = _cfg([(self.first, 3), (self.second, 1)], val_split_percent=10.0, seed=4242)
        ds, dataloader = build_dataloader(cfg)
        held = self._held_out_indices(ds)
        held_paths = {str(ds.records[i]["path"]) for i in held}
        train_paths = {str(ds.records[i]["path"]) for i in range(len(ds.records)) if i not in held}
        held_cache_paths = {str(self._cache_path(ds, i)) for i in held}
        self.assertEqual(len(held_paths), ds.val_image_count)

        for epoch in range(3):
            ds.set_epoch(epoch)
            dataloader.batch_sampler.set_epoch(epoch)
            for batch in dataloader:
                paths = set(batch["image_path"])
                self.assertEqual(paths & held_paths, set(), f"a held-out image trained in epoch {epoch}")
                self.assertTrue(paths <= train_paths, f"an unknown image trained in epoch {epoch}")
                self.assertEqual(set(batch["cache_path"]) & held_cache_paths, set())

        # Every remaining image is drawn exactly `repeat` times in one epoch: the split is not a loss.
        dataloader.batch_sampler.set_epoch(0)
        drawn: Counter = Counter()
        for batch in dataloader.batch_sampler:
            drawn.update(batch)
        expected = Counter(
            {index: ds.records[index]["repeat"] for index in range(len(ds.records)) if index not in held}
        )
        self.assertEqual(drawn, expected)

    def _cache_path(self, ds: LoraImageDataset, index: int) -> Path:
        record = ds.records[index]
        return ds._cache_path(record["path"], record["geom"], record["cache_dir"])

    def test_the_validation_pass_only_ever_sees_held_out_images(self):
        ds = self._dataset(val_sample_count=5)
        held = self._held_out_indices(ds)
        held_paths = {str(ds.records[i]["path"]) for i in held}
        self.assertEqual(len(held), 2)  # fewer than the requested five: the pass takes what exists

        for step in range(1, 6):
            indices = ds.sample_validation_indices(5, step)
            self.assertEqual(len(indices), len(set(indices)))
            self.assertTrue(set(indices) <= held)
            self.assertEqual({str(ds[i]["image_path"]) for i in indices}, held_paths)

    def test_the_pass_is_capped_at_the_requested_count(self):
        ds = self._dataset(val_split_percent=50.0, val_sample_count=3)
        self.assertEqual(ds.val_image_count, 9)
        for step in range(1, 4):
            self.assertEqual(len(ds.sample_validation_indices(3, step)), 3)
        self.assertEqual(len(ds.sample_validation_indices(100, 1)), 9)

    def test_no_held_out_image_is_still_in_the_warm_cache_plan(self):
        """The pass reuses the cached latent, so a held-out image must still be encoded."""
        ds = self._dataset()
        held = self._held_out_indices(ds)
        self.assertEqual(len(ds.cache_entries()), len(ds.records))
        held_cache_paths = {str(self._cache_path(ds, index)) for index in held}
        plan = {str(entry["cache_path"]) for entry in ds.cache_entries()}
        self.assertEqual(held_cache_paths & plan, held_cache_paths)

        from trainer.cache import warm_latent_cache

        finished = warm_latent_cache(ds, MockVAE(), ds.cfg, torch.device("cpu"), torch.float32)
        self.assertTrue(finished)
        for index in held:
            item = ds[index]
            self.assertEqual(item["img_type"], "latent", "the held-out image was not cached")

    def test_zero_percent_trains_everything(self):
        ds = self._dataset(val_split_percent=0.0)
        self.assertEqual(ds.val_image_count, 0)
        self.assertEqual(ds.val_sample_count, 0)
        self.assertEqual(ds.total_samples, 42)  # 12*3 + 6*1
        self.assertEqual(ds.sample_validation_indices(5, 1), [])

    def test_changing_the_dataset_content_redraws_the_split(self):
        """A different folder content is a different split; the seed alone does not pin images."""
        cfg = _cfg([(self.first, 1)], val_split_percent=50.0, seed=9)
        before = {str(r["path"]) for r in LoraImageDataset(cfg).records if r["is_val"]}
        _write_images(self.first, 20)
        after = {str(r["path"]) for r in LoraImageDataset(cfg).records if r["is_val"]}
        self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()
