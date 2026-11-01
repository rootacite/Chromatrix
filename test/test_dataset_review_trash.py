import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

import api


def _image(path: Path, size=(16, 12)) -> Path:
    Image.new("RGB", size, (10, 20, 30)).save(path, "JPEG", quality=95)
    return path


class DatasetReviewTrashTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        self.out = self.root / "out"
        self.trash = self.root / "trash"
        self.data.mkdir()
        self.out.mkdir()
        os.environ["AXL_TRASH_DIR"] = str(self.trash)
        self.cfg = {
            "train_data_dir": str(self.data),
            "output_dir": str(self.out),
            "output_name": "rein",
        }
        self._orig = api._train_config_dict
        api._train_config_dict = lambda: dict(self.cfg)

    def tearDown(self):
        api._train_config_dict = self._orig
        self.tmp.cleanup()
        os.environ.pop("AXL_TRASH_DIR", None)

    def _write_group(self, stem="alpha"):
        image = _image(self.data / f"{stem}.jpg")
        (self.data / f"{stem}.txt").write_text("red, hat", encoding="utf-8")
        Image.new("L", (16, 12), 255).save(self.data / f"{stem}.mask.png")
        return image

    def test_review_trash_moves_group_and_lists_it(self):
        self._write_group()
        moved = api.dispatch(
            "dataset_trash",
            {"directory": str(self.data), "stem": "alpha"},
        )
        self.assertEqual(moved["stem"], "alpha")
        item = moved["item"]
        self.assertEqual(item["tags"], ["red", "hat"])
        self.assertTrue(item["has_sidecar_mask"])
        self.assertTrue(Path(item["image"]).is_file())
        self.assertTrue(Path(item["txt"]).is_file())
        self.assertTrue(Path(item["mask"]).is_file())

        listing = api.dispatch("dataset_list", {"directory": str(self.data)})
        self.assertEqual(listing["items"], [])
        self.assertEqual(listing["orphans"], [])

        trashed = api.dispatch("dataset_trash_list", {"directory": str(self.data)})
        self.assertEqual(len(trashed["items"]), 1)
        listed = trashed["items"][0]
        self.assertEqual(listed["trash_id"], item["trash_id"])
        self.assertEqual(listed["tags"], ["red", "hat"])

        blob = api.dispatch(
            "blob_batch",
            {
                "paths": [listed["image"]],
                "max_edge": 32,
                "quality": 80,
                "format": "jpeg",
            },
        )
        self.assertNotIn("error", blob["items"][0])

    def test_restore_puts_the_whole_group_back(self):
        self._write_group("beta")
        moved = api.dispatch(
            "dataset_trash",
            {"directory": str(self.data), "stem": "beta"},
        )
        restored = api.dispatch(
            "dataset_restore",
            {"directory": str(self.data), "trash_id": moved["item"]["trash_id"]},
        )
        self.assertEqual(restored["stem"], "beta")
        self.assertTrue((self.data / "beta.jpg").is_file())
        self.assertTrue((self.data / "beta.txt").is_file())
        self.assertTrue((self.data / "beta.mask.png").is_file())
        self.assertEqual(
            api.dispatch("dataset_trash_list", {"directory": str(self.data)})["items"],
            [],
        )
        listing = api.dispatch("dataset_list", {"directory": str(self.data)})
        self.assertEqual(len(listing["items"]), 1)
        self.assertEqual(listing["items"][0]["tags"], ["red", "hat"])

    def test_restore_never_overwrites_a_new_file(self):
        self._write_group("gamma")
        moved = api.dispatch(
            "dataset_trash",
            {"directory": str(self.data), "stem": "gamma"},
        )
        _image(self.data / "gamma.jpg")
        with self.assertRaises(ValueError):
            api.dispatch(
                "dataset_restore",
                {"directory": str(self.data), "trash_id": moved["item"]["trash_id"]},
            )
        self.assertEqual(
            len(api.dispatch("dataset_trash_list", {"directory": str(self.data)})["items"]),
            1,
        )

    def test_trash_inside_a_dataset_is_refused(self):
        os.environ["AXL_TRASH_DIR"] = str(self.data / "trash")
        self._write_group("delta")
        with self.assertRaises(ValueError) as ctx:
            api.dispatch(
                "dataset_trash",
                {"directory": str(self.data), "stem": "delta"},
            )
        self.assertIn("inside train data", str(ctx.exception))
        self.assertTrue((self.data / "delta.jpg").is_file())

    def test_trash_list_is_scoped_to_the_selected_folder(self):
        self._write_group("one")
        api.dispatch("dataset_trash", {"directory": str(self.data), "stem": "one"})
        other = self.root / "other"
        other.mkdir()
        self.cfg["train_data"] = [
            {"path": str(self.data), "repeat": 1},
            {"path": str(other), "repeat": 1},
        ]
        self.cfg["train_data_dir"] = str(self.data)
        self.assertEqual(
            api.dispatch("dataset_trash_list", {"directory": str(other)})["items"],
            [],
        )


if __name__ == "__main__":
    unittest.main()
