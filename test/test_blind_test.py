"""Tests for Blind Test rating engine, persistence, and consistency metrics."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import api
from trainer.checkpoints import (
    blind_test_path,
    read_blind_test_state,
    reset_blind_test_state,
    write_blind_test_state,
)


class BlindTestBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_dir = Path(self.temp_dir.name)
        self.run_id = "test_run_001"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_read_write_reset_state(self):
        state = read_blind_test_state(self.log_dir)
        self.assertEqual(state["ratings"], {})
        self.assertEqual(state["history"], [])
        self.assertEqual(state["metrics"]["total_comparisons"], 0)

        state["ratings"]["cp_1"] = {"rating": 1550.0, "uncertainty": 320.0, "wins": 1, "losses": 0, "matches": 1}
        write_blind_test_state(self.log_dir, self.run_id, state)

        reloaded = read_blind_test_state(self.log_dir)
        self.assertEqual(reloaded["run_id"], self.run_id)
        self.assertIn("cp_1", reloaded["ratings"])
        self.assertEqual(reloaded["ratings"]["cp_1"]["rating"], 1550.0)

        reset_blind_test_state(self.log_dir, self.run_id)
        reset_state = read_blind_test_state(self.log_dir)
        self.assertEqual(reset_state["ratings"], {})
        self.assertEqual(reset_state["history"], [])

    def test_handle_blind_test_record_and_ratings(self):
        with patch.object(api, "_train_config_dict", return_value={"output_dir": str(self.log_dir)}), \
             patch.object(api, "_resolve_run", return_value=(self.run_id, "test_output")), \
             patch.object(api, "_log_dir", return_value=self.log_dir):

            # Initial ratings query
            res = api.handle_blind_test_ratings({})
            self.assertEqual(res["run_id"], self.run_id)
            self.assertEqual(res["ratings"], {})

            # Record match: cp_A vs cp_B, Left (cp_A) wins on profile 0
            rec_res = api.handle_blind_test_record({
                "left_checkpoint": "/path/cp_A.safetensors",
                "right_checkpoint": "/path/cp_B.safetensors",
                "profile_index": 0,
                "chosen": "left",
                "is_canary": False,
            })

            ratings = rec_res["ratings"]
            self.assertIn("/path/cp_A.safetensors", ratings)
            self.assertIn("/path/cp_B.safetensors", ratings)

            r_a = ratings["/path/cp_A.safetensors"]
            r_b = ratings["/path/cp_B.safetensors"]

            # cp_A should have gained rating and has 1 win
            self.assertGreater(r_a["rating"], 1500.0)
            self.assertEqual(r_a["wins"], 1)
            self.assertLess(r_a["uncertainty"], 350.0)

            # cp_B should have lost rating and has 1 loss
            self.assertLess(r_b["rating"], 1500.0)
            self.assertEqual(r_b["losses"], 1)
            self.assertLess(r_b["uncertainty"], 350.0)

            # Cross-profile consistency check: record cp_A vs cp_B on profile 1, cp_A wins again
            rec_res2 = api.handle_blind_test_record({
                "left_checkpoint": "/path/cp_A.safetensors",
                "right_checkpoint": "/path/cp_B.safetensors",
                "profile_index": 1,
                "chosen": "left",
                "is_canary": False,
            })
            metrics = rec_res2["metrics"]
            self.assertEqual(metrics["cross_profile_consistency"], 1.0)
            self.assertEqual(metrics["multi_profile_pairs"], 1)

            # Canary check: repeat match on profile 1, but flip sides so cp_A is right, user picks right
            canary_res = api.handle_blind_test_record({
                "left_checkpoint": "/path/cp_B.safetensors",
                "right_checkpoint": "/path/cp_A.safetensors",
                "profile_index": 1,
                "chosen": "right",  # cp_A chosen again
                "is_canary": True,
            })
            canary_metrics = canary_res["metrics"]
            self.assertEqual(canary_metrics["canary_consistency"], 1.0)
            self.assertEqual(canary_metrics["canary_total"], 1)

            # Reset
            reset_res = api.handle_blind_test_reset({})
            self.assertEqual(reset_res["ratings"], {})
            self.assertEqual(reset_res["history"], [])


if __name__ == "__main__":
    unittest.main()
