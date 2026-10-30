"""Unit tests for atomic output and submission_guard enforcement in evaluate_single_intervention_rule.py (CX-0011)."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.eval_20261030.submission_guard import REGISTRY_JSON, S48_JSON
from src.intensive_20261030.evaluate_single_intervention_rule import (
    write_and_validate_candidate_csv,
)


class TestSingleInterventionRuleGuard(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_guard_failure_unlinks_temp_and_does_not_write_final_csv(self):
        """When submission_guard rejects a candidate (e.g. duplicate of S48 or wrong IDs),
        no final CSV is written and no temporary file is left behind."""
        with open(S48_JSON, "r", encoding="utf-8") as f:
            s48_dict = {int(k): v for k, v in json.load(f).items()}
        with open(REGISTRY_JSON, "r", encoding="utf-8") as f:
            reg_data = json.load(f)

        # Case 1: Bad IDs (0..29999 instead of 1..30000) -> must fail guard and leave no file
        bad_ids = np.arange(0, 30000, dtype=int)
        s48_used = np.array([1 if s48_dict[i] == "used" else 0 for i in range(1, 30001)], dtype=int)
        out_csv_1 = self.temp_dir / "bad_ids_candidate.csv"

        with self.assertRaises(RuntimeError):
            write_and_validate_candidate_csv(bad_ids, s48_used, out_csv_1, reg_data, s48_dict)

        self.assertFalse(out_csv_1.exists())
        self.assertEqual(list(self.temp_dir.glob("*.tmp")), [])

        # Case 2: Exact duplicate of S48 (0 flips vs 48) -> must fail guard and leave no file
        good_ids = np.arange(1, 30001, dtype=int)
        out_csv_2 = self.temp_dir / "duplicate_s48_candidate.csv"

        with self.assertRaises(RuntimeError):
            write_and_validate_candidate_csv(good_ids, s48_used, out_csv_2, reg_data, s48_dict)

        self.assertFalse(out_csv_2.exists())
        self.assertEqual(list(self.temp_dir.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
