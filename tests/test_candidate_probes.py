"""Unit tests for the 4 informative probes, adaptive 5th shot, and submission guard compliance (CX-0016)."""
import hashlib
import json
import unittest
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "intensive_20261030"
MANIFEST_PATH = OUT_DIR / "unified_candidate_manifest_20261030.json"
S48_PATH = ROOT / "submissions" / "20261030" / "submission_48_reference.csv"
S48_JSON = ROOT / "results" / "revision_submissions" / "posterior_flips_20261027" / "s48.json"
REGISTRY_JSON = ROOT / "results" / "eval_20261030" / "scored_registry.json"

PROBE_FILES = [
    OUT_DIR / "candidate_probe1_seen_used_to_new.csv",
    OUT_DIR / "candidate_probe2_seen_new_to_used.csv",
    OUT_DIR / "candidate_probe3_seen_low_freq.csv",
    OUT_DIR / "candidate_probe4_new_orthogonal_teacher.csv",
]


class TestCandidateProbesHarness(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(S48_JSON, "r", encoding="utf-8") as f:
            cls.s48_dict = {int(k): v for k, v in json.load(f).items()}
        with open(REGISTRY_JSON, "r", encoding="utf-8") as f:
            cls.registry = json.load(f)

    def test_s48_reference_hash_and_format(self):
        """Reference 48 must match exact canonical hash 3f8d08e0... and have 30000 rows."""
        self.assertTrue(S48_PATH.exists(), f"Reference 48 not found at {S48_PATH}")
        content = S48_PATH.read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        self.assertEqual(
            sha,
            "3f8d08e0e61854c2805a02140b1f7428016d63deae4bd7d6c820c7478f6c9722",
            "submission_48_reference.csv SHA256 does not match canonical reference!",
        )
        df48 = pd.read_csv(S48_PATH)
        self.assertEqual(len(df48), 30000)
        self.assertEqual(list(df48.columns), ["ID", "condition"])

    def test_probe_files_exist_and_pass_submission_guard(self):
        """All 4 probe candidate CSVs must exist and pass check_submission_file without error."""
        from src.eval_20261030.submission_guard import check_submission_file

        for p in PROBE_FILES:
            self.assertTrue(p.exists(), f"Probe file {p.name} does not exist!")
            ok, msg = check_submission_file(p, self.registry, self.s48_dict, verbose=False)
            self.assertTrue(ok, f"Submission guard rejected {p.name}: {msg}")

    def test_probe_flip_counts_within_budget(self):
        """Each candidate probe must have bounded flip counts between 15 and 35."""
        for p in PROBE_FILES:
            df = pd.read_csv(p)
            flips = sum(df["condition"].values != [self.s48_dict[i] for i in range(1, 30001)])
            self.assertGreaterEqual(flips, 15, f"{p.name} has fewer than 15 flips ({flips})")
            self.assertLessEqual(flips, 35, f"{p.name} has more than 35 flips ({flips})")

    def test_probes_are_mutually_disjoint(self):
        """Probes 1, 2, 3, and 4 must have pairwise mutually disjoint flip sets."""
        flip_sets = {}
        for p in PROBE_FILES:
            df = pd.read_csv(p)
            flipped_ids = set()
            for idx, row in df.iterrows():
                row_id = int(row["ID"])
                if row["condition"] != self.s48_dict[row_id]:
                    flipped_ids.add(row_id)
            flip_sets[p.name] = flipped_ids

        probe_names = list(flip_sets.keys())
        for i in range(len(probe_names)):
            for j in range(i + 1, len(probe_names)):
                name_a = probe_names[i]
                name_b = probe_names[j]
                inter = flip_sets[name_a].intersection(flip_sets[name_b])
                self.assertEqual(
                    len(inter),
                    0,
                    f"Probes {name_a} and {name_b} share {len(inter)} overlapping flips: {inter}",
                )

    def test_probe_manifest_structure_and_adaptive_tree(self):
        """Manifest must contain all 4 probes, valid hashes, and the formal 5th shot adaptive tree."""
        self.assertTrue(MANIFEST_PATH.exists(), f"Manifest {MANIFEST_PATH} not found!")
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertIn("probes", manifest)
        self.assertEqual(len(manifest["probes"]), 4)
        for p_key, p_data in manifest["probes"].items():
            self.assertIn("n_flips", p_data)
            self.assertIn("direction", p_data)
            self.assertIn("segment", p_data)
            self.assertIn("sha256", p_data)
            self.assertIn("flipped_ids", p_data)
            self.assertIn("historical_overlaps", p_data)

        self.assertIn("adaptive_decision_tree_slot5", manifest)
        tree = manifest["adaptive_decision_tree_slot5"]
        self.assertIn("fallback_baseline", tree)
        self.assertIn("abandonment_rule", tree)
        self.assertEqual(
            tree["fallback_baseline"]["sha256"],
            "3f8d08e0e61854c2805a02140b1f7428016d63deae4bd7d6c820c7478f6c9722",
        )

    def test_candidate_models_havg_and_havg_plus_valid(self):
        """H-AVG and H-AVG-Plus candidates must exist, have 30000 rows, and pass submission guard."""
        from src.eval_20261030.submission_guard import check_submission_file

        for name in ["candidate_model_havg_b10_variant_c_full70k.csv", "candidate_model_havg_plus_full70k.csv"]:
            p = OUT_DIR / name
            self.assertTrue(p.exists(), f"Candidate {name} does not exist at {p}")
            df = pd.read_csv(p)
            self.assertEqual(len(df), 30000, f"{name} does not have 30000 rows")
            self.assertEqual(list(df.columns), ["ID", "condition"])
            ok, msg = check_submission_file(p, self.registry, self.s48_dict, verbose=False)
            self.assertTrue(ok, f"Submission guard rejected {name}: {msg}")


if __name__ == "__main__":
    unittest.main()
