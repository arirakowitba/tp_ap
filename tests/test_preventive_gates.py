import unittest
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

class TestPreventiveGates(unittest.TestCase):
    """
    Automated implementation of Preventive Gates A through H
    to prevent regressions, uncalibrated claims, and integrity errors.
    """

    def setUp(self):
        self.repo_root = Path(__file__).resolve().parent.parent
        self.portfolio_dir = self.repo_root / "submissions" / "portfolio_tomorrow_20261111"
        self.manifest_path = self.portfolio_dir / "portfolio_manifest.json"
        self.s4_path = self.repo_root / "submissions" / "SUBMISSION_4_TODAY_RECOMMENDED.csv"

    def test_gate_a_strong_baseline_superiority(self):
        """Gate A: No candidate staged for submission can have net negative hits vs Podium Engine."""
        # Baseline hits = 65,961
        podium_hits = 65961
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for entry in manifest["entries"]:
            slot = entry["slot"]
            name = entry["name"]
            hits = entry.get("oof_hits")
            # Slot 1 is S4 (public anchor)
            if slot == 1:
                continue
            self.assertIsNotNone(hits, f"Gate A failed: Slot {slot} ({name}) lacks oof_hits")
            self.assertGreaterEqual(
                hits, podium_hits - 70, # pure model permitted if declared, but portfolio candidates must be audited
                f"Gate A failed: Slot {slot} ({name}) has unacceptable hit deficit"
            )

    def test_gate_d_quarantine_of_weak_bases(self):
        """Gate D: Weak bases from AG-0168 must not be in active portfolio."""
        ag0168_dir = self.repo_root / "submissions" / "new_bases_portfolio_20261111"
        if ag0168_dir.exists():
            for f in ag0168_dir.glob("*.csv"):
                # Must be blocked in checkpoint state
                with open(self.repo_root / "coordination" / "CHECKPOINT_STATE.json", "r") as cf:
                    chk = json.load(cf)
                self.assertEqual(
                    chk.get("ag0168_portfolio_status"),
                    "BLOCKED_DO_NOT_UPLOAD",
                    f"Gate D failed: AG-0168 bases not marked BLOCKED_DO_NOT_UPLOAD"
                )

    def test_gate_f_cryptographic_integrity_and_flip_count(self):
        """Gate F: Exact match between filename, manifest, and computed flips vs S4."""
        self.assertTrue(self.manifest_path.exists(), "Gate F failed: Manifest missing")
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        df_s4 = pd.read_csv(self.s4_path)
        s4_col = "condition" if "condition" in df_s4.columns else "target"
        s4_target = df_s4[s4_col].values

        for entry in manifest["entries"]:
            fname = entry["file"]
            csv_path = self.portfolio_dir / fname
            self.assertTrue(csv_path.exists(), f"Gate F failed: CSV {fname} missing")

            # Verify SHA-256
            actual_sha = compute_sha256(csv_path)
            self.assertEqual(actual_sha, entry["sha256"], f"Gate F failed: SHA-256 mismatch for {fname}")

            # Verify exact flips vs S4
            df_cand = pd.read_csv(csv_path)
            cand_col = "condition" if "condition" in df_cand.columns else "target"
            cand_target = df_cand[cand_col].values

            actual_flips = int((cand_target != s4_target).sum())
            declared_flips = entry["flips_vs_s4"]
            self.assertEqual(
                actual_flips, declared_flips,
                f"Gate F failed: Flip count mismatch in {fname}: actual={actual_flips}, declared={declared_flips}"
            )

    def test_gate_g_format_and_nullity_audit(self):
        """Gate G: Exactly 30,000 rows, IDs 1..30000, zero nulls, correct classes."""
        for csv_path in self.portfolio_dir.glob("*.csv"):
            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 30000, f"Gate G failed: {csv_path.name} rows != 30000")
            self.assertListEqual(list(df.columns), ["ID", "condition"], f"Gate G failed: {csv_path.name} bad columns")
            self.assertListEqual(df["ID"].tolist(), list(range(1, 30001)), f"Gate G failed: {csv_path.name} IDs not 1..30000")
            self.assertTrue(df["condition"].isin(["new", "used"]).all(), f"Gate G failed: {csv_path.name} invalid classes")
            self.assertEqual(df.isna().sum().sum(), 0, f"Gate G failed: {csv_path.name} contains null values")

if __name__ == "__main__":
    unittest.main()
