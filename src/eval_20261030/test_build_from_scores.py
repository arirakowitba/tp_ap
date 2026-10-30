"""Comprehensive Verification & Synthetic Test Suite for build_from_scores.py (Agent 5).

Tests all mandatory safety, content-addressing, nested contrast, disjoint union,
duplicate guard, and schema validation requirements, and writes a structured
JSON report to results/intensive_20261030/agent5_test_results.json.
"""
from __future__ import annotations

import inspect
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.eval_20261030.build_from_scores as bfs
from src.eval_20261030.build_from_scores import (
    SubmissionBuildError,
    compute_canonical_csv_sha256,
    compute_flip_set_hash,
    compute_sha256,
    compute_vector_sha256,
    evaluate_and_build,
)
from src.eval_20261030.submission_guard import REGISTRY_JSON, S48_JSON

RESULTS_DIR = ROOT / "results" / "intensive_20261030"
RESULTS_JSON = RESULTS_DIR / "agent5_test_results.json"


def _opposite_label(label: str) -> str:
    return "new" if label == "used" else "used"


def _write_candidate_csv(
    path: Path,
    s48_dict: dict[int, str],
    flip_ids: list[int] | set[int],
    override_labels: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Write a 30,000-row candidate CSV flipping flip_ids against s48_dict."""
    flip_set = {int(i) for i in flip_ids}
    flips: dict[int, str] = {}
    lines = ["ID,condition\n"]
    for i in range(1, 30001):
        if override_labels and i in override_labels:
            cond = override_labels[i]
        elif i in flip_set:
            cond = _opposite_label(s48_dict[i])
        else:
            cond = s48_dict[i]
        if cond in {"new", "used"} and cond != s48_dict[i]:
            flips[i] = cond
        lines.append(f"{i},{cond}\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)

    return {
        "path": str(path),
        "sha256": compute_sha256(path),
        "flip_set_hash": compute_flip_set_hash(flips, s48_dict) if all(
            c in {"new", "used"} for c in flips.values()
        ) else None,
        "flips": flips,
    }


class TestBuildFromScores(unittest.TestCase):
    """Comprehensive test suite for content-addressed submission builder."""

    @classmethod
    def setUpClass(cls) -> None:
        with open(S48_JSON, "r", encoding="utf-8") as f:
            cls.s48_dict = {int(k): v for k, v in json.load(f).items()}
        with open(REGISTRY_JSON, "r", encoding="utf-8") as f:
            cls.reg_data = json.load(f)
        cls.baseline_sha256 = compute_canonical_csv_sha256(cls.s48_dict)
        cls.base_score = 0.94633
        cls.test_records: list[dict[str, Any]] = []

        # Representative ID blocks matching real candidate cardinalities:
        # Block A (15 flips, e.g. conservative seen-seller core)
        cls.ids_block_a15 = [
            159, 3802, 4152, 6128, 6487, 6940, 7575, 11386,
            13384, 15152, 16565, 19403, 23244, 25976, 29552,
        ]
        # Block M (5 marginal flips disjoint from A15; B20 = A15 U M5)
        cls.ids_block_m5 = [732, 4969, 18735, 18790, 25747]
        cls.ids_block_b20 = sorted(cls.ids_block_a15 + cls.ids_block_m5)

        # Block C_disjoint (15 consensus flips strictly disjoint from B20)
        cls.ids_block_c15_disjoint = [
            3619, 4599, 5028, 5205, 6562, 7278, 12641, 17714,
            19545, 20200, 22327, 25326, 25600, 25940, 28448,
        ]
        # Block C_legacy25 (shares 10 IDs with B20/A15 + 15 unique consensus IDs)
        cls.ids_shared_10 = [732, 3802, 6128, 6487, 6940, 7575, 18790, 19403, 25976, 29552]
        cls.ids_block_c25_legacy = sorted(cls.ids_block_c15_disjoint + cls.ids_shared_10)

        # Block D (20 clean-text flips strictly disjoint from B20 and C25)
        cls.ids_block_d20 = [
            1228, 1842, 2663, 4742, 6190, 8117, 10615, 12136, 12183, 12832,
            13726, 14197, 14536, 17052, 18242, 20519, 21164, 22372, 26106, 29489,
        ]

    def _record(self, scenario_id: str, title: str, passed: bool, details: dict[str, Any]) -> None:
        self.__class__.test_records.append(
            {
                "scenario_id": scenario_id,
                "title": title,
                "passed": passed,
                "details": details,
            }
        )

    def test_00_no_filename_heuristics_in_source(self) -> None:
        """Verify build_from_scores.py contains zero filename-number heuristics or hardcoded ID sets."""
        src = inspect.getsource(bfs)
        forbidden_tokens = [
            "SUB77_MINUS_SUB80",
            "SUB78_CONSENSUS_UNIQUE",
            "SUB78_SHARED_WITH_77_80",
            'if "77" in',
            'if "78" in',
            'elif "78" in',
            'elif "79" in',
            'elif "80" in',
        ]
        found = [tok for tok in forbidden_tokens if tok in src]
        self.assertEqual(found, [], f"Forbidden filename heuristics found in source: {found}")
        self._record(
            "scenario_0_zero_filename_heuristics",
            "Source inspection confirms zero filename-number heuristics or hardcoded flip sets",
            True,
            {"forbidden_tokens_checked": forbidden_tokens, "violations_found": found},
        )

    def test_01_two_disjoint_winning_blocks(self) -> None:
        """Scenario 1: Two disjoint winning blocks (15 + 20 flips) combine cleanly into 35 flips."""
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            b1 = _write_candidate_csv(tdir / "alpha_probe.csv", self.s48_dict, self.ids_block_c15_disjoint)
            b2 = _write_candidate_csv(tdir / "beta_probe.csv", self.s48_dict, self.ids_block_d20)
            b_tie = _write_candidate_csv(tdir / "tie_probe.csv", self.s48_dict, [10, 20, 30])
            b_loss = _write_candidate_csv(tdir / "loss_probe.csv", self.s48_dict, [40, 50, 60])
            out_csv = tdir / "combined_35.csv"

            res = evaluate_and_build(
                probe_list=[
                    {"file": b1["path"], "sha256": b1["sha256"], "score": 0.94645},
                    {"file": b2["path"], "sha256": b2["sha256"], "score": 0.94650},
                    {"file": b_tie["path"], "sha256": b_tie["sha256"], "score": 0.94633},
                    {"file": b_loss["path"], "sha256": b_loss["sha256"], "score": 0.94610},
                ],
                s48_dict=self.s48_dict,
                reg_data=self.reg_data,
                base_score=self.base_score,
                out_csv=out_csv,
            )

            self.assertEqual(res["status"], "COMBINED_DISJOINT_UNION")
            self.assertTrue(res["output_written"])
            self.assertEqual(res["n_flips"], 35)
            self.assertEqual(
                set(res["flips"].keys()),
                set(self.ids_block_c15_disjoint) | set(self.ids_block_d20),
            )
            self.assertIsNone(res["expected_score"])
            self.assertTrue(res["additive_delta_sum_prohibited"])
            self.assertTrue(out_csv.exists())

            self._record(
                "scenario_1_two_disjoint_winners",
                "Two disjoint winning blocks (15 + 20 flips) + tie + loss -> 35-flip union, no delta sum",
                True,
                {
                    "status": res["status"],
                    "n_flips": res["n_flips"],
                    "combined_sha256": res["sha256"],
                    "combined_flip_set_hash": res["flip_set_hash"],
                    "expected_score": res["expected_score"],
                },
            )

    def test_02_partially_overlapping_winning_blocks_rejected(self) -> None:
        """Scenario 2: Two partially overlapping winning blocks (20 and 25 sharing 10 IDs) raise error."""
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            b1 = _write_candidate_csv(tdir / "block_20.csv", self.s48_dict, self.ids_block_b20)
            b2 = _write_candidate_csv(tdir / "block_25_overlap10.csv", self.s48_dict, self.ids_block_c25_legacy)

            with self.assertRaises(SubmissionBuildError) as ctx:
                evaluate_and_build(
                    probe_list=[
                        {"file": b1["path"], "sha256": b1["sha256"], "score": 0.94645},
                        {"file": b2["path"], "sha256": b2["sha256"], "score": 0.94650},
                    ],
                    s48_dict=self.s48_dict,
                    reg_data=self.reg_data,
                    base_score=self.base_score,
                    out_csv=tdir / "should_not_exist.csv",
                )

            err_msg = str(ctx.exception)
            self.assertIn("share 10 overlapping IDs", err_msg)
            for shared_id in self.ids_shared_10:
                self.assertIn(str(shared_id), err_msg)
            self.assertFalse((tdir / "should_not_exist.csv").exists())

            self._record(
                "scenario_2_partial_overlap_blocked",
                "Partially overlapping winning blocks (20 and 25 sharing 10 IDs) blocked with exact ID report",
                True,
                {"error_message": err_msg, "shared_ids_verified": self.ids_shared_10},
            )

    def test_03_valid_nested_contrasts(self) -> None:
        """Scenario 3: Valid nested contrast A (15) < B (20) when B improves over A (3a) vs when B does not (3b)."""
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            p_a15 = _write_candidate_csv(tdir / "nested_sub_a15.csv", self.s48_dict, self.ids_block_a15)
            p_b20 = _write_candidate_csv(tdir / "nested_sup_b20.csv", self.s48_dict, self.ids_block_b20)
            p_d20 = _write_candidate_csv(tdir / "disjoint_d20.csv", self.s48_dict, self.ids_block_d20)

            # Case 3a: score(A)=0.94645 > base, score(B)=0.94655 > score(A) -> promotes B (20) + D (20) = 40 flips
            out_3a = tdir / "out_3a_40flips.csv"
            res_3a = evaluate_and_build(
                probe_list=[
                    {"file": p_a15["path"], "sha256": p_a15["sha256"], "score": 0.94645},
                    {"file": p_b20["path"], "sha256": p_b20["sha256"], "score": 0.94655},
                    {"file": p_d20["path"], "sha256": p_d20["sha256"], "score": 0.94640},
                ],
                s48_dict=self.s48_dict,
                reg_data=self.reg_data,
                base_score=self.base_score,
                out_csv=out_3a,
            )
            self.assertEqual(res_3a["status"], "COMBINED_DISJOINT_UNION")
            self.assertEqual(res_3a["n_flips"], 40)
            self.assertEqual(res_3a["nested_contrasts"][0]["resolution"], "PROMOTE_SUPERSET_B")
            self.assertTrue(set(self.ids_block_m5).issubset(set(res_3a["flips"].keys())))

            # Case 3b: score(A)=0.94650 > base, score(B)=0.94645 <= score(A) -> retains A (15) + D (20) = 35 flips
            out_3b = tdir / "out_3b_35flips.csv"
            res_3b = evaluate_and_build(
                probe_list=[
                    {"file": p_a15["path"], "sha256": p_a15["sha256"], "score": 0.94650},
                    {"file": p_b20["path"], "sha256": p_b20["sha256"], "score": 0.94645},
                    {"file": p_d20["path"], "sha256": p_d20["sha256"], "score": 0.94640},
                ],
                s48_dict=self.s48_dict,
                reg_data=self.reg_data,
                base_score=self.base_score,
                out_csv=out_3b,
            )
            self.assertEqual(res_3b["status"], "COMBINED_DISJOINT_UNION")
            self.assertEqual(res_3b["n_flips"], 35)
            self.assertEqual(res_3b["nested_contrasts"][0]["resolution"], "RETAIN_SUBSET_A_REJECT_MARGIN")
            self.assertEqual(set(self.ids_block_m5) & set(res_3b["flips"].keys()), set())

            self._record(
                "scenario_3_valid_nested_contrast",
                "Nested contrast A (15) < B (20): promotes B when score(B)>score(A) (40 flips) and retains A when score(B)<=score(A) (35 flips)",
                True,
                {
                    "case_3a_n_flips": res_3a["n_flips"],
                    "case_3a_resolution": res_3a["nested_contrasts"][0]["resolution"],
                    "case_3b_n_flips": res_3b["n_flips"],
                    "case_3b_resolution": res_3b["nested_contrasts"][0]["resolution"],
                },
            )

    def test_04_same_number_in_filename_different_content_hashes(self) -> None:
        """Scenario 4: Files sharing '78' or '77' in filename with different contents are distinguished by hash."""
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            # Both files have 'sub78' in filename, but v1 has 25 legacy flips (overlapping 10 with B20)
            # while v2 has 15 disjoint consensus flips!
            sub77_full = _write_candidate_csv(
                tdir / "candidate_sub77_v1.csv", self.s48_dict, self.ids_block_b20
            )
            sub78_v1_legacy25 = _write_candidate_csv(
                tdir / "candidate_sub78_v1.csv", self.s48_dict, self.ids_block_c25_legacy
            )
            sub78_v2_disjoint15 = _write_candidate_csv(
                tdir / "candidate_sub78_v2.csv", self.s48_dict, self.ids_block_c15_disjoint
            )

            self.assertNotEqual(sub78_v1_legacy25["sha256"], sub78_v2_disjoint15["sha256"])
            self.assertNotEqual(sub78_v1_legacy25["flip_set_hash"], sub78_v2_disjoint15["flip_set_hash"])

            # v1 (25 flips) + sub77 (20 flips) MUST fail due to 10 shared IDs
            with self.assertRaises(SubmissionBuildError):
                evaluate_and_build(
                    probe_list=[
                        {"file": sub77_full["path"], "sha256": sub77_full["sha256"], "score": 0.94645},
                        {"file": sub78_v1_legacy25["path"], "sha256": sub78_v1_legacy25["sha256"], "score": 0.94650},
                    ],
                    s48_dict=self.s48_dict,
                    reg_data=self.reg_data,
                    base_score=self.base_score,
                )

            # v2 (15 disjoint flips) + sub77 (20 flips) MUST succeed -> 35 flips
            out_v2 = tdir / "combined_v2_35.csv"
            res_v2 = evaluate_and_build(
                probe_list=[
                    {"file": sub77_full["path"], "sha256": sub77_full["sha256"], "score": 0.94645},
                    {"file": sub78_v2_disjoint15["path"], "sha256": sub78_v2_disjoint15["sha256"], "score": 0.94650},
                ],
                s48_dict=self.s48_dict,
                reg_data=self.reg_data,
                base_score=self.base_score,
                out_csv=out_v2,
            )
            self.assertEqual(res_v2["status"], "COMBINED_DISJOINT_UNION")
            self.assertEqual(res_v2["n_flips"], 35)

            self._record(
                "scenario_4_same_filename_number_different_hashes",
                "Distinguishes candidate_sub78_v1.csv (25 legacy flips) vs candidate_sub78_v2.csv (15 disjoint flips) strictly by SHA256 and flip_set_hash",
                True,
                {
                    "v1_sha256": sub78_v1_legacy25["sha256"],
                    "v1_flip_set_hash": sub78_v1_legacy25["flip_set_hash"],
                    "v2_sha256": sub78_v2_disjoint15["sha256"],
                    "v2_flip_set_hash": sub78_v2_disjoint15["flip_set_hash"],
                    "v2_combined_n_flips": res_v2["n_flips"],
                },
            )

    def test_05_winning_superset_with_losing_subset_refuses_synthetic_extraction(self) -> None:
        """Scenario 5: Superset B (20 flips) wins while subset A (15 flips) loses -> refuses to extract B \\ A."""
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            p_a15_lost = _write_candidate_csv(tdir / "sub_a15_lost.csv", self.s48_dict, self.ids_block_a15)
            p_b20_won = _write_candidate_csv(tdir / "sup_b20_won.csv", self.s48_dict, self.ids_block_b20)
            p_d20_won = _write_candidate_csv(tdir / "disjoint_d20_won.csv", self.s48_dict, self.ids_block_d20)

            with self.assertRaises(SubmissionBuildError) as ctx:
                evaluate_and_build(
                    probe_list=[
                        {"file": p_a15_lost["path"], "sha256": p_a15_lost["sha256"], "score": 0.94620},
                        {"file": p_b20_won["path"], "sha256": p_b20_won["sha256"], "score": 0.94650},
                        {"file": p_d20_won["path"], "sha256": p_d20_won["sha256"], "score": 0.94645},
                    ],
                    s48_dict=self.s48_dict,
                    reg_data=self.reg_data,
                    base_score=self.base_score,
                )

            err_msg = str(ctx.exception)
            self.assertIn("Refusing to extract the unprobed 5-flip exclusive subset B \\ A", err_msg)

            self._record(
                "scenario_5_unprobed_exclusive_subset_blocked",
                "Winning superset B (20 flips, 0.94650) with losing subset A (15 flips, 0.94620) blocked from extracting unprobed 5-flip B \\ A",
                True,
                {"error_message": err_msg, "unprobed_marginal_ids": self.ids_block_m5},
            )

    def test_06_duplicate_and_single_winner_guards(self) -> None:
        """Scenario 6: Rejects S48/S76 duplicates, registry duplicates, and blocks single-winner duplicate emission."""
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            # 6a: Zero-flip duplicate of S48 / S76
            dup_s48 = _write_candidate_csv(tdir / "dup_s48.csv", self.s48_dict, [])
            with self.assertRaises(SubmissionBuildError) as ctx_s48:
                evaluate_and_build(
                    probe_list=[{"file": dup_s48["path"], "sha256": dup_s48["sha256"], "score": 0.94633}],
                    s48_dict=self.s48_dict,
                    reg_data=self.reg_data,
                    base_score=self.base_score,
                )
            self.assertIn("0 flips vs baseline", str(ctx_s48.exception))

            # 6b: Duplicate of historical registry submission (e.g. entry '75' in scored_registry.json)
            entry_75 = next(e for e in self.reg_data["entries"] if e["key"] == "75")
            dup_75 = _write_candidate_csv(
                tdir / "dup_sub75.csv", self.s48_dict, [int(x) for x in entry_75["flipped_ids"]]
            )
            with self.assertRaises(SubmissionBuildError) as ctx_75:
                evaluate_and_build(
                    probe_list=[{"file": dup_75["path"], "sha256": dup_75["sha256"], "score": 0.94650}],
                    s48_dict=self.s48_dict,
                    reg_data=self.reg_data,
                    base_score=self.base_score,
                )
            self.assertIn("75", str(ctx_75.exception))

            # 6c: Single winner already scored -> returns SINGLE_WINNER_ALREADY_SCORED and writes no file
            sole_win = _write_candidate_csv(tdir / "sole_winner.csv", self.s48_dict, self.ids_block_d20)
            loser = _write_candidate_csv(tdir / "loser.csv", self.s48_dict, self.ids_block_a15)
            out_single = tdir / "should_not_write_single.csv"
            res_single = evaluate_and_build(
                probe_list=[
                    {"file": sole_win["path"], "sha256": sole_win["sha256"], "score": 0.94650},
                    {"file": loser["path"], "sha256": loser["sha256"], "score": 0.94620},
                ],
                s48_dict=self.s48_dict,
                reg_data=self.reg_data,
                base_score=self.base_score,
                out_csv=out_single,
            )
            self.assertEqual(res_single["status"], "SINGLE_WINNER_ALREADY_SCORED")
            self.assertFalse(res_single["output_written"])
            self.assertFalse(out_single.exists())

            self._record(
                "scenario_6_duplicates_and_single_winner_guard",
                "Blocks S48/S76 duplicate, blocks historical registry duplicate (Sub 75), and prevents re-emitting a single already-scored winner",
                True,
                {
                    "s48_dup_error": str(ctx_s48.exception),
                    "reg75_dup_error": str(ctx_75.exception),
                    "single_winner_status": res_single["status"],
                    "single_winner_output_written": res_single["output_written"],
                },
            )

    def test_07_schema_scores_hashes_and_label_validation(self) -> None:
        """Scenario 7: Rejects NaN/Inf/None scores, SHA256 mismatches, invalid labels, duplicate/missing IDs."""
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            valid_cand = _write_candidate_csv(tdir / "valid.csv", self.s48_dict, self.ids_block_a15)

            # 7a: NaN and Inf scores
            for bad_score in [float("nan"), float("inf"), float("-inf"), None, "0.94650", True]:
                with self.assertRaises(SubmissionBuildError):
                    evaluate_and_build(
                        probe_list=[{"file": valid_cand["path"], "score": bad_score}],  # type: ignore[arg-type]
                        s48_dict=self.s48_dict,
                        reg_data=self.reg_data,
                    )

            # 7b: Wrong expected SHA256
            with self.assertRaises(SubmissionBuildError) as ctx_sha:
                evaluate_and_build(
                    probe_list=[{"file": valid_cand["path"], "sha256": "0" * 64, "score": 0.94650}],
                    s48_dict=self.s48_dict,
                    reg_data=self.reg_data,
                )
            self.assertIn("SHA256 mismatch", str(ctx_sha.exception))

            # 7c: Wrong baseline SHA256
            with self.assertRaises(SubmissionBuildError) as ctx_bsha:
                evaluate_and_build(
                    probe_list=[{"file": valid_cand["path"], "base_sha256": "f" * 64, "score": 0.94650}],
                    s48_dict=self.s48_dict,
                    reg_data=self.reg_data,
                )
            self.assertIn("Baseline SHA256 mismatch", str(ctx_bsha.exception))

            # 7d: Invalid label "refurbished"
            bad_label_csv = tdir / "bad_label.csv"
            _write_candidate_csv(
                bad_label_csv, self.s48_dict, self.ids_block_a15, override_labels={100: "refurbished"}
            )
            with self.assertRaises(SubmissionBuildError) as ctx_lbl:
                evaluate_and_build(
                    probe_list=[{"file": str(bad_label_csv), "score": 0.94650}],
                    s48_dict=self.s48_dict,
                    reg_data=self.reg_data,
                )
            self.assertIn("Invalid condition label", str(ctx_lbl.exception))

            # 7e: Duplicate ID and missing ID
            dup_id_csv = tdir / "dup_id.csv"
            lines = ["ID,condition\n", f"1,{self.s48_dict[1]}\n", f"1,{self.s48_dict[2]}\n"]
            for i in range(3, 30001):
                lines.append(f"{i},{self.s48_dict[i]}\n")
            dup_id_csv.write_text("".join(lines), encoding="utf-8")
            with self.assertRaises(SubmissionBuildError) as ctx_dup:
                evaluate_and_build(
                    probe_list=[{"file": str(dup_id_csv), "score": 0.94650}],
                    s48_dict=self.s48_dict,
                    reg_data=self.reg_data,
                )
            self.assertIn("Duplicate IDs", str(ctx_dup.exception))

            # 7f: Missing row (29,999 rows)
            short_csv = tdir / "short.csv"
            short_csv.write_text("".join(lines[:-1]), encoding="utf-8")
            with self.assertRaises(SubmissionBuildError) as ctx_short:
                evaluate_and_build(
                    probe_list=[{"file": str(short_csv), "score": 0.94650}],
                    s48_dict=self.s48_dict,
                    reg_data=self.reg_data,
                )
            self.assertIn("Invalid row count", str(ctx_short.exception))

            self._record(
                "scenario_7_schema_hash_score_validation",
                "Rejects NaN/Inf/None/bool scores, SHA256 mismatch, baseline mismatch, invalid label 'refurbished', duplicate IDs, and missing rows",
                True,
                {
                    "sha_mismatch_error": str(ctx_sha.exception),
                    "baseline_mismatch_error": str(ctx_bsha.exception),
                    "invalid_label_error": str(ctx_lbl.exception),
                    "duplicate_id_error": str(ctx_dup.exception),
                    "row_count_error": str(ctx_short.exception),
                },
            )

    def test_08_real_candidates_on_disk_content_addressed(self) -> None:
        """Scenario 8: Content-addressed audit and combination verification on real candidate CSVs in submissions/20261030."""
        sub_dir = ROOT / "submissions" / "20261030"
        f_77_full = sub_dir / "candidate_sub77_dual_cb_high_margin_unprobed.csv"
        f_77_margin5 = sub_dir / "candidate_sub77_nested_margin5_unprobed.csv"
        f_78_legacy25 = sub_dir / "candidate_sub78_model_consensus_unprobed.csv"
        f_78_disjoint15 = sub_dir / "candidate_sub78_consensus_disjoint15_unprobed.csv"
        f_79_clean20 = sub_dir / "candidate_sub79_new_sellers_clean_text.csv"
        f_80_seen15 = sub_dir / "candidate_sub80_seen_sellers_conservative_unprobed.csv"

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            # 8a: 4-block strictly disjoint suite on disk (15 + 5 + 15 + 20 = 55 flips)
            res_disjoint4 = evaluate_and_build(
                probe_list=[
                    {"file": str(f_80_seen15), "score": 0.94645},
                    {"file": str(f_77_margin5), "score": 0.94640},
                    {"file": str(f_78_disjoint15), "score": 0.94650},
                    {"file": str(f_79_clean20), "score": 0.94645},
                ],
                s48_dict=self.s48_dict,
                reg_data=self.reg_data,
                base_score=self.base_score,
                out_csv=tdir / "real_disjoint_55.csv",
            )
            self.assertEqual(res_disjoint4["status"], "COMBINED_DISJOINT_UNION")
            self.assertEqual(res_disjoint4["n_flips"], 55)

            # 8b: Legacy Sub 77 (20) + Legacy Sub 78 (25) on disk -> blocked due to 10 shared IDs
            with self.assertRaises(SubmissionBuildError) as ctx_leg:
                evaluate_and_build(
                    probe_list=[
                        {"file": str(f_77_full), "score": 0.94645},
                        {"file": str(f_78_legacy25), "score": 0.94650},
                    ],
                    s48_dict=self.s48_dict,
                    reg_data=self.reg_data,
                    base_score=self.base_score,
                )
            self.assertIn("share 10 overlapping IDs", str(ctx_leg.exception))

            self._record(
                "scenario_8_real_candidates_on_disk",
                "Verified real candidate CSVs in submissions/20261030: 4-block disjoint suite combines to 55 flips, while legacy 77+78 overlap is blocked",
                True,
                {
                    "disjoint4_n_flips": res_disjoint4["n_flips"],
                    "disjoint4_combined_sha256": res_disjoint4["sha256"],
                    "disjoint4_combined_flip_set_hash": res_disjoint4["flip_set_hash"],
                    "legacy_overlap_blocked_error": str(ctx_leg.exception),
                },
            )


def run_suite_and_save() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestBuildFromScores)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "agent": "AGENTE 5 — CONSTRUCTOR Y DISEÑO DE SUBMISSIONS",
        "module_under_test": "src/eval_20261030/build_from_scores.py",
        "test_module": "src/eval_20261030/test_build_from_scores.py",
        "total_tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "all_passed": result.wasSuccessful(),
        "baseline_s48_vector_sha256": TestBuildFromScores.baseline_sha256,
        "scenarios": TestBuildFromScores.test_records,
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved structured test report to: {RESULTS_JSON}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_suite_and_save())
