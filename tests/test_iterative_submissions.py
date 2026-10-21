"""Check recovery of hidden cohort effects from actual leaderboard-style scores."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ap2026q2_submission_review import displayed_score
from ap2026q2_iterative_submissions import (
    choose_combination, cohort_confidence, infer_cohorts, public_correct, validate_baseline,
)


class IterativeEvidenceTest(unittest.TestCase):
    def test_four_crossed_probes_identify_effects_and_inconsistency(self):
        effects = dict(A=3, B=-2, C=7, D=-1)
        sizes = dict(A=5, B=4, C=9, D=3)
        designs = {"16": ["A", "B"], "17": ["A", "C"],
                   "18": ["B", "C"], "19": ["B", "D"]}
        baseline_count = 19850
        score = lambda numerator: f"{displayed_score(numerator, 21000, 'truncate'):.5f}"
        observed = {name: score(baseline_count + sum(effects[g] for g in groups))
                    for name, groups in designs.items()}
        summary, joint = infer_cohorts(sizes, designs, observed, score(baseline_count))
        self.assertEqual(len(joint), 1)
        for name, gain in effects.items():
            self.assertEqual(int(joint.iloc[0][name]), gain)
            self.assertEqual(int(summary.set_index("cohort").loc[name, "feasible_public_gains"]), 1)
        observed["19"] = score(baseline_count + 100)
        with self.assertRaisesRegex(ValueError, "no configuration"):
            infer_cohorts(sizes, designs, observed, score(baseline_count))

    def test_pending_16_keeps_control_and_rejects_overlap(self):
        baseline = pd.DataFrame({"ID": np.arange(1, 30001),
                                 "condition": ["new"] * 30000})
        cohorts = {"A": pd.DataFrame({"ID": [21, 22],
                                        "alternative_label": ["used", "used"]}),
                   "B": pd.DataFrame({"ID": [23], "alternative_label": ["used"]})}
        summary, joint = infer_cohorts({"A": 2, "B": 1}, {"16": ["A", "B"]},
                                       {"16": None}, "0.94523")
        ranked, csv = choose_combination(summary, joint, cohorts, baseline)
        self.assertEqual(int(ranked.iloc[0].flips), 0)
        self.assertTrue(csv.equals(baseline))
        self.assertEqual(public_correct("0.94523"), 19850)
        # If only A+B was scored, a winning bundle can still be retained
        # without inventing separate labels or effects for A and B.
        winning_score = f"{displayed_score(19851, 21000, 'truncate'):.5f}"
        partial, feasible = infer_cohorts({"A": 2, "B": 1}, {"16": ["A", "B"]},
                                          {"16": winning_score}, "0.94523")
        self.assertGreater(len(feasible), 1)
        bundled, exported = choose_combination(partial, feasible, cohorts, baseline)
        self.assertEqual(int(bundled.iloc[0].flips), 3)
        self.assertEqual(int((exported.condition != baseline.condition).sum()), 3)
        bad = cohorts | {"B": pd.DataFrame({"ID": [21],
                                            "alternative_label": ["used"]})}
        with self.assertRaisesRegex(ValueError, "overlap"):
            choose_combination(summary, joint, bad, baseline)
        validate_baseline(csv)

    def test_five_probe_matrix_resolves_large_extension(self):
        effects = dict(A=2, B=-1, C=3, D=-2, E=5)
        sizes = dict(A=4, B=3, C=5, D=4, E=7)
        designs = {"16":["A","B"],"17":["A","C"],"18":["B","C"],
                   "19":["B","D"],"20":["C","D","E"]}
        score = lambda k:f"{displayed_score(k,21000,'truncate'):.5f}"
        observed = {name:score(19850 + sum(effects[g] for g in included))
                    for name,included in designs.items()}
        summary, joint = infer_cohorts(sizes, designs, observed, score(19850))
        self.assertEqual(len(joint), 1)
        self.assertEqual([int(joint.iloc[0][g]) for g in sizes], list(effects.values()))
        self.assertTrue(summary.feasible_public_gains.eq(1).all())

    def test_observed_round_identifies_only_a_as_positive(self):
        import json
        record = json.loads((ROOT / "results/revision_submissions/feedback_iterativo_20261020.json").read_text())
        sizes = dict(A=32, B=32, C=64, D=64, E=128)
        summary, joint = infer_cohorts(sizes, record["experiments"], record["scores"],
                                       record["baseline"]["score"])
        self.assertEqual(len(joint), 1)
        self.assertEqual([int(joint.iloc[0][g]) for g in sizes], [2, -7, -18, -17, -15])
        self.assertEqual(public_correct("0.94533"), 19852)
        self.assertEqual(summary.loc[summary.expected_public_gain.gt(0), "cohort"].tolist(), ["A"])

    def test_hidden_prediction_remains_uncertain_after_public_gain(self):
        confidence = cohort_confidence(32, 2)
        self.assertLess(confidence["flip_correctness_ci_low"], .5)
        self.assertGreater(confidence["flip_correctness_ci_high"], .5)
        self.assertLess(confidence["prob_hidden_gain_positive"], .75)
        self.assertGreater(confidence["prob_hidden_gain_negative"], .1)
        finite = cohort_confidence(1, 1)
        self.assertEqual(finite["expected_hidden_gain"], 0)
        self.assertEqual(finite["prob_hidden_gain_zero"], 1)


if __name__ == "__main__":
    unittest.main()
