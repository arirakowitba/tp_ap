"""Check weighted targets, finite-count thresholds and conditional assurance."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ap2026q2_global_objective import (
    global_accuracy, required_public_correct, required_hidden_correct,
    expected_flip_budget, evaluate_joint_hidden_draws,
)


class GlobalObjectiveTest(unittest.TestCase):
    def test_user_scenario_and_integer_frontier(self):
        self.assertAlmostEqual(global_accuracy(.95, .94523), .948569)
        count = required_public_correct(.94523)
        self.assertEqual(count, 20293)
        self.assertGreaterEqual(global_accuracy(count / 21000, .94523), .96)
        self.assertLess(global_accuracy((count - 1) / 21000, .94523), .96)
        self.assertEqual(required_hidden_correct(20370), 8430)

    def test_public_only_credit_costs_more_flips(self):
        self.assertEqual(expected_flip_budget(520, .75, .7), 1486)
        self.assertEqual(expected_flip_budget(520, .75), 1040)
        self.assertIsNone(expected_flip_budget(520, .5))

    def test_good_mean_does_not_imply_high_confidence(self):
        # Mean hidden accuracy 95 %; a 10 % low scenario still fails the gate.
        result = evaluate_joint_hidden_draws(20370, np.r_[np.full(90, 8650), np.full(10, 7650)])
        self.assertGreater(result["expected_global_accuracy"], .96)
        self.assertEqual(result["prob_global_at_least_target"], .9)
        self.assertFalse(result["passes_conditional_gate"])
        self.assertTrue(evaluate_joint_hidden_draws(20370, np.full(100, 8500))["passes_conditional_gate"])
        with self.assertRaisesRegex(ValueError, "integer hidden"):
            evaluate_joint_hidden_draws(20370, [.95])


if __name__ == "__main__":
    unittest.main()
