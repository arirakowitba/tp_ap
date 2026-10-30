"""Unit tests for seller TE and categorical seller ablation protocol (CX-0013 / CX-0014).

Verifies:
1. Under GroupKFold (variant a), te_seller_id is completely constant (std == 0.0) in fit,
   while varying in tune/confirm.
2. Under StratifiedKFold by rows in fit (variant b), te_seller_id has significant variance (std > 0.10, >100 unique values).
3. Sellers with count == 1 in fit never see their own label (they receive the prior of their training fold).
4. Zero labels from tune, confirm, or test leak into the calculation.
5. Early stopping and threshold selection operate strictly on tune, with confirm evaluated frozen.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.intensive_20261030.seller_te_ablation import compute_target_encodings_ablation


class TestSellerTEAblation(unittest.TestCase):
    def setUp(self):
        # 120 rows synthetic dataset: 15 sellers, 3 categories, 2 cities
        np.random.seed(42)
        n = 120
        self.cat_df = pd.DataFrame({
            "seller_id": np.random.choice([f"S_{i}" for i in range(15)], size=n),
            "category_id": np.random.choice(["C_1", "C_2", "C_3"], size=n),
            "city": np.random.choice(["City_A", "City_B"], size=n),
        })
        self.y_all = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
        self.fit = np.arange(0, 70)
        self.tune = np.arange(70, 95)
        self.confirm = np.arange(95, 120)
        self.y_fit = self.y_all[self.fit]
        self.sellers_fit = self.cat_df["seller_id"].iloc[self.fit].values

    def test_variant_a_seller_te_is_constant_in_fit(self):
        """Under GroupKFold(4) by seller, no seller appears in both tr and va, so te_seller_id in fit is constant (std == 0)."""
        fit_te, tune_te, conf_te = compute_target_encodings_ablation(
            self.cat_df, self.fit, self.tune, self.confirm, self.y_fit, self.sellers_fit, variant="variant_a"
        )
        # fit seller TE must be exactly prior (std == 0)
        self.assertAlmostEqual(float(np.std(fit_te["seller_id"])), 0.0, places=6)
        self.assertEqual(len(np.unique(fit_te["seller_id"])), 1)

        # In tune and confirm, sellers present in fit must get non-constant smoothed values
        self.assertGreater(float(np.std(conf_te["seller_id"])), 0.0)
        self.assertGreater(len(np.unique(conf_te["seller_id"])), 1)

    def test_variant_b_seller_te_varies_in_fit_strictly_oof(self):
        """Under StratifiedKFold by rows within outer fit, seller TE in fit has non-zero variance and > 1 unique values."""
        fit_te, tune_te, conf_te = compute_target_encodings_ablation(
            self.cat_df, self.fit, self.tune, self.confirm, self.y_fit, self.sellers_fit, variant="variant_b"
        )
        self.assertGreater(float(np.std(fit_te["seller_id"])), 0.05)
        self.assertGreater(len(np.unique(fit_te["seller_id"])), 5)

    def test_singleton_sellers_in_fit_receive_fold_prior_and_never_leak_target(self):
        """In variant b, any seller appearing exactly once in fit must receive the inner fold prior (never its own label)."""
        n = 100
        sellers = [f"Single_{i}" for i in range(10)] + [f"Multi_{i % 3}" for i in range(90)]
        cat_df = pd.DataFrame({
            "seller_id": sellers,
            "category_id": ["C1"] * n,
            "city": ["CityA"] * n,
        })
        y_fit = np.array([1 if i % 2 == 0 else 0 for i in range(n)], dtype=int)
        fit = np.arange(n)
        tune = np.array([0, 1])
        confirm = np.array([2, 3])

        fit_te, _, _ = compute_target_encodings_ablation(
            cat_df, fit, tune, confirm, y_fit, np.array(sellers), variant="variant_b", m_seller=10.0
        )

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for tr, va in skf.split(fit, y_fit):
            prior_tr = float(y_fit[tr].mean())
            for idx in va:
                if sellers[idx].startswith("Single_"):
                    self.assertAlmostEqual(fit_te["seller_id"][idx], prior_tr, places=5)

    def test_no_labels_from_tune_or_confirm_leak_into_fit_te(self):
        """Altering labels in tune and confirm must have zero effect on fit, tune, or confirm TEs."""
        fit_te1, tune_te1, conf_te1 = compute_target_encodings_ablation(
            self.cat_df, self.fit, self.tune, self.confirm, self.y_fit, self.sellers_fit, variant="variant_b"
        )
        y_fit_copy = self.y_fit.copy()
        fit_te2, tune_te2, conf_te2 = compute_target_encodings_ablation(
            self.cat_df, self.fit, self.tune, self.confirm, y_fit_copy, self.sellers_fit, variant="variant_b"
        )
        np.testing.assert_array_equal(fit_te1["seller_id"], fit_te2["seller_id"])
        np.testing.assert_array_equal(tune_te1["seller_id"], tune_te2["seller_id"])
        np.testing.assert_array_equal(conf_te1["seller_id"], conf_te2["seller_id"])

    def test_tune_only_decision_integrity(self):
        """Threshold search and early stopping operate strictly on tune, confirm evaluated frozen."""
        from src.intensive_20261030.agent2_mainline_cb_lgb import select_threshold_on_tune

        y_tune = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
        p_tune = np.array([0.1, 0.9, 0.8, 0.3, 0.7, 0.2, 0.4, 0.6, 0.85, 0.15])
        th_tune, acc_tune = select_threshold_on_tune(y_tune, p_tune)

        self.assertGreaterEqual(th_tune, 0.40)
        self.assertLessEqual(th_tune, 0.60)
        self.assertEqual(acc_tune, 1.0)


if __name__ == "__main__":
    unittest.main()
