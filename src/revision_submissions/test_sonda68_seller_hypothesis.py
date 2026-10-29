"""Empirical Evaluation of Seller History Prior Gate against Deep CatBoost Baseline.

Directly tests the hypothesis behind the initial Sonda 68 on labeled confirm splits (seed 2026 and 314159):
Rule: For seen sellers in fit with n_fit >= 3 and used_ratio >= 0.80, if the strong model predicts 'new',
flip prediction to 'used'.

Results:
Seed 2026: Activated on 44 rows: fixes=13, breaks=31, net=-18
Seed 314159: Activated on 54 rows: fixes=23, breaks=31, net=-8

Conclusion:
Consistently breaks more correct predictions than it fixes when applied against a strong model.
Refutes seller-history prior gate on strong models, matching Round 141 findings (-49, -47, -48 net).
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRAIN_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "train_data.jsonlines"
RESULTS_DIR = ROOT / "results" / "revision_submissions"


def main():
    rows = []
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    sellers = np.array([str(r["seller_id"]) for r in rows])
    y_used = np.array([1 if r["condition"] == "used" else 0 for r in rows], dtype=int)

    def get_splits(seed):
        rng = np.random.default_rng(seed)
        unique = np.unique(sellers)
        rng.shuffle(unique)
        n_new = len(unique) // 10
        new_tune = set(unique[:n_new])
        new_confirm = set(unique[n_new: 2 * n_new])
        seen_sellers = ~(np.isin(sellers, list(new_tune | new_confirm)))
        seen_row_indices = np.where(seen_sellers)[0]
        rng.shuffle(seen_row_indices)
        n_seen_tune = len(seen_row_indices) // 10
        seen_tune = seen_row_indices[:n_seen_tune]
        seen_confirm = seen_row_indices[n_seen_tune: 2 * n_seen_tune]
        tune_rows = np.flatnonzero(np.isin(sellers, list(new_tune)))
        confirm_rows = np.flatnonzero(np.isin(sellers, list(new_confirm)))
        tune = np.r_[tune_rows, seen_tune]
        confirm = np.r_[confirm_rows, seen_confirm]
        fit_mask = np.ones(len(rows), dtype=bool)
        fit_mask[tune] = False
        fit_mask[confirm] = False
        fit = np.where(fit_mask)[0]
        return fit, tune, confirm

    for seed in [2026, 314159]:
        fit_idx, tune_idx, confirm_idx = get_splits(seed)
        fit_seller_cnt = defaultdict(int)
        fit_seller_used = defaultdict(int)
        for idx in fit_idx:
            s = sellers[idx]
            fit_seller_cnt[s] += 1
            if y_used[idx] == 1:
                fit_seller_used[s] += 1

        paired_file = RESULTS_DIR / f"paired_deep_ablation_seed_{seed}.csv"
        df_p = pd.read_csv(paired_file)

        cnt_rule = 0
        fixes = 0
        breaks = 0

        for _, r in df_p.iterrows():
            s = str(r["seller_id"])
            if r["is_seen"] == 1 and fit_seller_cnt[s] >= 3:
                ratio = fit_seller_used[s] / fit_seller_cnt[s]
                if ratio >= 0.80:
                    pred_base = r["pred_base"]
                    y = int(r["y_true"])
                    if pred_base == 0:  # model predicted 'new'
                        cnt_rule += 1
                        if y == 1:
                            fixes += 1
                        else:
                            breaks += 1

        print(f"Seed {seed}: Activated on {cnt_rule} rows: fixes={fixes}, breaks={breaks}, net={fixes-breaks}")


if __name__ == "__main__":
    main()
