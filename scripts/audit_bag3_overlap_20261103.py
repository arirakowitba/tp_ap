"""Deduplicate bag3 confirm folds and audit its test-only intervention.

Requires the private paired CSV checkpoint and original 30k test prediction
checkpoint. No public score or test label is interpreted as a per-ID label.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "results/catboost_bag3"
TEST = ROOT / "results/catboost_bag3_full70k/test_probabilities.csv"
ANCHOR = ROOT / "results/historical_text_seller_full70k/submission_final_best_d1_d2_20261031.csv"
LEDGER = ROOT / "results/historical_text_seller_full70k/submission_ledger_04_current.json"


def validation(threshold):
    df = pd.concat([pd.read_csv(PAIRS / f"paired_{seed}.csv").assign(seed=seed)
                    for seed in (20261124, 20261125, 20261126)], ignore_index=True)
    assert df.groupby("item_id").y_true.nunique().eq(1).all()
    assert df.groupby("item_id").seller_id.nunique().eq(1).all()
    df["delta"] = ((df.p_bag3 >= threshold) == df.y_true).astype(int) - \
                  ((df.p_single >= threshold) == df.y_true).astype(int)
    unique = df.groupby(["item_id", "seller_id"], as_index=False).agg(
        mean_delta=("delta", "mean"), n=("delta", "size"),
        min_delta=("delta", "min"), max_delta=("delta", "max"))
    seller = unique.groupby("seller_id").mean_delta.sum().to_numpy()
    rng = np.random.default_rng(1234)
    boot = np.array([seller[rng.integers(0, len(seller), len(seller))].sum()
                     for _ in range(3000)])
    return {"rows": len(df), "unique_items": len(unique),
            "repeated_items": int((unique.n > 1).sum()),
            "row_net": int(df.delta.sum()),
            "unique_mean_net": float(unique.mean_delta.sum()),
            "unique_seller_bootstrap_ci95": np.quantile(boot, [.025, .975]).tolist(),
            "repeated_sign_conflicts": int(((unique.n > 1) & (unique.min_delta < 0)
                                            & (unique.max_delta > 0)).sum())}


def test_intervention(threshold):
    p = pd.read_csv(TEST)
    anchor = pd.read_csv(ANCHOR)
    assert len(p) == len(anchor) == 30000
    assert np.array_equal(p.ID.to_numpy(), anchor.ID.to_numpy())
    bag = p.p_bag3 >= threshold
    single = p.p_seed0 >= threshold
    anchor_used = anchor.condition.eq("used")
    bag_only = (bag != single) & (anchor_used == single)
    entries = json.loads(LEDGER.read_text())["entries"]
    touched = set().union(*(set(e.get("flipped_ids") or []) for e in entries))
    chosen = set(p.loc[bag_only, "ID"].astype(int))
    return {"bag_only_vs_anchor": len(chosen),
            "seen": int(p.loc[bag_only, "seller_seen"].sum()),
            "new": int((~p.loc[bag_only, "seller_seen"].astype(bool)).sum()),
            "historically_touched": len(chosen & touched),
            "historically_untouched": len(chosen - touched),
            "overlap_conditional_exact_four": sorted(chosen & {1383, 10965, 20534, 20806}),
            "margin_bag_at_least_0_05": int((bag_only & ((p.p_bag3 - threshold).abs() >= .05)).sum())}


if __name__ == "__main__":
    print(json.dumps({"fixed_threshold_0_475_validation": validation(.475),
                      "fixed_threshold_0_5_validation": validation(.5),
                      "test_intervention_0_475": test_intervention(.475)}, indent=2))
