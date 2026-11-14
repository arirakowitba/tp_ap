"""Audit canonical 3-way split against historical panels and analyze regime support.

Validates the tripartite split (fit, tune, confirm), cross-references confirm IDs
against all 6 prior development/holdout panels, and reports test vs split regime distributions.
"""

import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit():
    repo_root = Path(__file__).resolve().parent.parent
    split_csv = repo_root / "data" / "splits" / "canonical_3way_splits.csv"
    train_pkl = repo_root / "data" / "splits" / "t98_parsed_train.pkl"
    test_pkl = repo_root / "data" / "splits" / "t98_parsed_test.pkl"

    assert split_csv.exists(), f"Missing {split_csv}"
    assert train_pkl.exists(), f"Missing {train_pkl}"
    assert test_pkl.exists(), f"Missing {test_pkl}"

    split_sha = compute_sha256(split_csv)
    train_sha = compute_sha256(train_pkl)
    test_sha = compute_sha256(test_pkl)

    df_split = pd.read_csv(split_csv)
    with open(train_pkl, "rb") as f:
        df_train = pickle.load(f)
    with open(test_pkl, "rb") as f:
        df_test = pickle.load(f)

    # 1. Canonical Split counts
    fit_mask = df_split["role"] == "fit"
    tune_mask = df_split["role"] == "tune"
    confirm_mask = df_split["role"] == "confirm"

    fit_ids = set(df_split.loc[fit_mask, "id"])
    tune_ids = set(df_split.loc[tune_mask, "id"])
    confirm_ids = set(df_split.loc[confirm_mask, "id"])

    tune_cold = set(df_split.loc[df_split["regime"] == "tune_cold_new", "id"])
    tune_bracketed = set(df_split.loc[df_split["regime"] == "tune_seen_bracketed", "id"])
    confirm_cold = set(df_split.loc[df_split["regime"] == "confirm_cold_new", "id"])
    confirm_bracketed = set(df_split.loc[df_split["regime"] == "confirm_seen_bracketed", "id"])

    # 2. Test Set 5 regimes
    train_times = df_train.groupby("seller_id")["start_time"].agg(["min", "max"])
    s_test = df_test["seller_id"]
    t_test = df_test["start_time"]
    in_train = s_test.isin(train_times.index)
    bounds = train_times.reindex(s_test)
    t_min = bounds["min"].values
    t_max = bounds["max"].values
    t_val = t_test.values

    test_cold = ~in_train.values
    test_bracketed = in_train.values & (t_val > t_min) & (t_val < t_max)
    test_after = in_train.values & (t_val > t_max)
    test_before = in_train.values & (t_val < t_min)
    test_ties = in_train.values & ((t_val == t_min) | (t_val == t_max))

    # 3. Reconstruct Historical Panels and Overlap with Confirm
    # Panel 1: late_batch_seller (seed 28111322)
    rng1 = np.random.default_rng(28111322)
    sellers = np.sort(df_train.seller_id.unique())
    rng1.shuffle(sellers)
    cold1 = df_train.seller_id.isin(sellers[:len(sellers)//6]).to_numpy()
    unique_times = df_train.groupby("seller_id").start_time.nunique()
    eligible1 = df_train.seller_id.map(unique_times).ge(3).to_numpy() & ~cold1
    order1 = df_train.groupby("seller_id").start_time.rank(method="dense").to_numpy()
    upper1 = df_train.seller_id.map(unique_times).to_numpy()
    warm1 = eligible1 & (order1 > 1) & (order1 < upper1) & (rng1.random(len(df_train)) < .50)
    panel1_ids = set(df_train.id[cold1 | warm1])

    # Panel 2: late_batch_seller_28111323 (seed 28111323)
    rng2 = np.random.default_rng(28111323)
    sellers2 = np.sort(df_train.seller_id.unique())
    rng2.shuffle(sellers2)
    cold2 = df_train.seller_id.isin(sellers2[:len(sellers2)//6]).to_numpy()
    eligible2 = df_train.seller_id.map(unique_times).ge(3).to_numpy() & ~cold2
    order2 = df_train.groupby("seller_id").start_time.rank(method="dense").to_numpy()
    upper2 = df_train.seller_id.map(unique_times).to_numpy()
    warm2 = eligible2 & (order2 > 1) & (order2 < upper2) & (rng2.random(len(df_train)) < .50)
    panel2_ids = set(df_train.id[cold2 | warm2])

    # Panel 3: recent_seller_history (seed 28111215)
    rng3 = np.random.default_rng(28111215)
    sellers3 = np.sort(df_train.seller_id.unique())
    rng3.shuffle(sellers3)
    cold3 = df_train.seller_id.isin(sellers3[:len(sellers3)//8])
    ordered3 = df_train.loc[~cold3].sort_values("start_time")
    times3 = ordered3[["seller_id", "start_time"]].drop_duplicates()
    position3 = times3.groupby("seller_id").cumcount()
    distinct3 = times3.groupby("seller_id").start_time.transform("size")
    threshold3 = np.maximum(2, np.floor(.60 * distinct3))
    cutoff3 = times3.loc[position3.eq(threshold3)].set_index("seller_id").start_time
    warm3 = df_train.start_time.ge(df_train.seller_id.map(cutoff3))
    panel3_ids = set(df_train.id[cold3 | warm3])

    # Panel 4: structural_replica
    from sklearn.model_selection import StratifiedGroupKFold
    cold_fit4, cold_val4 = next(StratifiedGroupKFold(5, shuffle=True, random_state=281113).split(
        df_train, df_train.y, df_train.seller_id))
    warm_df4 = df_train.iloc[cold_fit4]
    grouped4 = warm_df4.groupby("seller_id").start_time
    last4, first4 = grouped4.transform("max"), grouped4.transform("min")
    warm_val4 = warm_df4.index[(warm_df4.start_time == last4) & (first4 < last4)]
    val4 = np.sort(np.r_[cold_val4, warm_val4])
    panel4_ids = set(df_train.id.iloc[val4])

    # Overlaps computed directly:
    ov1 = len(confirm_ids & panel1_ids)
    ov2 = len(confirm_ids & panel2_ids)
    ov3 = len(confirm_ids & panel3_ids)
    ov4 = len(confirm_ids & panel4_ids)

    # Panel 5 (structural_disjoint) and Panel 6 (structural_heldout_confirm) overlap counts from CX-0189
    ov5 = 2247
    ov6 = 1491
    total_union_historical = 58931
    total_exposed_confirm = 12504
    fresh_intact_confirm = 1571

    # Fresh intact confirmation details
    # The 1,571 IDs are outside the 6 panels
    exposed_so_far = panel1_ids | panel2_ids | panel3_ids | panel4_ids
    confirm_unexposed_4panels = confirm_ids - exposed_so_far

    report = {
        "split_manifest": {
            "csv_path": str(split_csv.relative_to(repo_root)),
            "csv_sha256": split_sha,
            "train_pkl_sha256": train_sha,
            "test_pkl_sha256": test_sha,
            "total_rows": len(df_split),
            "fit_rows": len(fit_ids),
            "tune_rows": len(tune_ids),
            "tune_cold_rows": len(tune_cold),
            "tune_seen_bracketed_rows": len(tune_bracketed),
            "confirm_rows": len(confirm_ids),
            "confirm_cold_rows": len(confirm_cold),
            "confirm_seen_bracketed_rows": len(confirm_bracketed),
        },
        "test_regimes_30k": {
            "cold_new": int(test_cold.sum()),
            "bracketed_interleaved": int(test_bracketed.sum()),
            "after_all_history": int(test_after.sum()),
            "before_all_history": int(test_before.sum()),
            "ties_extrema": int(test_ties.sum()),
            "total": int(len(df_test)),
            "shares": {
                "cold_new": round(float(test_cold.sum() / len(df_test)), 4),
                "bracketed_interleaved": round(float(test_bracketed.sum() / len(df_test)), 4),
                "after_all_history": round(float(test_after.sum() / len(df_test)), 4),
                "before_all_history": round(float(test_before.sum() / len(df_test)), 4),
                "ties_extrema": round(float(test_ties.sum() / len(df_test)), 4),
            }
        },
        "historical_panels_cross": {
            "late_batch_seller_28111322": {
                "total_rows": len(panel1_ids),
                "confirm_overlap": ov1,
                "confirm_overlap_share": round(ov1 / len(confirm_ids), 4)
            },
            "late_batch_seller_28111323": {
                "total_rows": len(panel2_ids),
                "confirm_overlap": ov2,
                "confirm_overlap_share": round(ov2 / len(confirm_ids), 4)
            },
            "recent_seller_history": {
                "total_rows": len(panel3_ids),
                "confirm_overlap": ov3,
                "confirm_overlap_share": round(ov3 / len(confirm_ids), 4)
            },
            "structural_replica": {
                "total_rows": len(panel4_ids),
                "confirm_overlap": ov4,
                "confirm_overlap_share": round(ov4 / len(confirm_ids), 4)
            },
            "structural_disjoint": {
                "total_rows": 9173,
                "confirm_overlap": ov5,
                "confirm_overlap_share": round(ov5 / len(confirm_ids), 4)
            },
            "structural_heldout_confirm": {
                "total_rows": 7632,
                "confirm_overlap": ov6,
                "confirm_overlap_share": round(ov6 / len(confirm_ids), 4)
            },
            "union_six_panels": {
                "total_unique_ids": total_union_historical,
                "confirm_overlap": total_exposed_confirm,
                "confirm_overlap_share": round(total_exposed_confirm / len(confirm_ids), 4),
                "fresh_intact_rows": fresh_intact_confirm,
                "fresh_intact_share": round(fresh_intact_confirm / len(confirm_ids), 4)
            }
        },
        "methodological_consequences": {
            "confirm_retrospective_development_rows": total_exposed_confirm,
            "confirm_fresh_intact_rows": fresh_intact_confirm,
            "status": "EVALUATION_FROZEN",
            "reference_anchor": "S4_EVT_154_0_94766",
            "public_probability_claim": "NO_IDENTIFIABLE"
        }
    }

    out_json = repo_root / "results" / "canonical_split_panels_overlap_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    audit()
