"""Exhaustive Historical Holdout Overlap Audit Across ALL Repository Partitions.

Audits the overlap in rows and unique sellers between all current evaluation cuts
(2026, 314159, 20261030, disjoint_3seed_20261101, nested_outer_777001, nested_outer_777002)
and ALL historical tune/confirm/validation holdouts preserved or reconstructable in the repository:
1. R151_seed2026_tune & R151_seed2026_confirm
2. R151_seed314159_tune & R151_seed314159_confirm
3. R151_seed20261030_tune & R151_seed20261030_confirm
4. R157_seed2026_tune & R157_seed2026_confirm
5. R157_seed314159_tune & R157_seed314159_confirm
6. round_v3_prior_confirmation_seed731 (14,001 rows, results/round_v3/prior_confirmation_predictions.csv)
7. round_v5_selection_seed42 (14,000 rows, results/round_v5/round3_selection_predictions.csv)
8. optimize_leaderboard_seed42_holdout3 (21,000 rows, StratifiedGroupKFold(10, seed=42) folds 0..2)
9. round47_seed46077_tune & round47_seed46077_confirm (seller-group split seed 46077)
10. R151_seed42_tune & R151_seed42_confirm (seller-group split seed 42)
11. Documented Round 61/62 historical coverage (68,103 / 70,000 rows covered across rounds 11-62).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark_20261030.split_utils import split_seller_groups_r151, split_seller_groups_r157
from src.intensive_20261030.shared_cache import build_or_load_cache
from src.intensive_20261030.agent3_external_validation import build_virgin_confirm_split

OUT_DIR = ROOT / "results" / "intensive_20261030"


def reconstruct_parent_group_split(y, parent_groups, seed, n_holdout):
    folds = list(
        StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), y, parent_groups
        )
    )
    v = np.concatenate([folds[i][1] for i in range(n_holdout)])
    return set(int(x) for x in v)


def main():
    cache = build_or_load_cache()
    y = cache["y_tr"]
    sellers = cache["sellers_tr"]

    # Load parent_item_id from train_data.jsonlines for parent-grouped historical splits
    train_path = ROOT / "data" / "ap_2026_q_2" / "raw" / "train_data.jsonlines"
    parent_groups = []
    with open(train_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            obj = json.loads(line)
            pid = obj.get("parent_item_id")
            parent_groups.append(str(pid) if pid is not None else f"__ROW__{obj['id']}")
    parent_groups = np.array(parent_groups, dtype=str)

    # 1. Reconstruct all historical holdout sets (0-based row indices)
    hist_holdouts = {}

    for s in [2026, 314159, 20261030]:
        _, t_151, c_151 = split_seller_groups_r151(sellers, s)
        hist_holdouts[f"R151_seed{s}_tune"] = set(int(x) for x in t_151)
        hist_holdouts[f"R151_seed{s}_confirm"] = set(int(x) for x in c_151)

    for s in [2026, 314159]:
        _, t_157, c_157 = split_seller_groups_r157(sellers, s)
        hist_holdouts[f"R157_seed{s}_tune"] = set(int(x) for x in t_157)
        hist_holdouts[f"R157_seed{s}_confirm"] = set(int(x) for x in c_157)

    # Round 47 seed 46077 & seed 42 seller splits
    for s in [46077, 42]:
        _, t_s, c_s = split_seller_groups_r151(sellers, s)
        hist_holdouts[f"SellerSplit_seed{s}_tune"] = set(int(x) for x in t_s)
        hist_holdouts[f"SellerSplit_seed{s}_confirm"] = set(int(x) for x in c_s)

    # Saved CSV holdouts from round_v3 and round_v5
    v3_conf_path = ROOT / "results" / "round_v3" / "prior_confirmation_predictions.csv"
    if v3_conf_path.exists():
        df_v3 = pd.read_csv(v3_conf_path)
        hist_holdouts["round_v3_confirmation_seed731_saved"] = set(int(x) for x in df_v3["row_index"].values)

    v5_sel_path = ROOT / "results" / "round_v5" / "round3_selection_predictions.csv"
    if v5_sel_path.exists():
        df_v5 = pd.read_csv(v5_sel_path)
        hist_holdouts["round_v5_selection_seed42_saved"] = set(int(x) for x in df_v5["row_index"].values)

    # Reconstructed parent_item_id StratifiedGroupKFold holdouts (optimize_leaderboard & confirm_and_submit)
    hist_holdouts["optimize_leaderboard_seed42_holdout3"] = reconstruct_parent_group_split(y, parent_groups, seed=42, n_holdout=3)
    hist_holdouts["confirm_and_submit_seed731_holdout2"] = reconstruct_parent_group_split(y, parent_groups, seed=731, n_holdout=2)

    # Unions of historical holdouts
    union_r151_3seeds_confirm = (
        hist_holdouts["R151_seed2026_confirm"]
        | hist_holdouts["R151_seed314159_confirm"]
        | hist_holdouts["R151_seed20261030_confirm"]
    )
    union_r151_3seeds_tune_confirm = (
        union_r151_3seeds_confirm
        | hist_holdouts["R151_seed2026_tune"]
        | hist_holdouts["R151_seed314159_tune"]
        | hist_holdouts["R151_seed20261030_tune"]
    )
    union_all_reconstructed = set()
    for k, v in hist_holdouts.items():
        union_all_reconstructed |= v

    hist_holdouts["UNION_R151_3Seeds_Confirm"] = union_r151_3seeds_confirm
    hist_holdouts["UNION_R151_3Seeds_TuneAndConfirm"] = union_r151_3seeds_tune_confirm
    hist_holdouts["UNION_All_Reconstructed_Holdouts"] = union_all_reconstructed

    # 2. Reconstruct all 6 evaluation cuts from Agent 2 & Agent 3
    eval_cuts = {
        "R151_seed2026_confirm": hist_holdouts["R151_seed2026_confirm"],
        "R151_seed314159_confirm": hist_holdouts["R151_seed314159_confirm"],
        "R151_seed20261030_confirm": hist_holdouts["R151_seed20261030_confirm"],
        "disjoint_3seed_20261101_confirm": set(int(x) for x in build_virgin_confirm_split(sellers, seed=20261101)[2]),
        "nested_outer_777001_confirm": set(int(x) for x in split_seller_groups_r151(sellers, 777001)[2]),
        "nested_outer_777002_confirm": set(int(x) for x in split_seller_groups_r151(sellers, 777002)[2]),
    }

    matrix = {}
    rows_csv = []
    for cut_name, cut_rows in eval_cuts.items():
        cut_sellers = set(sellers[list(cut_rows)])
        matrix[cut_name] = {
            "n_rows": len(cut_rows),
            "n_sellers": len(cut_sellers),
            "overlaps": {},
        }
        for h_name, h_rows in hist_holdouts.items():
            h_sellers = set(sellers[list(h_rows)])
            ov_r = len(cut_rows & h_rows)
            ov_s = len(cut_sellers & h_sellers)
            pct_r = round(100.0 * ov_r / max(len(cut_rows), 1), 2)
            pct_s = round(100.0 * ov_s / max(len(cut_sellers), 1), 2)
            matrix[cut_name]["overlaps"][h_name] = {
                "hist_rows": len(h_rows),
                "hist_sellers": len(h_sellers),
                "overlap_rows": ov_r,
                "overlap_rows_pct": pct_r,
                "overlap_sellers": ov_s,
                "overlap_sellers_pct": pct_s,
            }
            rows_csv.append({
                "eval_cut": cut_name,
                "eval_rows": len(cut_rows),
                "eval_sellers": len(cut_sellers),
                "historical_holdout": h_name,
                "hist_rows": len(h_rows),
                "hist_sellers": len(h_sellers),
                "overlap_rows": ov_r,
                "overlap_rows_pct": pct_r,
                "overlap_sellers": ov_s,
                "overlap_sellers_pct": pct_s,
            })

    df_out = pd.DataFrame(rows_csv)
    df_out.to_csv(OUT_DIR / "all_historical_holdouts_overlap_audit.csv", index=False)

    payload = {
        "total_train_rows": 70000,
        "total_train_sellers": int(len(np.unique(sellers))),
        "union_all_reconstructed_holdouts_rows": len(union_all_reconstructed),
        "union_all_reconstructed_holdouts_pct": round(100.0 * len(union_all_reconstructed) / 70000.0, 2),
        "round61_62_documented_coverage": {
            "four_recent_splits_union_rows": 61181,
            "with_round11_12_13_union_rows": 65035,
            "seller_exclusive_untouched_rows_remaining": 1897,
            "seller_exclusive_untouched_sellers_remaining": 1647,
            "conclusion": "NO_PRISTINE_CONFIRM_POOL_EXISTS_IN_70K_TRAIN",
        },
        "eval_cuts_overlap_matrix": matrix,
    }
    with open(OUT_DIR / "all_historical_holdouts_overlap_audit.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Saved exhaustive historical holdout overlap audit:")
    print(f"  Union of all reconstructed holdouts in repo: {len(union_all_reconstructed)} / 70,000 rows ({100.0*len(union_all_reconstructed)/70000:.2f}%)")
    for c_name, c_info in matrix.items():
        ov_3c = c_info["overlaps"]["UNION_R151_3Seeds_Confirm"]
        ov_3tc = c_info["overlaps"]["UNION_R151_3Seeds_TuneAndConfirm"]
        ov_all = c_info["overlaps"]["UNION_All_Reconstructed_Holdouts"]
        print(
            f"  {c_name} (N={c_info['n_rows']}): "
            f"vs 3Seeds_Confirm={ov_3c['overlap_rows']} ({ov_3c['overlap_rows_pct']}%), "
            f"vs 3Seeds_Tune+Confirm={ov_3tc['overlap_rows']} ({ov_3tc['overlap_rows_pct']}%), "
            f"vs ALL_Reconstructed={ov_all['overlap_rows']} ({ov_all['overlap_rows_pct']}%)"
        )


if __name__ == "__main__":
    main()
