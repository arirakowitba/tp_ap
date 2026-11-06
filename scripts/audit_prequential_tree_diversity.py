"""Prequential and Deduplicated Audit of Tree Diversity Ensembles.

Loads paired prediction CSVs across multiple historical cuts,
analyzes item overlap, performs deduplication under both chronological
and reverse priorities, and computes seller-clustered bootstrap CI.
"""

import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "tree_diversity_clean_te"


def run_bootstrap_seller(y_true, pred_ctrl, pred_cand, sellers, n_boot=3000, seed=42):
    rng = np.random.default_rng(seed)
    unique_sellers = np.unique(sellers)
    n_sellers = len(unique_sellers)
    seller_to_indices = {s: np.where(sellers == s)[0] for s in unique_sellers}

    diffs = []
    for _ in range(n_boot):
        sample_sellers = rng.choice(unique_sellers, size=n_sellers, replace=True)
        idx = np.concatenate([seller_to_indices[s] for s in sample_sellers])
        acc_a = (pred_ctrl[idx] == y_true[idx]).sum()
        acc_b = (pred_cand[idx] == y_true[idx]).sum()
        diffs.append(acc_b - acc_a)

    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def evaluate_set(df, pred_cand_col, pred_ctrl_col, seed=42):
    y_true = df["y_true"].to_numpy()
    pred_cand = df[pred_cand_col].to_numpy()
    pred_ctrl = df[pred_ctrl_col].to_numpy()
    sellers = df["seller_id"].to_numpy()
    is_seen = df["is_seen"].to_numpy().astype(bool)

    hits_cand = int((pred_cand == y_true).sum())
    hits_ctrl = int((pred_ctrl == y_true).sum())
    delta = hits_cand - hits_ctrl
    fixes = int(((pred_ctrl != y_true) & (pred_cand == y_true)).sum())
    breaks = int(((pred_ctrl == y_true) & (pred_cand != y_true)).sum())
    disagreements = int((pred_ctrl != pred_cand).sum())

    net_seen = int(((pred_cand[is_seen] == y_true[is_seen]).sum()) - ((pred_ctrl[is_seen] == y_true[is_seen]).sum()))
    net_new = int(((pred_cand[~is_seen] == y_true[~is_seen]).sum()) - ((pred_ctrl[~is_seen] == y_true[~is_seen]).sum()))
    ci_low, ci_high = run_bootstrap_seller(y_true, pred_ctrl, pred_cand, sellers, n_boot=3000, seed=seed)

    return {
        "n_rows": len(df),
        "hits_cand": hits_cand,
        "hits_ctrl": hits_ctrl,
        "acc_cand": float(accuracy_score(y_true, pred_cand)),
        "acc_ctrl": float(accuracy_score(y_true, pred_ctrl)),
        "delta": delta,
        "fixes": fixes,
        "breaks": breaks,
        "disagreements": disagreements,
        "net_seen": net_seen,
        "net_new": net_new,
        "ci_95": [ci_low, ci_high]
    }


def main():
    cuts = [20261143, 20261144]
    dfs = []
    for c in cuts:
        f = RESULTS_DIR / f"paired_trees_diversity_clean_te_{c}_bf0.csv"
        if not f.exists():
            print(f"File not found: {f}")
            return
        df = pd.read_csv(f)
        df["cut_seed"] = c
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    n_total = len(df_all)
    n_unique = df_all["item_id"].nunique()
    n_overlap = n_total - n_unique
    print(f"Total rows: {n_total}, Unique items: {n_unique}, Overlap: {n_overlap} ({n_overlap/n_total*100:.2f}%)")

    # Add binary prediction columns at common 0.500
    df_all["pred_cb_05"] = (df_all["p_cb_bag3"] >= 0.50).astype(int)
    df_all["pred_ens8020_05"] = (df_all["p_ens8020"] >= 0.50).astype(int)
    df_all["pred_ens9010_05"] = (df_all["p_ens9010"] >= 0.50).astype(int)

    # 1. Raw Pooled (with duplicates)
    raw_05_8020 = evaluate_set(df_all, "pred_ens8020_05", "pred_cb_05")
    raw_05_9010 = evaluate_set(df_all, "pred_ens9010_05", "pred_cb_05")

    # 2. Deduplicated (Priority: Chronological First)
    df_dedup_first = df_all.drop_duplicates(subset=["item_id"], keep="first")
    dedup_first_05_8020 = evaluate_set(df_dedup_first, "pred_ens8020_05", "pred_cb_05")
    dedup_first_05_9010 = evaluate_set(df_dedup_first, "pred_ens9010_05", "pred_cb_05")

    # 3. Deduplicated (Priority: Latest First)
    df_dedup_last = df_all.drop_duplicates(subset=["item_id"], keep="last")
    dedup_last_05_8020 = evaluate_set(df_dedup_last, "pred_ens8020_05", "pred_cb_05")
    dedup_last_05_9010 = evaluate_set(df_dedup_last, "pred_ens9010_05", "pred_cb_05")

    # Logloss
    ll_cb = float(log_loss(df_dedup_first["y_true"], df_dedup_first["p_cb_bag3"]))
    ll_8020 = float(log_loss(df_dedup_first["y_true"], df_dedup_first["p_ens8020"]))
    ll_9010 = float(log_loss(df_dedup_first["y_true"], df_dedup_first["p_ens9010"]))

    summary = {
        "cuts": cuts,
        "n_total_evaluations": n_total,
        "n_unique_items": n_unique,
        "n_overlap_items": n_overlap,
        "logloss_dedup_first": {
            "cb_bag3": ll_cb,
            "ens8020": ll_8020,
            "ens9010": ll_9010,
            "delta_8020": ll_8020 - ll_cb,
            "delta_9010": ll_9010 - ll_cb
        },
        "raw_pooled_050": {
            "ens8020": raw_05_8020,
            "ens9010": raw_05_9010
        },
        "dedup_chronological_first_050": {
            "ens8020": dedup_first_05_8020,
            "ens9010": dedup_first_05_9010
        },
        "dedup_latest_last_050": {
            "ens8020": dedup_last_05_8020,
            "ens9010": dedup_last_05_9010
        }
    }

    out_file = RESULTS_DIR / "report_prequential_tree_diversity.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
