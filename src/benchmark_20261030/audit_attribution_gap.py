"""Dissect the .924 -> .915 gap between R151 and R157.

Isolates:
1. Split composition effect: evaluate Pipeline A on R151 split vs R157 split.
2. Feature set effect: evaluate Pipeline A features vs Pipeline B (R157) features on the exact same split.
3. TE smoothing effect: alpha 5 vs alpha 10 on the exact same split.
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.revision_submissions.deep_catboost_nbsvm_ablation_20261029 import (
    load_dataset, extract_deep_features, split_seller_groups as split_r151
)
from src.benchmark_20261030.split_utils import split_seller_groups_r157 as split_r157
from src.benchmark_20261030.feature_extractor import compute_leak_free_te_r151, compute_txt_sparse_p

RESULTS_DIR = ROOT / "results" / "benchmark_20261030"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CAT_FEATURES = ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode"]

def train_and_eval_cb(X_fit, y_fit, X_tune, y_tune, X_conf, y_conf, seed=2026):
    cb = CatBoostClassifier(iterations=900, depth=6, learning_rate=0.07, l2_leaf_reg=6.0,
                            random_seed=seed, thread_count=8, verbose=0)
    cb.fit(Pool(X_fit, y_fit, cat_features=CAT_FEATURES))
    p_tune = cb.predict_proba(Pool(X_tune, cat_features=CAT_FEATURES))[:, 1]
    p_conf = cb.predict_proba(Pool(X_conf, cat_features=CAT_FEATURES))[:, 1]
    
    best_th, best_acc = 0.50, 0.0
    for th in np.linspace(0.40, 0.60, 41):
        acc = accuracy_score(y_tune, p_tune >= th)
        if acc > best_acc:
            best_acc = acc
            best_th = th
            
    pred_conf = (p_conf >= best_th).astype(int)
    acc_conf = float(accuracy_score(y_conf, pred_conf))
    ll_conf = float(log_loss(y_conf, p_conf))
    return best_th, best_acc, acc_conf, ll_conf, p_conf, pred_conf

def main():
    print("Loading dataset...", flush=True)
    rows, ids, sellers, y_used = load_dataset()
    print("Extracting deep R151 features...", flush=True)
    F_df, cat_df, titles, warranties, combos = extract_deep_features(rows)
    
    records = []
    
    for seed in [2026, 314159]:
        print(f"\n==================== SEED {seed} ====================")
        
        # 1. Pipeline A on R151 canonical split (reproduction of R151 base)
        fit_151, tune_151, conf_151 = split_r151(sellers, seed)
        sellers_fit = sellers[fit_151]
        
        te_fit_a, (te_tu_a, te_co_a) = compute_leak_free_te_r151(cat_df, fit_151, [tune_151, conf_151], y_used[fit_151], sellers_fit, alpha=5.0)
        oof_txt_a, (p_tu_txt_a, p_co_txt_a) = compute_txt_sparse_p(combos, titles, fit_151, [tune_151, conf_151], y_used[fit_151], sellers_fit, seed=seed)
        
        X_fit_a = F_df.iloc[fit_151].copy()
        X_tu_a = F_df.iloc[tune_151].copy()
        X_co_a = F_df.iloc[conf_151].copy()
        
        for c in ["seller_id", "category_id", "city"]:
            X_fit_a[f"te_{c}"] = te_fit_a[c]
            X_tu_a[f"te_{c}"] = te_tu_a[c]
            X_co_a[f"te_{c}"] = te_co_a[c]
            
        X_fit_a["txt_sparse_p"] = oof_txt_a
        X_tu_a["txt_sparse_p"] = p_tu_txt_a
        X_co_a["txt_sparse_p"] = p_co_txt_a
        
        for c in CAT_FEATURES:
            X_fit_a[c] = X_fit_a[c].astype(str)
            X_tu_a[c] = X_tu_a[c].astype(str)
            X_co_a[c] = X_co_a[c].astype(str)
            
        print("Training Pipeline A on R151 split (Exact R151 Base)...", flush=True)
        th_a, tu_acc_a, conf_acc_a, conf_ll_a, p_co_a, pred_co_a = train_and_eval_cb(
            X_fit_a, y_used[fit_151], X_tu_a, y_used[tune_151], X_co_a, y_used[conf_151], seed=seed
        )
        print(f"  -> Pipeline A Confirm: Acc = {conf_acc_a:.5f} (Tune th = {th_a:.2f}), LogLoss = {conf_ll_a:.5f}")
        
        # Check agreement with paired_deep_ablation_seed_*.csv
        f_hist = ROOT / "results" / "revision_submissions" / f"paired_deep_ablation_seed_{seed}.csv"
        if f_hist.exists():
            df_hist = pd.read_csv(f_hist)
            agree = np.mean(df_hist["pred_base"].values == pred_co_a)
            corr = np.corrcoef(df_hist["p_base"].values, p_co_a)[0, 1]
            print(f"  -> Exact Match with historical paired_deep_ablation: Agreement={agree*100:.2f}%, Pearson Corr={corr:.6f}")

        # Save predictions
        df_co_out = pd.DataFrame({
            "row_idx": conf_151,
            "id": conf_151 + 1,
            "item_id": ids[conf_151],
            "seller_id": sellers[conf_151],
            "y_used": y_used[conf_151],
            "is_seen": np.isin(sellers[conf_151], list(set(sellers_fit))).astype(int),
            "p_used": p_co_a,
            "pred": pred_co_a,
            "threshold": th_a,
        })
        p_out = RESULTS_DIR / f"preds_A_seed{seed}_confirm.csv.gz"
        df_co_out.to_csv(p_out, index=False, compression="gzip")
        print(f"  -> Saved {p_out}")

        records.append({
            "seed": seed,
            "pipeline": "Pipeline A (R151 Base)",
            "split": "R151 split_seller_groups",
            "features": "R151 full deep (101 feats)",
            "te_alpha": 5.0,
            "tune_acc": tu_acc_a,
            "confirm_acc": conf_acc_a,
            "confirm_ll": conf_ll_a,
        })

        # 2. Pipeline A on R157 split (to isolate pure split composition effect)
        fit_157, tune_157, conf_157 = split_r157(sellers, seed)
        sellers_fit_157 = sellers[fit_157]
        
        te_fit_split, (te_tu_split, te_co_split) = compute_leak_free_te_r151(cat_df, fit_157, [tune_157, conf_157], y_used[fit_157], sellers_fit_157, alpha=5.0)
        oof_txt_split, (p_tu_txt_split, p_co_txt_split) = compute_txt_sparse_p(combos, titles, fit_157, [tune_157, conf_157], y_used[fit_157], sellers_fit_157, seed=seed)
        
        X_fit_split = F_df.iloc[fit_157].copy()
        X_tu_split = F_df.iloc[tune_157].copy()
        X_co_split = F_df.iloc[conf_157].copy()
        for c in ["seller_id", "category_id", "city"]:
            X_fit_split[f"te_{c}"] = te_fit_split[c]
            X_tu_split[f"te_{c}"] = te_tu_split[c]
            X_co_split[f"te_{c}"] = te_co_split[c]
        X_fit_split["txt_sparse_p"] = oof_txt_split
        X_tu_split["txt_sparse_p"] = p_tu_txt_split
        X_co_split["txt_sparse_p"] = p_co_txt_split
        for c in CAT_FEATURES:
            X_fit_split[c] = X_fit_split[c].astype(str)
            X_tu_split[c] = X_tu_split[c].astype(str)
            X_co_split[c] = X_co_split[c].astype(str)
            
        print("Training Pipeline A on R157 split (Split Composition Ablation)...", flush=True)
        th_s, tu_acc_s, conf_acc_s, conf_ll_s, _, _ = train_and_eval_cb(
            X_fit_split, y_used[fit_157], X_tu_split, y_used[tune_157], X_co_split, y_used[conf_157], seed=seed
        )
        print(f"  -> Pipeline A on R157 split Confirm: Acc = {conf_acc_s:.5f}, LogLoss = {conf_ll_s:.5f}")
        records.append({
            "seed": seed,
            "pipeline": "Pipeline A (on R157 split)",
            "split": "R157 canonical_split",
            "features": "R151 full deep (101 feats)",
            "te_alpha": 5.0,
            "tune_acc": tu_acc_s,
            "confirm_acc": conf_acc_s,
            "confirm_ll": conf_ll_s,
        })

        # 3. Pipeline A with TE alpha=10 (to isolate TE alpha effect on R151 split)
        te_fit_a10, (te_tu_a10, te_co_a10) = compute_leak_free_te_r151(cat_df, fit_151, [tune_151, conf_151], y_used[fit_151], sellers_fit, alpha=10.0)
        X_fit_a10 = X_fit_a.copy()
        X_tu_a10 = X_tu_a.copy()
        X_co_a10 = X_co_a.copy()
        for c in ["seller_id", "category_id", "city"]:
            X_fit_a10[f"te_{c}"] = te_fit_a10[c]
            X_tu_a10[f"te_{c}"] = te_tu_a10[c]
            X_co_a10[f"te_{c}"] = te_co_a10[c]
            
        print("Training Pipeline A with TE alpha=10.0 (TE Alpha Ablation)...", flush=True)
        th_a10, tu_acc_a10, conf_acc_a10, conf_ll_a10, _, _ = train_and_eval_cb(
            X_fit_a10, y_used[fit_151], X_tu_a10, y_used[tune_151], X_co_a10, y_used[conf_151], seed=seed
        )
        print(f"  -> Pipeline A with alpha=10 Confirm: Acc = {conf_acc_a10:.5f}, LogLoss = {conf_ll_a10:.5f}")
        records.append({
            "seed": seed,
            "pipeline": "Pipeline A (TE alpha=10)",
            "split": "R151 split_seller_groups",
            "features": "R151 full deep (101 feats)",
            "te_alpha": 10.0,
            "tune_acc": tu_acc_a10,
            "confirm_acc": conf_acc_a10,
            "confirm_ll": conf_ll_a10,
        })

    df_res = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "attribution_gap_table.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\nSaved attribution table to {csv_path}")
    print(df_res.to_string())

if __name__ == "__main__":
    main()
