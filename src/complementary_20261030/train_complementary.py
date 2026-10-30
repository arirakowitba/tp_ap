"""Complementary Model Training: Deep CatBoost + Leaf-Wise LightGBM Ensemble.

Evaluated strictly on canonical R151 partitions:
- Models:
  1. Base: Deep CatBoost (oblivious trees, 101 deep features + leak-free TE alpha=5.0 + leak-free txt_sparse_p)
  2. Complement: Leaf-Wise LightGBM on the exact same features
- Protocol:
  * Blend weight w* and threshold th* frozen strictly on tune holdout.
  * Evaluated on confirm ONCE.
  * Matched-pair statistics computed via paired_eval.py.
  * Seeds: 2026, 314159, and predeclared 20261030.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.revision_submissions.deep_catboost_nbsvm_ablation_20261029 import (
    load_dataset, extract_deep_features, split_seller_groups as split_r151
)
from src.benchmark_20261030.feature_extractor import compute_leak_free_te_r151, compute_txt_sparse_p
from src.eval_20261030.paired_eval import evaluate_paired_frames, format_markdown_report

RESULTS_DIR = ROOT / "results" / "complementary_20261030"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CAT_FEATURES = ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode"]

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def train_catboost_base(X_fit, y_fit, X_tu, y_tu, X_co, y_co, seed=2026):
    cb = CatBoostClassifier(
        iterations=900,
        depth=6,
        learning_rate=0.07,
        l2_leaf_reg=6.0,
        random_seed=seed,
        thread_count=8,
        verbose=0
    )
    cb.fit(Pool(X_fit, y_fit, cat_features=CAT_FEATURES))
    p_tu_cb = cb.predict_proba(Pool(X_tu, cat_features=CAT_FEATURES))[:, 1]
    p_co_cb = cb.predict_proba(Pool(X_co, cat_features=CAT_FEATURES))[:, 1]
    
    # Tune threshold
    best_th, best_acc = 0.50, 0.0
    for th in np.linspace(0.40, 0.60, 41):
        acc = accuracy_score(y_tu, p_tu_cb >= th)
        if acc > best_acc:
            best_acc = acc
            best_th = th
            
    pred_co_cb = (p_co_cb >= best_th).astype(int)
    acc_co = float(accuracy_score(y_co, pred_co_cb))
    ll_co = float(log_loss(y_co, p_co_cb))
    return cb, p_tu_cb, p_co_cb, best_th, best_acc, acc_co, ll_co

def train_lightgbm_complementary(X_fit, y_fit, X_tu, y_tu, X_co, y_co, seed=2026):
    # Prepare data for LightGBM
    X_fit_lgb = X_fit.copy()
    X_tu_lgb = X_tu.copy()
    X_co_lgb = X_co.copy()
    
    for c in CAT_FEATURES:
        X_fit_lgb[c] = X_fit_lgb[c].astype("category")
        X_tu_lgb[c] = X_tu_lgb[c].astype("category")
        X_co_lgb[c] = X_co_lgb[c].astype("category")
        
    lgbm = lgb.LGBMClassifier(
        n_estimators=900,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed,
        n_jobs=8,
        verbose=-1
    )
    lgbm.fit(X_fit_lgb, y_fit)
    p_tu_lgb = lgbm.predict_proba(X_tu_lgb)[:, 1]
    p_co_lgb = lgbm.predict_proba(X_co_lgb)[:, 1]
    
    best_th, best_acc = 0.50, 0.0
    for th in np.linspace(0.40, 0.60, 41):
        acc = accuracy_score(y_tu, p_tu_lgb >= th)
        if acc > best_acc:
            best_acc = acc
            best_th = th
            
    pred_co_lgb = (p_co_lgb >= best_th).astype(int)
    acc_co = float(accuracy_score(y_co, pred_co_lgb))
    ll_co = float(log_loss(y_co, p_co_lgb))
    return lgbm, p_tu_lgb, p_co_lgb, best_th, best_acc, acc_co, ll_co

def tune_blend_and_evaluate(p_tu_cb, p_tu_lgb, y_tu, p_co_cb, p_co_lgb, y_co):
    best_w = 0.5
    best_th = 0.5
    best_acc = 0.0
    
    weights = np.linspace(0.0, 1.0, 21)
    thresholds = np.linspace(0.40, 0.60, 41)
    
    for w in weights:
        p_tu_blend = w * p_tu_cb + (1.0 - w) * p_tu_lgb
        for th in thresholds:
            acc = accuracy_score(y_tu, p_tu_blend >= th)
            if acc > best_acc:
                best_acc = acc
                best_w = w
                best_th = th
                
    # Freeze best_w and best_th, evaluate ONCE on confirm
    p_co_blend = best_w * p_co_cb + (1.0 - best_w) * p_co_lgb
    pred_co_blend = (p_co_blend >= best_th).astype(int)
    
    acc_co = float(accuracy_score(y_co, pred_co_blend))
    ll_co = float(log_loss(y_co, p_co_blend))
    
    return best_w, best_th, best_acc, acc_co, ll_co, p_co_blend, pred_co_blend

def run_experiment_for_seed(seed, F_df, cat_df, combos, titles, sellers, y_used, ids):
    print(f"\n==================================================", flush=True)
    print(f"RUNNING COMPLEMENTARY EXPERIMENT SEED {seed}", flush=True)
    print(f"==================================================", flush=True)
    
    fit_idx, tu_idx, co_idx = split_r151(sellers, seed)
    sellers_fit = sellers[fit_idx]
    
    # 1. Compute leak-free TE (alpha=5.0)
    te_fit, (te_tu, te_co) = compute_leak_free_te_r151(
        cat_df, fit_idx, [tu_idx, co_idx], y_used[fit_idx], sellers_fit, alpha=5.0
    )
    
    # 2. Compute leak-free txt_sparse_p
    oof_txt, (p_tu_txt, p_co_txt) = compute_txt_sparse_p(
        combos, titles, fit_idx, [tu_idx, co_idx], y_used[fit_idx], sellers_fit, seed=seed
    )
    
    X_fit = F_df.iloc[fit_idx].copy()
    X_tu = F_df.iloc[tu_idx].copy()
    X_co = F_df.iloc[co_idx].copy()
    
    for c in ["seller_id", "category_id", "city"]:
        X_fit[f"te_{c}"] = te_fit[c]
        X_tu[f"te_{c}"] = te_tu[c]
        X_co[f"te_{c}"] = te_co[c]
        
    X_fit["txt_sparse_p"] = oof_txt
    X_tu["txt_sparse_p"] = p_tu_txt
    X_co["txt_sparse_p"] = p_co_txt
    
    for c in CAT_FEATURES:
        X_fit[c] = X_fit[c].astype(str)
        X_tu[c] = X_tu[c].astype(str)
        X_co[c] = X_co[c].astype(str)
        
    y_fit = y_used[fit_idx]
    y_tu = y_used[tu_idx]
    y_co = y_used[co_idx]
    
    # Train Base CatBoost
    print("Training Base Deep CatBoost...", flush=True)
    cb_model, p_tu_cb, p_co_cb, th_cb, tu_acc_cb, co_acc_cb, co_ll_cb = train_catboost_base(
        X_fit, y_fit, X_tu, y_tu, X_co, y_co, seed=seed
    )
    print(f"  Base CatBoost: Tune Acc = {tu_acc_cb:.5f} (th={th_cb:.2f}) | Confirm Acc = {co_acc_cb:.5f}, LL = {co_ll_cb:.5f}", flush=True)
    
    # Train Complementary LightGBM
    print("Training Complementary Leaf-Wise LightGBM...", flush=True)
    lgb_model, p_tu_lgb, p_co_lgb, th_lgb, tu_acc_lgb, co_acc_lgb, co_ll_lgb = train_lightgbm_complementary(
        X_fit, y_fit, X_tu, y_tu, X_co, y_co, seed=seed
    )
    print(f"  LightGBM:      Tune Acc = {tu_acc_lgb:.5f} (th={th_lgb:.2f}) | Confirm Acc = {co_acc_lgb:.5f}, LL = {co_ll_lgb:.5f}", flush=True)
    
    # Optimize blend on tune and evaluate confirm
    w_star, th_star, tu_acc_blend, co_acc_blend, co_ll_blend, p_co_blend, pred_co_blend = tune_blend_and_evaluate(
        p_tu_cb, p_tu_lgb, y_tu, p_co_cb, p_co_lgb, y_co
    )
    print(f"  Ensemble:      Tune Acc = {tu_acc_blend:.5f} (w*={w_star:.2f}, th*={th_star:.2f}) | Confirm Acc = {co_acc_blend:.5f}, LL = {co_ll_blend:.5f}", flush=True)
    
    # Prepare paired dataframe for paired_eval
    pred_co_cb = (p_co_cb >= th_cb).astype(int)
    is_seen_co = np.isin(sellers[co_idx], list(set(sellers_fit))).astype(int)
    
    df_pair = pd.DataFrame({
        "id": co_idx + 1,
        "item_id": ids[co_idx],
        "seller_id": sellers[co_idx],
        "category_id": cat_df.iloc[co_idx]["category_id"].values,
        "is_seen": is_seen_co,
        "y_true": y_co,
        "p_base": p_co_cb,
        "pred_base": pred_co_cb,
        "p_cand": p_co_blend,
        "pred_cand": pred_co_blend,
    })
    
    # Save paired CSV
    pair_csv = RESULTS_DIR / f"paired_cb_lgb_seed_{seed}.csv.gz"
    df_pair.to_csv(pair_csv, index=False, compression="gzip")
    print(f"  Saved paired predictions to {pair_csv}", flush=True)
    
    # Run matched-pair evaluation
    df_base = pd.DataFrame({
        "id": df_pair["id"],
        "seller_id": df_pair["seller_id"],
        "is_seen": df_pair["is_seen"],
        "y_used": df_pair["y_true"],
        "p_used": df_pair["p_base"],
        "pred": df_pair["pred_base"]
    })
    df_cand = pd.DataFrame({
        "id": df_pair["id"],
        "seller_id": df_pair["seller_id"],
        "is_seen": df_pair["is_seen"],
        "y_used": df_pair["y_true"],
        "p_used": df_pair["p_cand"],
        "pred": df_pair["pred_cand"]
    })
    metrics = evaluate_paired_frames(df_base, df_cand)
    
    return {
        "seed": seed,
        "w_star": w_star,
        "th_star": th_star,
        "tu_acc_cb": tu_acc_cb,
        "co_acc_cb": co_acc_cb,
        "co_ll_cb": co_ll_cb,
        "tu_acc_lgb": tu_acc_lgb,
        "co_acc_lgb": co_acc_lgb,
        "co_ll_lgb": co_ll_lgb,
        "tu_acc_blend": tu_acc_blend,
        "co_acc_blend": co_acc_blend,
        "co_ll_blend": co_ll_blend,
        "net_delta": metrics["net"],
        "ci_lower": metrics["bootstrap_seller_ci95"][0],
        "ci_upper": metrics["bootstrap_seller_ci95"][1],
        "seen_net": metrics["seen_sellers"]["net"],
        "new_net": metrics["new_sellers"]["net"],
        "pair_sha256": compute_sha256(pair_csv),
        "metrics": metrics
    }

def main():
    print("Loading 70k train dataset...", flush=True)
    rows, ids, sellers, y_used = load_dataset()
    print("Extracting deep R151 features...", flush=True)
    F_df, cat_df, titles, warranties, combos = extract_deep_features(rows)
    
    results = []
    for seed in [2026, 314159, 20261030]:
        res = run_experiment_for_seed(seed, F_df, cat_df, combos, titles, sellers, y_used, ids)
        results.append(res)
        
    df_summary = pd.DataFrame([{
        "seed": r["seed"],
        "w_cb": r["w_star"],
        "th": r["th_star"],
        "cb_confirm_acc": r["co_acc_cb"],
        "cb_confirm_ll": r["co_ll_cb"],
        "lgb_confirm_acc": r["co_acc_lgb"],
        "lgb_confirm_ll": r["co_ll_lgb"],
        "blend_confirm_acc": r["co_acc_blend"],
        "blend_confirm_ll": r["co_ll_blend"],
        "net_delta": r["net_delta"],
        "ci_95": f"[{r['ci_lower']:+.1f}, {r['ci_upper']:+.1f}]",
        "seen_net": r["seen_net"],
        "new_net": r["new_net"],
        "pair_sha256": r["pair_sha256"][:12] + "..."
    } for r in results])
    
    summary_path = RESULTS_DIR / "complementary_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print("\n==================== COMPLEMENTARY SUMMARY ====================")
    print(df_summary.to_string())
    
    # Write COMPLEMENTARY_REPORT.md
    md_lines = [
        "# Complementary Model Experiment Report: Deep CatBoost + Leaf-Wise LightGBM",
        "",
        "## 1. Experimental Protocol",
        "- **Base Model**: Deep CatBoost (oblivious trees, depth 6, lr 0.07, l2 6.0, 900 iters) on 101 deep features + leak-free TE (alpha=5.0) + leak-free txt_sparse_p.",
        "- **Complementary Model**: Leaf-Wise LightGBM (num_leaves 31, min_child_samples 30, lr 0.04, 900 iters) on the exact same feature matrix.",
        "- **Ensemble Formulation**: p_blend = w* * p_cb + (1 - w*) * p_lgb.",
        "- **Strict Leakage Prevention**: w* and decision threshold th* chosen strictly on `tune` holdout; evaluated on `confirm` ONCE.",
        "- **Statistical Significance**: Matched-pair evaluation with 2,000 cluster-grouped seller bootstrap resamples (95% CI).",
        "",
        "## 2. Multi-Seed Results Summary",
        "",
        df_summary.to_markdown(index=False),
        "",
        "## 3. Matched-Pair Detailed Analysis",
        ""
    ]
    
    for r in results:
        md_lines.append(f"### Seed {r['seed']}")
        md_lines.append(format_markdown_report(r["metrics"], base_name=f"Deep_CatBoost_Seed_{r['seed']}", cand_name=f"CB_LGB_Blend_Seed_{r['seed']}"))
        md_lines.append("")
        
    report_path = RESULTS_DIR / "COMPLEMENTARY_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"\nSaved report to {report_path}")

if __name__ == "__main__":
    main()
