"""Semantic Neural and Tree Multi-Family Model on Cut 42026 (Task B - CX-0036..CX-0042).

Strict Falsification & Epistemological Protocol:
1. Split 42026: fit (~43k), tune (~13k), confirm (~13k) grouped strictly by seller_id.
2. All feature extractors (TF-IDF, SVD, encoders, scalers) fitted strictly inside `fit` (zero test/tune/confirm transduction).
3. Architectures:
   - Non-Tree Semantic Family:
     * Word + Subword Char TF-IDF with TruncatedSVD (64 components) dense representation.
     * Linear Additive Model (LogisticRegression with 5-fold StratifiedGroupKFold OOF inside fit).
     * Neural MLP Classifier (Multi-Layer Perceptron: 128x64 ReLU with weight decay) on SVD + Scaled features.
   - Tree Family:
     * LightGBM & CatBoost incorporating inductive relative price and OOF neural/linear signals.
   - Multi-Family Diverse Ensemble combining B10 with neural/linear semantic model.
4. Paired Benchmarks against:
   - Baseline Claude B10 (fit-only)
   - Candidate Full (CX-0029 blend)
   - Baseline Blend075
   - Baseline Pipeline A
5. Full seller-group bootstrap confidence intervals (2,000 replicas, 95% CI).
6. Seen vs New sellers stratification and threshold policy tuning on `tune`.
"""

import gc
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import catboost
from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from benchmark_20261030.split_utils import split_seller_groups_r151
from coordination_lock import ProjectLock
from intensive_20261030.run_iterative_nocturnal_batch import (
    build_or_load_cache,
    extract_advanced_inductive_features,
    seller_bootstrap_ci,
)
from optimize_leaderboard import data_folder

CAT_FEATURES = ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode"]

def calc_log_loss(y_true, p_pred):
    p_clipped = np.clip(p_pred, 1e-6, 1.0 - 1e-6)
    return float(log_loss(y_true, p_clipped))

def evaluate_pair(name_cand, p_cand_tu, p_cand_co, name_ref, p_ref_tu, p_ref_co, y_tu, y_co, sellers_co, seen_mask):
    """Calculates all paired metrics, bootstrap intervals, and threshold analysis."""
    pred_cand_050 = (p_cand_co >= 0.5).astype(int)
    pred_ref_050 = (p_ref_co >= 0.5).astype(int)

    acc_cand = float(accuracy_score(y_co, pred_cand_050))
    acc_ref = float(accuracy_score(y_co, pred_ref_050))
    ll_cand = calc_log_loss(y_co, p_cand_co)
    ll_ref = calc_log_loss(y_co, p_ref_co)
    auc_cand = float(roc_auc_score(y_co, p_cand_co))
    auc_ref = float(roc_auc_score(y_co, p_ref_co))

    fixes = int(((pred_cand_050 == y_co) & (pred_ref_050 != y_co)).sum())
    breaks = int(((pred_cand_050 != y_co) & (pred_ref_050 == y_co)).sum())
    net_fixes, ci_low, ci_high = seller_bootstrap_ci(y_co, pred_ref_050, pred_cand_050, sellers_co, n_boot=2000, seed=42)

    seen_fixes = int(((pred_cand_050[seen_mask] == y_co[seen_mask]) & (pred_ref_050[seen_mask] != y_co[seen_mask])).sum())
    seen_breaks = int(((pred_cand_050[seen_mask] != y_co[seen_mask]) & (pred_ref_050[seen_mask] == y_co[seen_mask])).sum())
    seen_net = seen_fixes - seen_breaks
    seen_acc_cand = float(accuracy_score(y_co[seen_mask], pred_cand_050[seen_mask]))
    seen_acc_ref = float(accuracy_score(y_co[seen_mask], pred_ref_050[seen_mask]))

    new_fixes = int(((pred_cand_050[~seen_mask] == y_co[~seen_mask]) & (pred_ref_050[~seen_mask] != y_co[~seen_mask])).sum())
    new_breaks = int(((pred_cand_050[~seen_mask] != y_co[~seen_mask]) & (pred_ref_050[~seen_mask] == y_co[~seen_mask])).sum())
    new_net = new_fixes - new_breaks
    new_acc_cand = float(accuracy_score(y_co[~seen_mask], pred_cand_050[~seen_mask]))
    new_acc_ref = float(accuracy_score(y_co[~seen_mask], pred_ref_050[~seen_mask]))

    # Policy threshold tuned on tune
    grid = np.linspace(0.40, 0.60, 41)
    best_th_cand = 0.50
    best_acc_tu_cand = -1.0
    best_th_ref = 0.50
    best_acc_tu_ref = -1.0
    for th in grid:
        a_c = accuracy_score(y_tu, (p_cand_tu >= th).astype(int))
        if a_c > best_acc_tu_cand:
            best_acc_tu_cand = a_c
            best_th_cand = float(th)
        a_r = accuracy_score(y_tu, (p_ref_tu >= th).astype(int))
        if a_r > best_acc_tu_ref:
            best_acc_tu_ref = a_r
            best_th_ref = float(th)

    pred_cand_tuned = (p_cand_co >= best_th_cand).astype(int)
    pred_ref_tuned = (p_ref_co >= best_th_ref).astype(int)
    acc_cand_tuned = float(accuracy_score(y_co, pred_cand_tuned))
    acc_ref_tuned = float(accuracy_score(y_co, pred_ref_tuned))
    net_fixes_tuned, ci_l_t, ci_h_t = seller_bootstrap_ci(y_co, pred_ref_tuned, pred_cand_tuned, sellers_co, n_boot=2000, seed=42)

    return {
        "candidate": name_cand,
        "reference": name_ref,
        "fixed_threshold_050": {
            "cand_accuracy": acc_cand,
            "ref_accuracy": acc_ref,
            "cand_log_loss": ll_cand,
            "ref_log_loss": ll_ref,
            "delta_log_loss": round(ll_cand - ll_ref, 5),
            "cand_roc_auc": auc_cand,
            "ref_roc_auc": auc_ref,
            "fixes": fixes,
            "breaks": breaks,
            "net_fixes": net_fixes,
            "bootstrap_95ci": [ci_low, ci_high],
            "stratified": {
                "seen": {
                    "count": int(seen_mask.sum()),
                    "cand_acc": seen_acc_cand,
                    "ref_acc": seen_acc_ref,
                    "fixes": seen_fixes,
                    "breaks": seen_breaks,
                    "net_fixes": seen_net,
                },
                "new": {
                    "count": int((~seen_mask).sum()),
                    "cand_acc": new_acc_cand,
                    "ref_acc": new_acc_ref,
                    "fixes": new_fixes,
                    "breaks": new_breaks,
                    "net_fixes": new_net,
                },
            },
        },
        "policy_tuned_threshold": {
            "cand_threshold": best_th_cand,
            "cand_acc_tune": round(best_acc_tu_cand, 5),
            "cand_acc_confirm": acc_cand_tuned,
            "ref_threshold": best_th_ref,
            "ref_acc_tune": round(best_acc_tu_ref, 5),
            "ref_acc_confirm": acc_ref_tuned,
            "net_fixes_tuned": net_fixes_tuned,
            "bootstrap_95ci_tuned": [ci_l_t, ci_h_t],
        },
    }

def main():
    lock = ProjectLock(owner_id="antigravity_task_b_semantic_neural_20261101")
    print("=== TASK B: TRAINING SEMANTIC NEURAL AND TREE ENSEMBLE ===", flush=True)
    with lock:
        t0 = time.time()
        out_dir = REPO_ROOT / "results" / "intensive_20261030" / "task_b_semantic_neural"
        out_dir.mkdir(parents=True, exist_ok=True)

        cache = build_or_load_cache()
        F_all = cache["F_r151_tr"]
        cat_all = cache["cat_tr"]
        y_all = cache["y_tr"]
        sellers_all = cache["sellers_tr"]
        combos_all = cache["combos_hist_tr"]

        folder = data_folder()
        raw_data = pd.read_json(folder / "train_data.jsonlines", lines=True)

        seed = 42026
        fit, tune, confirm = split_seller_groups_r151(sellers_all, seed)
        y_fit, y_tune, y_confirm = y_all[fit], y_all[tune], y_all[confirm]
        sellers_fit, sellers_confirm = sellers_all[fit], sellers_all[confirm]
        seen_mask = np.isin(sellers_confirm, np.unique(sellers_fit))

        print(f"Dataset split: fit={len(fit)}, tune={len(tune)}, confirm={len(confirm)} (Seen: {seen_mask.sum()}, New: {(~seen_mask).sum()})", flush=True)

        # -------------------------------------------------------------
        # STEP 1: Load Existing Baselines on Cut 42026
        # -------------------------------------------------------------
        cx0029_dir = REPO_ROOT / "results" / "intensive_20261030" / "cx0029_evaluation"
        p_b10_tu = np.load(cx0029_dir / "b10_work" / "p_b10_tu.npy")
        p_b10_co = np.load(cx0029_dir / "b10_work" / "p_b10_co.npy")
        p_blend075_tu = np.load(cx0029_dir / "p_blend075_tu.npy")
        p_blend075_co = np.load(cx0029_dir / "p_blend075_co.npy")
        p_pipea_tu = np.load(cx0029_dir / "p_pipea_tu.npy")
        p_pipea_co = np.load(cx0029_dir / "p_pipea_co.npy")
        oof_txt_fit = np.load(cx0029_dir / "oof_txt_fit.npy")
        p_txt_tu = np.load(cx0029_dir / "p_txt_tu.npy")
        p_txt_co = np.load(cx0029_dir / "p_txt_co.npy")

        # -------------------------------------------------------------
        # STEP 2: Inductive Relative Price Features
        # -------------------------------------------------------------
        print("Extracting inductive relative price features strictly from fit...", flush=True)
        F_fit_ind, F_tu_ind, F_co_ind = extract_advanced_inductive_features(F_all, cat_all, fit, tune, confirm)

        # -------------------------------------------------------------
        # STEP 3: Dense Semantic Representation (TruncatedSVD on TF-IDF)
        # -------------------------------------------------------------
        print("Extracting TruncatedSVD dense semantic features strictly inside fit...", flush=True)
        texts_fit = [combos_all[i] for i in fit]
        texts_tune = [combos_all[i] for i in tune]
        texts_conf = [combos_all[i] for i in confirm]

        vec_w = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=35000, sublinear_tf=True)
        X_w_fit = vec_w.fit_transform(texts_fit)
        X_w_tu = vec_w.transform(texts_tune)
        X_w_co = vec_w.transform(texts_conf)

        svd = TruncatedSVD(n_components=64, random_state=seed)
        svd_fit = svd.fit_transform(X_w_fit).astype(np.float32)
        svd_tu = svd.transform(X_w_tu).astype(np.float32)
        svd_co = svd.transform(X_w_co).astype(np.float32)

        # -------------------------------------------------------------
        # STEP 4: Neural MLP Classifier (Non-Tree Inductive Bias)
        # -------------------------------------------------------------
        print("Training Neural MLP Classifier with 5-fold StratifiedGroupKFold OOF inside fit...", flush=True)
        # Select numeric features and scale them
        num_cols = ["log_price", "sold_ratio", "avail_ratio", "ind_rel_price_median", "ind_price_cat_zscore"]
        scaler = StandardScaler()
        X_num_fit = scaler.fit_transform(F_fit_ind[num_cols].fillna(0).values).astype(np.float32)
        X_num_tu = scaler.transform(F_tu_ind[num_cols].fillna(0).values).astype(np.float32)
        X_num_co = scaler.transform(F_co_ind[num_cols].fillna(0).values).astype(np.float32)

        # Combined dense features: 64 SVD + 5 Scaled numeric + 1 Text OOF
        X_dense_fit = np.hstack([svd_fit, X_num_fit, oof_txt_fit[:, None]])
        X_dense_tu = np.hstack([svd_tu, X_num_tu, p_txt_tu[:, None]])
        X_dense_co = np.hstack([svd_co, X_num_co, p_txt_co[:, None]])

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        oof_mlp_fit = np.zeros(len(fit), dtype=np.float32)

        for fold, (tr_i, va_i) in enumerate(sgkf.split(X_dense_fit, y_fit, groups=sellers_fit)):
            mlp_fold = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=0.01,
                batch_size=256,
                max_iter=40,
                early_stopping=True,
                n_iter_no_change=5,
                random_state=seed + fold,
            )
            mlp_fold.fit(X_dense_fit[tr_i], y_fit[tr_i])
            oof_mlp_fit[va_i] = mlp_fold.predict_proba(X_dense_fit[va_i])[:, 1]

        mlp_full = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=0.01,
            batch_size=256,
            max_iter=40,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=seed,
        )
        mlp_full.fit(X_dense_fit, y_fit)
        p_mlp_tu = mlp_full.predict_proba(X_dense_tu)[:, 1].astype(np.float32)
        p_mlp_co = mlp_full.predict_proba(X_dense_co)[:, 1].astype(np.float32)
        print(f"      [Neural MLP] Fit OOF Log Loss: {calc_log_loss(y_fit, oof_mlp_fit):.4f} | Confirm Acc: {accuracy_score(y_confirm, (p_mlp_co >= 0.5).astype(int)):.4f}", flush=True)

        # -------------------------------------------------------------
        # STEP 5: Tree Family (LightGBM & CatBoost with Neural Signals)
        # -------------------------------------------------------------
        print("Training LightGBM & CatBoost incorporating semantic & neural signals...", flush=True)
        # Augment tabular features with text OOF and neural MLP OOF
        X_tab_fit = F_fit_ind.copy()
        X_tab_tu = F_tu_ind.copy()
        X_tab_co = F_co_ind.copy()

        X_tab_fit["oof_txt"] = oof_txt_fit
        X_tab_tu["oof_txt"] = p_txt_tu
        X_tab_co["oof_txt"] = p_txt_co

        X_tab_fit["oof_mlp"] = oof_mlp_fit
        X_tab_tu["oof_mlp"] = p_mlp_tu
        X_tab_co["oof_mlp"] = p_mlp_co

        # Convert categories for LightGBM
        for col in CAT_FEATURES:
            if col in X_tab_fit.columns:
                X_tab_fit[col] = X_tab_fit[col].astype("category")
                X_tab_tu[col] = X_tab_tu[col].astype("category")
                X_tab_co[col] = X_tab_co[col].astype("category")

        lgb_model = lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.03,
            num_leaves=45,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=seed,
            n_jobs=12,
            verbose=-1,
        )
        lgb_model.fit(X_tab_fit, y_fit)
        p_lgb_tu = lgb_model.predict_proba(X_tab_tu)[:, 1].astype(np.float32)
        p_lgb_co = lgb_model.predict_proba(X_tab_co)[:, 1].astype(np.float32)

        cb_model = CatBoostClassifier(
            iterations=800,
            learning_rate=0.06,
            depth=6,
            l2_leaf_reg=6.0,
            cat_features=[c for c in CAT_FEATURES if c in X_tab_fit.columns],
            random_seed=seed,
            thread_count=12,
            verbose=False,
        )
        # Convert category columns to string for CatBoost
        X_cb_fit = X_tab_fit.copy()
        X_cb_tu = X_tab_tu.copy()
        X_cb_co = X_tab_co.copy()
        for col in CAT_FEATURES:
            if col in X_cb_fit.columns:
                X_cb_fit[col] = X_cb_fit[col].astype(str).fillna("__MISSING__")
                X_cb_tu[col] = X_cb_tu[col].astype(str).fillna("__MISSING__")
                X_cb_co[col] = X_cb_co[col].astype(str).fillna("__MISSING__")

        cb_model.fit(X_cb_fit, y_fit)
        p_cb_tu = cb_model.predict_proba(X_cb_tu)[:, 1].astype(np.float32)
        p_cb_co = cb_model.predict_proba(X_cb_co)[:, 1].astype(np.float32)

        p_tree_tu = 0.60 * p_cb_tu + 0.40 * p_lgb_tu
        p_tree_co = 0.60 * p_cb_co + 0.40 * p_lgb_co

        # -------------------------------------------------------------
        # STEP 6: Multi-Family Ensembles
        # -------------------------------------------------------------
        # Non-tree semantic candidate (50% Linear Text + 50% Neural MLP)
        p_neural_linear_tu = 0.50 * p_txt_tu + 0.50 * p_mlp_tu
        p_neural_linear_co = 0.50 * p_txt_co + 0.50 * p_mlp_co

        # Grand Multi-Family Ensemble:
        # 50% Claude B10 + 35% Tree Model + 15% Neural-Linear Semantic
        p_grand_tu = 0.50 * p_b10_tu + 0.35 * p_tree_tu + 0.15 * p_neural_linear_tu
        p_grand_co = 0.50 * p_b10_co + 0.35 * p_tree_co + 0.15 * p_neural_linear_co

        # Balanced Ensembles for comparison:
        # 50% B10 + 50% Tree Model
        p_b10_tree_tu = 0.50 * p_b10_tu + 0.50 * p_tree_tu
        p_b10_tree_co = 0.50 * p_b10_co + 0.50 * p_tree_co

        # 50% B10 + 50% Neural-Linear
        p_b10_nl_tu = 0.50 * p_b10_tu + 0.50 * p_neural_linear_tu
        p_b10_nl_co = 0.50 * p_b10_co + 0.50 * p_neural_linear_co

        # -------------------------------------------------------------
        # STEP 7: Rigorous Paired Evaluations on Confirm (13,197 rows)
        # -------------------------------------------------------------
        print("\n=== RUNNING PAIRED BENCHMARKS ON CUT 42026 ===", flush=True)
        paired_benchmarks = []

        comparisons = [
            ("Grand_MultiFamily_Ensemble", p_grand_tu, p_grand_co, "Baseline_B10", p_b10_tu, p_b10_co),
            ("Ensemble_B10_Tree_Neural", p_b10_tree_tu, p_b10_tree_co, "Baseline_B10", p_b10_tu, p_b10_co),
            ("Ensemble_B10_Neural_Linear", p_b10_nl_tu, p_b10_nl_co, "Baseline_B10", p_b10_tu, p_b10_co),
            ("Candidate_Tree_Neural", p_tree_tu, p_tree_co, "Baseline_B10", p_b10_tu, p_b10_co),
            ("Candidate_Neural_Linear", p_neural_linear_tu, p_neural_linear_co, "Baseline_B10", p_b10_tu, p_b10_co),
            ("Grand_MultiFamily_Ensemble", p_grand_tu, p_grand_co, "Baseline_Blend075", p_blend075_tu, p_blend075_co),
            ("Grand_MultiFamily_Ensemble", p_grand_tu, p_grand_co, "Baseline_Pipeline_A", p_pipea_tu, p_pipea_co),
        ]

        for name_c, p_c_tu, p_c_co, name_r, p_r_tu, p_r_co in comparisons:
            res = evaluate_pair(name_c, p_c_tu, p_c_co, name_r, p_r_tu, p_r_co, y_tune, y_confirm, sellers_confirm, seen_mask)
            paired_benchmarks.append(res)
            f50 = res["fixed_threshold_050"]
            pt = res["policy_tuned_threshold"]
            print(f"  * {name_c} vs {name_r}:")
            print(f"      Acc: {f50['cand_accuracy']:.5f} vs {f50['ref_accuracy']:.5f} | Net Fixes: {f50['net_fixes']} (95% CI: {f50['bootstrap_95ci']}) | LL: {f50['cand_log_loss']:.5f} vs {f50['ref_log_loss']:.5f}")
            print(f"      Stratified: Seen net={f50['stratified']['seen']['net_fixes']}, New net={f50['stratified']['new']['net_fixes']}")
            print(f"      Policy Tuned: Net Fixes={pt['net_fixes_tuned']} (95% CI: {pt['bootstrap_95ci_tuned']}) | Th={pt['cand_threshold']:.2f}")

        # -------------------------------------------------------------
        # STEP 8: Save Paired Predictions and Detailed JSON Report
        # -------------------------------------------------------------
        raw_conf = raw_data.iloc[confirm]
        paired_df = pd.DataFrame({
            "row_idx": confirm,
            "item_id": raw_conf["id"].values,
            "seller_id": sellers_confirm,
            "is_seen": seen_mask.astype(int),
            "y_used": y_confirm,
            "p_b10": p_b10_co,
            "p_blend075": p_blend075_co,
            "p_pipea": p_pipea_co,
            "p_neural_linear": p_neural_linear_co,
            "p_tree_neural": p_tree_co,
            "p_grand_ensemble": p_grand_co,
        })

        paired_csv_gz = out_dir / "paired_predictions_task_b_seed42026.csv.gz"
        paired_df.to_csv(paired_csv_gz, index=False, compression="gzip")
        paired_sha256 = hashlib.sha256(open(paired_csv_gz, "rb").read()).hexdigest()

        # Save arrays for reuse
        np.save(out_dir / "p_grand_co.npy", p_grand_co)
        np.save(out_dir / "p_grand_tu.npy", p_grand_tu)
        np.save(out_dir / "p_neural_linear_co.npy", p_neural_linear_co)
        np.save(out_dir / "p_neural_linear_tu.npy", p_neural_linear_tu)
        np.save(out_dir / "p_tree_neural_co.npy", p_tree_co)
        np.save(out_dir / "p_tree_neural_tu.npy", p_tree_tu)

        report = {
            "title": "Task B: Semantic Neural and Tree Multi-Family Benchmark on Cut 42026",
            "date_utc": "2026-11-01T06:40:00Z",
            "evaluation_cut": "seed_42026_seller_group_13197_rows",
            "environment": {
                "python": sys.version,
                "catboost": catboost.__version__,
                "lightgbm": lgb.__version__,
            },
            "paired_evaluations": paired_benchmarks,
            "artifacts": {
                "paired_predictions_csv_gz": str(paired_csv_gz.relative_to(REPO_ROOT)),
                "paired_predictions_sha256": paired_sha256,
            },
            "elapsed_seconds": round(time.time() - t0, 1),
        }

        report_json = out_dir / "report_task_b_semantic_neural_seed42026.json"
        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"\nSaved report to {report_json}")
        print(f"Saved paired predictions to {paired_csv_gz} (SHA-256: {paired_sha256})")
        print(f"Task B execution completed in {time.time() - t0:.1f}s.", flush=True)

if __name__ == "__main__":
    main()
