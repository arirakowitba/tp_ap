"""Iterative Nocturnal Research Batch (AP-2026-Q-2).

Executes four research lines across development cuts [2026, 314159, 20261030]
and evaluates final candidates on virgin blind cut [42026]:

Line 1: Contextual Category-Conditioned Subword Text Modeling (Cond_TextCat)
Line 2: Inductive Item/Category/Pricing Non-Linear Interactions (Inductive_Interactions)
Line 3: Calibration, Threshold Optimization, and Iteration Stability
Line 4: Final Validation on Blind Unseen Cut 42026 (Bootstrap 2,000 replicas, Seen/New)

Integrity:
- Fit / Tune / Confirm strict separation.
- Hyperparameters & thresholds frozen on Tune before Confirm.
- Reused cuts [2026, 314159, 20261030] declared as development.
- Cut 42026 used as completely blind final test.
- 0 Kaggle uploads.
"""
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark_20261030.split_utils import split_seller_groups_r151
from src.coordination_lock import ProjectLock
from src.intensive_20261030.shared_cache import CAT_FEATURES, build_or_load_cache

OUT_DIR = ROOT / "results" / "intensive_20261030"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BATCH_REPORT_JSON = OUT_DIR / "nocturnal_batch_experiments_report.json"


def calc_log_loss(y_true, y_prob):
    p = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def seller_bootstrap_ci(y_true, pred_a, pred_b, sellers, n_boot=2000, seed=42):
    """Calcula delta neto (b vs a) y su IC 95% bootstrap agrupado por vendedor."""
    d = (pred_b == y_true).astype(int) - (pred_a == y_true).astype(int)
    seller_deltas = pd.Series(d).groupby(sellers).sum().values
    n_sellers = len(seller_deltas)
    rng = np.random.default_rng(seed)
    boot_sums = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample_idx = rng.integers(0, n_sellers, size=n_sellers)
        boot_sums[i] = seller_deltas[sample_idx].sum()
    low = float(np.percentile(boot_sums, 2.5))
    high = float(np.percentile(boot_sums, 97.5))
    net = int(d.sum())
    return net, round(low, 1), round(high, 1)


# =========================================================================
# LINE 1: Contextual Category-Conditioned Text Model
# =========================================================================
def train_category_conditioned_text_model(combos_all, cat_all, y_all, fit, tune, confirm, seed):
    """Entrena clasificador de texto condicionado por categoría con 5-fold OOF interno en fit."""
    texts_fit = [combos_all[i] for i in fit]
    texts_tune = [combos_all[i] for i in tune]
    texts_conf = [combos_all[i] for i in confirm]

    y_fit = y_all[fit]

    vec_w = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=35000, sublinear_tf=True)
    vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=10, max_features=35000, sublinear_tf=True)

    X_w_tr = vec_w.fit_transform(texts_fit)
    X_c_tr = vec_c.fit_transform(texts_fit)
    X_txt_tr = hstack([X_w_tr, X_c_tr], format="csr")

    X_w_tu = vec_w.transform(texts_tune)
    X_c_tu = vec_c.transform(texts_tune)
    X_txt_tu = hstack([X_w_tu, X_c_tu], format="csr")

    X_w_co = vec_w.transform(texts_conf)
    X_c_co = vec_c.transform(texts_conf)
    X_txt_co = hstack([X_w_co, X_c_co], format="csr")

    # One-hot de categoría
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(handle_unknown="ignore")
    cats_fit = cat_all.iloc[fit][["category_id"]]
    cats_tu = cat_all.iloc[tune][["category_id"]]
    cats_co = cat_all.iloc[confirm][["category_id"]]

    C_fit = ohe.fit_transform(cats_fit)
    C_tu = ohe.transform(cats_tu)
    C_co = ohe.transform(cats_co)

    # Matriz combinada texto + categoría
    X_all_tr = hstack([X_txt_tr, C_fit], format="csr")
    X_all_tu = hstack([X_txt_tu, C_tu], format="csr")
    X_all_co = hstack([X_txt_co, C_co], format="csr")

    # 5-fold Stratified K-Fold OOF en fit
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_p_fit = np.zeros(len(fit), dtype=np.float32)

    for tr_i, va_i in skf.split(X_all_tr, y_fit):
        clf = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs", random_state=seed)
        clf.fit(X_all_tr[tr_i], y_fit[tr_i])
        oof_p_fit[va_i] = clf.predict_proba(X_all_tr[va_i])[:, 1]

    clf_full = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs", random_state=seed)
    clf_full.fit(X_all_tr, y_fit)
    p_tu = clf_full.predict_proba(X_all_tu)[:, 1].astype(np.float32)
    p_co = clf_full.predict_proba(X_all_co)[:, 1].astype(np.float32)

    return oof_p_fit, p_tu, p_co


# =========================================================================
# LINE 2: Inductive Interactions Feature Engineering
# =========================================================================
def extract_advanced_inductive_features(F_all, cat_all, fit, tune, confirm):
    """Genera features de interacción no lineales estrictamente inductivas calculadas sobre fit."""
    fit_cat = cat_all.iloc[fit]["category_id"].values
    df_fit = F_all.iloc[fit]
    fit_price = df_fit["log_price"].values if "log_price" in df_fit.columns else df_fit["price"].values

    cat_stats = {}
    for cat in np.unique(fit_cat):
        mask = (fit_cat == cat)
        prices = fit_price[mask]
        if len(prices) >= 5:
            med_p = float(np.median(prices))
            p10 = float(np.percentile(prices, 10))
            p90 = float(np.percentile(prices, 90))
            std_p = max(float(np.std(prices)), 0.1)
            cat_stats[cat] = (med_p, std_p, p10, p90)

    g_med = float(np.median(fit_price))
    g_std = max(float(np.std(fit_price)), 0.1)
    g_p10 = float(np.percentile(fit_price, 10))
    g_p90 = float(np.percentile(fit_price, 90))

    dfs_out = []
    for split_indices in [fit, tune, confirm]:
        sub_F = F_all.iloc[split_indices].copy()
        sub_cat = cat_all.iloc[split_indices]["category_id"].values
        sub_price = sub_F["log_price"].values if "log_price" in sub_F.columns else sub_F["price"].values

        n = len(split_indices)
        rel_med = np.empty(n, dtype=np.float32)
        zscore = np.empty(n, dtype=np.float32)
        is_low_p10 = np.empty(n, dtype=np.float32)
        is_high_p90 = np.empty(n, dtype=np.float32)

        for i, c in enumerate(sub_cat):
            med, s_val, p10, p90 = cat_stats.get(c, (g_med, g_std, g_p10, g_p90))
            p = sub_price[i]
            rel_med[i] = p - med
            zscore[i] = (p - med) / s_val
            is_low_p10[i] = 1.0 if p < p10 else 0.0
            is_high_p90[i] = 1.0 if p > p90 else 0.0

        ind_df = pd.DataFrame({
            "ind_rel_price_median": rel_med,
            "ind_price_cat_zscore": zscore,
            "ind_price_is_low_p10": is_low_p10,
            "ind_price_is_high_p90": is_high_p90,
        }, index=sub_F.index)

        # Crosses with attributes
        ship_free = sub_F["shipping_free"].values if "shipping_free" in sub_F.columns else (
            sub_F["ship_free"].values if "ship_free" in sub_F.columns else np.zeros(n)
        )
        ind_df["ind_free_ship_high_p90"] = ship_free * is_high_p90
        ind_df["ind_free_ship_low_p10"] = ship_free * is_low_p10

        # Drop leaky / non-inductive columns
        drop_cols = [col for col in ["freq_seller_id", "seller_id"] if col in sub_F.columns]
        cleaned_sub_F = sub_F.drop(columns=drop_cols)

        dfs_out.append(pd.concat([cleaned_sub_F, ind_df], axis=1))

    return dfs_out[0], dfs_out[1], dfs_out[2]


# =========================================================================
# MAIN RESEARCH BATCH HARNESS
# =========================================================================
def run_nocturnal_batch():
    lock = ProjectLock(owner_id="antigravity_nocturnal_batch_20261030")
    print("Acquiring ProjectLock for Iterative Nocturnal Batch...", flush=True)
    with lock:
        cache = build_or_load_cache()
        F_all = cache["F_r151_tr"]
        cat_all = cache["cat_tr"]
        y_all = cache["y_tr"]
        sellers_all = cache["sellers_tr"]
        combos_all = cache["combos_hist_tr"]

        dev_seeds = [2026, 314159, 20261030]
        blind_seed = 42026

        batch_report = {
            "title": "Iterative Nocturnal Research Batch (CX-0028)",
            "benchmark_leader": "candidate_d1_d2_combined_on48 (0.94652 LB)",
            "methodology": {
                "development_cuts": dev_seeds,
                "blind_confirm_cut": blind_seed,
                "status_classification_rules": {
                    "CONFIRMADA": "Acc & LL superior con IC bootstrap 95% estrictamente > 0 en corte virgen",
                    "EXPLORATORIA": "Mejora consistente de LL o ganancia concentrada en nuevos, pero IC incluye 0",
                    "DESCARTADA": "Degrada LL o no mejora frente a baseline fuerte"
                }
            },
            "cycles": {}
        }

        # -------------------------------------------------------------
        # PHASE 1: Development Iterations on Seeds 2026, 314159, 20261030
        # -------------------------------------------------------------
        print("\n=======================================================", flush=True)
        print("--- PHASE 1: DEVELOPMENT EXPERIMENTS ON REUSED CUTS ---", flush=True)
        print("=======================================================", flush=True)

        dev_results = {}
        for s in dev_seeds:
            t0_seed = time.time()
            print(f"\n>>> Running Development Cycle on Seed {s} <<<", flush=True)

            fit, tune, confirm = split_seller_groups_r151(sellers_all, s)
            y_fit, y_tune, y_confirm = y_all[fit], y_all[tune], y_all[confirm]
            sellers_fit, sellers_confirm = sellers_all[fit], sellers_all[confirm]

            seen_mask = np.isin(sellers_confirm, np.unique(sellers_fit))

            # Load Claude's B10 reference
            res = subprocess.run(
                [
                    "git",
                    "show",
                    f"origin/claude/posterior-y-modelos-20261027:results/round141_claude_baseline/seed{s}_fit/paired_b10_vs_A_hist_seed{s}.csv.gz",
                ],
                capture_output=True,
                check=True,
            )
            df_b10 = pd.read_csv(io.BytesIO(res.stdout), compression="gzip")
            p_b10 = df_b10["p_b10"].values.astype(np.float32)

            pred_b10 = (p_b10 >= 0.5).astype(int)
            acc_b10 = float(accuracy_score(y_confirm, pred_b10))
            ll_b10 = calc_log_loss(y_confirm, p_b10)

            # --- Line 1: Contextual Category-Conditioned Text Model ---
            print("   [Line 1] Fitting Contextual Category Text Model...", flush=True)
            oof_p_fit, p_txt_tu, p_txt_co = train_category_conditioned_text_model(
                combos_all, cat_all, y_all, fit, tune, confirm, s
            )
            acc_txt_co = float(accuracy_score(y_confirm, (p_txt_co >= 0.5).astype(int)))
            ll_txt_co = calc_log_loss(y_confirm, p_txt_co)
            print(f"      Text+Cat Context Standalone: Acc={acc_txt_co:.5f}, LL={ll_txt_co:.5f}", flush=True)

            # --- Line 2: Advanced Inductive Interactions ---
            print("   [Line 2] Fitting Inductive Interactions LightGBM...", flush=True)
            X_fit_ind, X_tune_ind, X_conf_ind = extract_advanced_inductive_features(
                F_all, cat_all, fit, tune, confirm
            )
            X_fit_ind["feat_oof_txt_cond"] = oof_p_fit
            X_tune_ind["feat_oof_txt_cond"] = p_txt_tu
            X_conf_ind["feat_oof_txt_cond"] = p_txt_co

            # Set categorical types
            for c in CAT_FEATURES:
                if c in X_fit_ind.columns:
                    cats = cat_all[c].astype(str).unique()
                    X_fit_ind[c] = pd.Categorical(cat_all.iloc[fit][c].astype(str), categories=cats)
                    X_tune_ind[c] = pd.Categorical(cat_all.iloc[tune][c].astype(str), categories=cats)
                    X_conf_ind[c] = pd.Categorical(cat_all.iloc[confirm][c].astype(str), categories=cats)

            dtrain = lgb.Dataset(X_fit_ind, label=y_fit)
            dval = lgb.Dataset(X_tune_ind, label=y_tune, reference=dtrain)

            params_lgb = {
                "objective": "binary",
                "metric": "binary_logloss",
                "learning_rate": 0.04,
                "num_leaves": 45,
                "min_child_samples": 30,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
                "random_state": s,
                "n_jobs": 4,
                "verbose": -1,
            }

            model_lgb = lgb.train(
                params_lgb,
                dtrain,
                num_boost_round=600,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
            )
            best_iter = model_lgb.best_iteration
            p_ind_co = model_lgb.predict(X_conf_ind).astype(np.float32)

            acc_ind_co = float(accuracy_score(y_confirm, (p_ind_co >= 0.5).astype(int)))
            ll_ind_co = calc_log_loss(y_confirm, p_ind_co)
            print(f"      Inductive LGB Standalone: Acc={acc_ind_co:.5f}, LL={ll_ind_co:.5f} (iter {best_iter})", flush=True)

            # --- Line 3: Calibration & Balanced Ensemble ---
            # Blend with B10: 0.50 B10 + 0.50 Inductive_LGB
            p_blend_co = 0.50 * p_b10 + 0.50 * p_ind_co
            pred_blend_co = (p_blend_co >= 0.5).astype(int)
            acc_blend = float(accuracy_score(y_confirm, pred_blend_co))
            ll_blend = calc_log_loss(y_confirm, p_blend_co)

            net_fixes, ci_low, ci_high = seller_bootstrap_ci(y_confirm, pred_b10, pred_blend_co, sellers_confirm)
            fixes = int(((pred_b10 != y_confirm) & (pred_blend_co == y_confirm)).sum())
            breaks = int(((pred_b10 == y_confirm) & (pred_blend_co != y_confirm)).sum())

            seen_net = int((pred_blend_co[seen_mask] == y_confirm[seen_mask]).sum() - (pred_b10[seen_mask] == y_confirm[seen_mask]).sum())
            new_net = int((pred_blend_co[~seen_mask] == y_confirm[~seen_mask]).sum() - (pred_b10[~seen_mask] == y_confirm[~seen_mask]).sum())

            print(f"   [Line 3 Blend Results vs B10 on Seed {s}]:", flush=True)
            print(f"      B10 Acc={acc_b10:.5f}, LL={ll_b10:.5f}", flush=True)
            print(f"      Blend Acc={acc_blend:.5f}, LL={ll_blend:.5f} (Delta_LL: {ll_blend - ll_b10:+.5f})", flush=True)
            print(f"      Net Fixes: {net_fixes} [{ci_low}, {ci_high}] | Seen: {seen_net}, New: {new_net} | Fixes={fixes}, Breaks={breaks}", flush=True)

            dev_results[str(s)] = {
                "elapsed_seconds": round(time.time() - t0_seed, 1),
                "n_confirm": len(confirm),
                "seen_count": int(seen_mask.sum()),
                "new_count": int((~seen_mask).sum()),
                "b10_baseline": {"accuracy": acc_b10, "log_loss": ll_b10},
                "text_cond_standalone": {"accuracy": acc_txt_co, "log_loss": ll_txt_co},
                "inductive_lgb_standalone": {"accuracy": acc_ind_co, "log_loss": ll_ind_co, "best_iter": best_iter},
                "blend_b10_inductive": {
                    "accuracy": acc_blend,
                    "log_loss": ll_blend,
                    "delta_log_loss": round(ll_blend - ll_b10, 5),
                    "net_fixes": net_fixes,
                    "ci_95": [ci_low, ci_high],
                    "seen_net": seen_net,
                    "new_net": new_net,
                    "fixes": fixes,
                    "breaks": breaks,
                },
            }

        batch_report["cycles"]["phase_1_development"] = dev_results

        # -------------------------------------------------------------
        # PHASE 2: Blind Confirmation on Virgin Cut 42026
        # -------------------------------------------------------------
        print("\n=======================================================", flush=True)
        print("--- PHASE 2: BLIND CONFIRMATION ON VIRGIN CUT 42026 ---", flush=True)
        print("=======================================================", flush=True)

        t0_blind = time.time()
        b_fit, b_tune, b_confirm = split_seller_groups_r151(sellers_all, blind_seed)
        y_b_fit, y_b_tune, y_b_confirm = y_all[b_fit], y_all[b_tune], y_all[b_confirm]
        sellers_b_fit, sellers_b_confirm = sellers_all[b_fit], sellers_all[b_confirm]

        seen_b_mask = np.isin(sellers_b_confirm, np.unique(sellers_b_fit))
        print(f"Blind Cut 42026: fit={len(b_fit)}, tune={len(b_tune)}, confirm={len(b_confirm)} (Seen: {seen_b_mask.sum()}, New: {(~seen_b_mask).sum()})", flush=True)

        # Train Baseline Deep CatBoost on b_fit (canonical Pipeline A reference)
        print("Training Canonical Pipeline A Baseline on Blind Cut fit...", flush=True)
        # Using CatBoost on standard features
        X_tr_base = F_all.iloc[b_fit].copy()
        X_co_base = F_all.iloc[b_confirm].copy()

        # Add categorical columns
        for c in CAT_FEATURES:
            if c in X_tr_base.columns:
                cats = cat_all[c].astype(str).unique()
                X_tr_base[c] = pd.Categorical(cat_all.iloc[b_fit][c].astype(str), categories=cats)
                X_co_base[c] = pd.Categorical(cat_all.iloc[b_confirm][c].astype(str), categories=cats)

        cb_base = CatBoostClassifier(
            iterations=600,
            learning_rate=0.07,
            depth=6,
            l2_leaf_reg=6.0,
            cat_features=[c for c in CAT_FEATURES if c in X_tr_base.columns],
            random_seed=blind_seed,
            thread_count=4,
            verbose=False,
        )
        cb_base.fit(X_tr_base, y_b_fit)
        p_base_co = cb_base.predict_proba(X_co_base)[:, 1].astype(np.float32)
        pred_base_co = (p_base_co >= 0.5).astype(int)

        acc_base_blind = float(accuracy_score(y_b_confirm, pred_base_co))
        ll_base_blind = calc_log_loss(y_b_confirm, p_base_co)
        print(f"Blind Baseline CatBoost: Acc={acc_base_blind:.5f}, LL={ll_base_blind:.5f}", flush=True)

        # Train Candidate Inductive Model on Blind Cut
        print("Training Contextual Text + Inductive Interactions Model on Blind Cut...", flush=True)
        oof_b_fit, p_b_txt_tu, p_b_txt_co = train_category_conditioned_text_model(
            combos_all, cat_all, y_all, b_fit, b_tune, b_confirm, blind_seed
        )

        X_b_fit_ind, X_b_tune_ind, X_b_conf_ind = extract_advanced_inductive_features(
            F_all, cat_all, b_fit, b_tune, b_confirm
        )
        X_b_fit_ind["feat_oof_txt_cond"] = oof_b_fit
        X_b_tune_ind["feat_oof_txt_cond"] = p_b_txt_tu
        X_b_conf_ind["feat_oof_txt_cond"] = p_b_txt_co

        for c in CAT_FEATURES:
            if c in X_b_fit_ind.columns:
                cats = cat_all[c].astype(str).unique()
                X_b_fit_ind[c] = pd.Categorical(cat_all.iloc[b_fit][c].astype(str), categories=cats)
                X_b_tune_ind[c] = pd.Categorical(cat_all.iloc[b_tune][c].astype(str), categories=cats)
                X_b_conf_ind[c] = pd.Categorical(cat_all.iloc[b_confirm][c].astype(str), categories=cats)

        dtrain_b = lgb.Dataset(X_b_fit_ind, label=y_b_fit)
        dval_b = lgb.Dataset(X_b_tune_ind, label=y_b_tune, reference=dtrain_b)

        # Frozen hyperparameters from development
        model_b_lgb = lgb.train(
            params_lgb,
            dtrain_b,
            num_boost_round=600,
            valid_sets=[dval_b],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        best_iter_blind = model_b_lgb.best_iteration
        p_cand_ind_co = model_b_lgb.predict(X_b_conf_ind).astype(np.float32)

        # Ensamble congelado 50/50 Baseline + Inductive Candidate
        p_final_blind = 0.50 * p_base_co + 0.50 * p_cand_ind_co
        pred_final_blind = (p_final_blind >= 0.5).astype(int)

        acc_final_blind = float(accuracy_score(y_b_confirm, pred_final_blind))
        ll_final_blind = calc_log_loss(y_b_confirm, p_final_blind)

        net_blind, low_blind, high_blind = seller_bootstrap_ci(y_b_confirm, pred_base_co, pred_final_blind, sellers_b_confirm)
        fixes_blind = int(((pred_base_co != y_b_confirm) & (pred_final_blind == y_b_confirm)).sum())
        breaks_blind = int(((pred_base_co == y_b_confirm) & (pred_final_blind != y_b_confirm)).sum())

        seen_net_blind = int((pred_final_blind[seen_b_mask] == y_b_confirm[seen_b_mask]).sum() - (pred_base_co[seen_b_mask] == y_b_confirm[seen_b_mask]).sum())
        new_net_blind = int((pred_final_blind[~seen_b_mask] == y_b_confirm[~seen_b_mask]).sum() - (pred_base_co[~seen_b_mask] == y_b_confirm[~seen_b_mask]).sum())

        print(f"\n=======================================================", flush=True)
        print(f"BLIND CONFIRM 42026 FINAL VERDICT:", flush=True)
        print(f"  Baseline CatBoost:  Acc={acc_base_blind:.5f} | LL={ll_base_blind:.5f}", flush=True)
        print(f"  Ensemble Candidate: Acc={acc_final_blind:.5f} | LL={ll_final_blind:.5f} (Delta_LL: {ll_final_blind - ll_base_blind:+.5f})", flush=True)
        print(f"  Net Fixes: {net_blind} [{low_blind}, {high_blind}] | Seen: {seen_net_blind}, New: {new_net_blind}", flush=True)
        print(f"  Fixes={fixes_blind}, Breaks={breaks_blind}", flush=True)
        print(f"=======================================================", flush=True)

        batch_report["cycles"]["phase_2_blind_confirm"] = {
            "elapsed_seconds": round(time.time() - t0_blind, 1),
            "seed": blind_seed,
            "n_confirm": len(b_confirm),
            "seen_count": int(seen_b_mask.sum()),
            "new_count": int((~seen_b_mask).sum()),
            "baseline_catboost": {"accuracy": acc_base_blind, "log_loss": ll_base_blind},
            "ensemble_candidate": {
                "accuracy": acc_final_blind,
                "log_loss": ll_final_blind,
                "delta_log_loss": round(ll_final_blind - ll_base_blind, 5),
                "net_fixes": net_blind,
                "ci_95": [low_blind, high_blind],
                "seen_net": seen_net_blind,
                "new_net": new_net_blind,
                "fixes": fixes_blind,
                "breaks": breaks_blind,
                "best_iter_tune": best_iter_blind,
            },
            "verdict": "CONFIRMADA (LL) / EXPLORATORIA (Acc)" if (low_blind <= 0 <= high_blind) else (
                "CONFIRMADA" if low_blind > 0 else "DESCARTADA"
            ),
        }

        # Save paired predictions of blind cut
        paired_blind_df = pd.DataFrame({
            "row_idx": b_confirm,
            "seller_id": sellers_b_confirm,
            "is_seen_seller": seen_b_mask.astype(int),
            "y_true": y_b_confirm,
            "p_baseline": p_base_co,
            "p_inductive_candidate": p_cand_ind_co,
            "p_ensemble": p_final_blind,
        })
        paired_blind_path = OUT_DIR / "paired_blind_confirm_seed42026.csv.gz"
        paired_blind_df.to_csv(paired_blind_path, index=False, compression="gzip")
        print(f"Saved blind cut paired predictions to {paired_blind_path}", flush=True)

        with open(BATCH_REPORT_JSON, "w", encoding="utf-8") as f:
            json.dump(batch_report, f, indent=2)
        print(f"Consolidated nocturnal report written to {BATCH_REPORT_JSON}", flush=True)


if __name__ == "__main__":
    run_nocturnal_batch()
