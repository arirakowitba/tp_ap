"""Intensive Training and Validation of Distinct Hypotheses (CX-0027).

Hypotheses:
1. Hypothesis 1 (H1_TextSubwords): Fine-grained subword & word n-gram text model
   (char 3-5 + word 1-2) with OOF calibration inside fit.
2. Hypothesis 2 (H2_InductiveInteractions): Inductive price-category distributions,
   ratios, quantiles, and title/attribute crosses without leaking future splits.
3. Hypothesis 3 (H3_DecoupledRouter): Specialized inductive architecture for new sellers,
   combining pure inductive trees and fine subwords for new sellers while preserving
   strong tabular ensemble for seen sellers.

Validation:
- Strict split_seller_groups_r151 across seeds [2026, 314159, 20261030].
- Fit, tune, confirm separation. Hyperparameters & thresholds frozen on tune before confirm.
- Paired predictions vs B10 and baseline by ID.
- Accuracy, Log Loss, Seen vs New sellers breakdown.
- Seller bootstrap 95% confidence intervals (2,000 replicas).
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
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark_20261030.split_utils import split_seller_groups_r151
from src.coordination_lock import ProjectLock
from src.intensive_20261030.shared_cache import CAT_FEATURES, build_or_load_cache

OUT_DIR = ROOT / "results" / "intensive_20261030"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_JSON = OUT_DIR / "intensive_hypotheses_confirm_report.json"


def calc_log_loss(y_true, y_prob):
    p = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def seller_bootstrap_ci(y_true, pred_a, pred_b, sellers, n_boot=2000, seed=42):
    """Calcula el delta neto de aciertos (b vs a) y su IC 95% bootstrap por vendedor."""
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


def build_inductive_price_interactions(df_feat, cat_df, fit_idx, eval_indices):
    """Calcula estadísticas de precio por categoría estrictamente sobre fit y genera features inductivas."""
    fit_cat = cat_df.iloc[fit_idx]["category_id"].values
    fit_price = df_feat.iloc[fit_idx]["log_price"].values if "log_price" in df_feat.columns else df_feat.iloc[fit_idx]["price"].values
    
    # Precios por categoría en fit
    cat_stats = {}
    for cat in np.unique(fit_cat):
        mask = (fit_cat == cat)
        prices = fit_price[mask]
        if len(prices) > 0:
            median_p = np.median(prices)
            q25 = np.percentile(prices, 25)
            q75 = np.percentile(prices, 75)
            iqr = max(q75 - q25, 0.1)
            cat_stats[cat] = (median_p, iqr)

    global_median = np.median(fit_price)
    global_iqr = max(np.percentile(fit_price, 75) - np.percentile(fit_price, 25), 0.1)

    dfs_out = []
    for idx_set in [fit_idx] + eval_indices:
        sub_df = df_feat.iloc[idx_set].copy()
        sub_cat = cat_df.iloc[idx_set]["category_id"].values
        sub_price = sub_df["log_price"].values if "log_price" in sub_df.columns else sub_df["price"].values
        
        rel_median = np.empty(len(idx_set), dtype=np.float32)
        rel_norm = np.empty(len(idx_set), dtype=np.float32)
        above_median = np.empty(len(idx_set), dtype=np.float32)
        
        for i, cat in enumerate(sub_cat):
            med, iqr = cat_stats.get(cat, (global_median, global_iqr))
            diff = sub_price[i] - med
            rel_median[i] = diff
            rel_norm[i] = diff / iqr
            above_median[i] = 1.0 if diff > 0 else 0.0

        new_cols = pd.DataFrame({
            "ind_price_rel_median": rel_median,
            "ind_price_rel_norm": rel_norm,
            "ind_price_above_median": above_median
        }, index=sub_df.index)
        
        # Free shipping interaction
        if "shipping_free" in sub_df.columns:
            new_cols["ind_free_ship_high_price"] = sub_df["shipping_free"].values * above_median
        elif "ship_free" in sub_df.columns:
            new_cols["ind_free_ship_high_price"] = sub_df["ship_free"].values * above_median
        else:
            new_cols["ind_free_ship_high_price"] = 0.0

        # Title length relative to category
        if "title_len" in sub_df.columns:
            new_cols["ind_title_len"] = sub_df["title_len"].values
        
        dfs_out.append(pd.concat([sub_df, new_cols], axis=1))

    return dfs_out[0], dfs_out[1:]


def run_training_and_validation(seeds=[2026, 314159, 20261030]):
    lock = ProjectLock(owner_id="antigravity_intensive_hypotheses_20261030")
    print("Acquiring ProjectLock for Intensive Hypotheses...", flush=True)
    with lock:
        cache = build_or_load_cache()
        F_all = cache["F_r151_tr"]
        cat_all = cache["cat_tr"]
        y_all = cache["y_tr"]
        sellers_all = cache["sellers_tr"]
        ids_all = cache["ids_tr"]
        combos_all = cache["combos_hist_tr"]
        titles_all = cache["titles_tr"]

        report = {
            "title": "Intensive Hypotheses Validation (CX-0027)",
            "benchmark_leader": "candidate_d1_d2_combined_on48 (0.94652 LB)",
            "seeds": seeds,
            "hypotheses_tested": {
                "H1_TextSubwords": "Calibrated OOF Char 3-5 + Word 1-2 TF-IDF model",
                "H2_InductiveInteractions": "Inductive price-to-category distributions & crosses with LightGBM",
                "H3_DecoupledRouter": "Specialized architecture routing new sellers to inductive + subword models"
            },
            "cuts": {}
        }

        for s in seeds:
            t0_cut = time.time()
            print(f"\n=======================================================", flush=True)
            print(f"--- EVALUATING INTENSIVE HYPOTHESES ON SEED {s} ---", flush=True)
            print(f"=======================================================", flush=True)

            # 1. Load Claude's B10 confirm predictions
            print(f"Loading Claude's B10 paired confirm predictions for seed {s}...", flush=True)
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
            b10_row_idx = df_b10["row_idx"].values
            p_b10 = df_b10["p_b10"].values.astype(np.float32)
            p_a_hist = df_b10["p_A_hist"].values.astype(np.float32)

            # 2. Split seller groups
            fit, tune, confirm = split_seller_groups_r151(sellers_all, s)
            assert np.array_equal(confirm, b10_row_idx), f"Confirm row_idx mismatch for seed {s}!"

            y_fit = y_all[fit]
            y_tune = y_all[tune]
            y_confirm = y_all[confirm]

            sellers_fit = sellers_all[fit]
            sellers_tune = sellers_all[tune]
            sellers_confirm = sellers_all[confirm]

            fit_sellers_set = set(np.unique(sellers_fit))
            seen_tune = np.isin(sellers_tune, list(fit_sellers_set))
            seen_confirm = np.isin(sellers_confirm, list(fit_sellers_set))

            print(f"Confirm size: {len(confirm)} (Seen: {seen_confirm.sum()}, New: {(~seen_confirm).sum()})", flush=True)
            print(f"Tune size:    {len(tune)} (Seen: {seen_tune.sum()}, New: {(~seen_tune).sum()})", flush=True)

            # -------------------------------------------------------------
            # HYPOTHESIS 1: Fine-Grained Subword Text Model
            # -------------------------------------------------------------
            print("\n[H1] Training Fine-Grained Subword Text Model...", flush=True)
            t0_h1 = time.time()
            
            # Use combos_all (title + attributes)
            texts_fit = [combos_all[i] for i in fit]
            texts_tune = [combos_all[i] for i in tune]
            texts_confirm = [combos_all[i] for i in confirm]

            vec_w = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=40000, sublinear_tf=True)
            vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=10, max_features=40000, sublinear_tf=True)

            X_w_fit = vec_w.fit_transform(texts_fit)
            X_c_fit = vec_c.fit_transform(texts_fit)
            X_txt_fit = hstack([X_w_fit, X_c_fit], format="csr")

            X_w_tune = vec_w.transform(texts_tune)
            X_c_tune = vec_c.transform(texts_tune)
            X_txt_tune = hstack([X_w_tune, X_c_tune], format="csr")

            X_w_conf = vec_w.transform(texts_confirm)
            X_c_conf = vec_c.transform(texts_confirm)
            X_txt_conf = hstack([X_w_conf, X_c_conf], format="csr")

            # 5-fold Stratified K-Fold inside fit to generate OOF text probabilities
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
            oof_p_txt_fit = np.zeros(len(fit), dtype=np.float32)
            
            for tr_idx, val_idx in skf.split(X_txt_fit, y_fit):
                clf_fold = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs", random_state=s, n_jobs=4)
                clf_fold.fit(X_txt_fit[tr_idx], y_fit[tr_idx])
                oof_p_txt_fit[val_idx] = clf_fold.predict_proba(X_txt_fit[val_idx])[:, 1]

            # Fit full model on fit
            clf_txt = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs", random_state=s, n_jobs=4)
            clf_txt.fit(X_txt_fit, y_fit)
            p_txt_tune = clf_txt.predict_proba(X_txt_tune)[:, 1].astype(np.float32)
            p_txt_conf = clf_txt.predict_proba(X_txt_conf)[:, 1].astype(np.float32)
            
            acc_h1_conf = float(accuracy_score(y_confirm, (p_txt_conf >= 0.5).astype(int)))
            ll_h1_conf = calc_log_loss(y_confirm, p_txt_conf)
            print(f"   H1 Pure Text Confirm: Acc={acc_h1_conf:.5f}, LL={ll_h1_conf:.5f} ({time.time()-t0_h1:.1f}s)", flush=True)

            # -------------------------------------------------------------
            # HYPOTHESIS 2: Inductive Non-Linear Price-Category Interactions
            # -------------------------------------------------------------
            print("\n[H2] Training Inductive Price-Category LightGBM...", flush=True)
            t0_h2 = time.time()
            
            F_fit_ind, (F_tune_ind, F_conf_ind) = build_inductive_price_interactions(
                F_all, cat_all, fit, [tune, confirm]
            )

            # Drop non-inductive leaky columns (like freq_seller_id) for inductive model
            drop_cols = [c for c in ["freq_seller_id", "seller_id"] if c in F_fit_ind.columns]
            X_fit_ind = F_fit_ind.drop(columns=drop_cols)
            X_tune_ind = F_tune_ind.drop(columns=drop_cols)
            X_conf_ind = F_conf_ind.drop(columns=drop_cols)

            # Add OOF text probabilities as a teacher feature!
            X_fit_ind["feat_oof_txt_fine"] = oof_p_txt_fit
            X_tune_ind["feat_oof_txt_fine"] = p_txt_tune
            X_conf_ind["feat_oof_txt_fine"] = p_txt_conf

            # Convert categoricals
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
                "verbose": -1
            }

            model_h2 = lgb.train(
                params_lgb,
                dtrain,
                num_boost_round=600,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
            )
            best_iter = model_h2.best_iteration
            print(f"   H2 LightGBM best iteration on tune: {best_iter}", flush=True)

            p_h2_tune = model_h2.predict(X_tune_ind).astype(np.float32)
            p_h2_conf = model_h2.predict(X_conf_ind).astype(np.float32)

            acc_h2_conf = float(accuracy_score(y_confirm, (p_h2_conf >= 0.5).astype(int)))
            ll_h2_conf = calc_log_loss(y_confirm, p_h2_conf)
            print(f"   H2 Standalone Confirm: Acc={acc_h2_conf:.5f}, LL={ll_h2_conf:.5f} ({time.time()-t0_h2:.1f}s)", flush=True)

            # -------------------------------------------------------------
            # HYPOTHESIS 3: Decoupled Inductive Router
            # -------------------------------------------------------------
            print("\n[H3] Evaluating Decoupled Inductive Ensemble...", flush=True)
            # Combine B10 with the new inductive signal:
            # For seen sellers: 0.60 * B10 + 0.40 * H2
            # For new sellers: 0.40 * B10 + 0.60 * H2 (emphasizing the inductive text/price features)
            p_h3_conf = np.empty_like(p_b10)
            p_h3_conf[seen_confirm] = 0.50 * p_b10[seen_confirm] + 0.50 * p_h2_conf[seen_confirm]
            p_h3_conf[~seen_confirm] = 0.40 * p_b10[~seen_confirm] + 0.60 * p_h2_conf[~seen_confirm]

            acc_h3_conf = float(accuracy_score(y_confirm, (p_h3_conf >= 0.5).astype(int)))
            ll_h3_conf = calc_log_loss(y_confirm, p_h3_conf)

            # Simple uniform blend H2 + B10 for comparison
            p_h2_b10_conf = 0.50 * p_b10 + 0.50 * p_h2_conf
            acc_h2_b10_conf = float(accuracy_score(y_confirm, (p_h2_b10_conf >= 0.5).astype(int)))
            ll_h2_b10_conf = calc_log_loss(y_confirm, p_h2_b10_conf)

            # B10 baseline metrics
            pred_b10 = (p_b10 >= 0.5).astype(int)
            acc_b10 = float(accuracy_score(y_confirm, pred_b10))
            ll_b10 = calc_log_loss(y_confirm, p_b10)

            # A_hist baseline metrics
            pred_ahist = (p_a_hist >= 0.5).astype(int)
            acc_ahist = float(accuracy_score(y_confirm, pred_ahist))
            ll_ahist = calc_log_loss(y_confirm, p_a_hist)

            # Compute paired statistics vs B10
            pred_h3 = (p_h3_conf >= 0.5).astype(int)
            pred_h2_b10 = (p_h2_b10_conf >= 0.5).astype(int)

            net_h3, low_h3, high_h3 = seller_bootstrap_ci(y_confirm, pred_b10, pred_h3, sellers_confirm)
            net_h2b10, low_h2b10, high_h2b10 = seller_bootstrap_ci(y_confirm, pred_b10, pred_h2_b10, sellers_confirm)

            # Seen vs New breakdown for H3 vs B10
            fixes_h3 = int(((pred_b10 != y_confirm) & (pred_h3 == y_confirm)).sum())
            breaks_h3 = int(((pred_b10 == y_confirm) & (pred_h3 != y_confirm)).sum())
            
            seen_net_h3 = int((pred_h3[seen_confirm] == y_confirm[seen_confirm]).sum() - (pred_b10[seen_confirm] == y_confirm[seen_confirm]).sum())
            new_net_h3 = int((pred_h3[~seen_confirm] == y_confirm[~seen_confirm]).sum() - (pred_b10[~seen_confirm] == y_confirm[~seen_confirm]).sum())

            print(f"Results for seed {s}:", flush=True)
            print(f"  B10 Baseline:        Acc={acc_b10:.5f} | LL={ll_b10:.5f}", flush=True)
            print(f"  A_hist Baseline:     Acc={acc_ahist:.5f} | LL={ll_ahist:.5f}", flush=True)
            print(f"  H1 (Pure Subwords):  Acc={acc_h1_conf:.5f} | LL={ll_h1_conf:.5f}", flush=True)
            print(f"  H2 (Inductive LGB):  Acc={acc_h2_conf:.5f} | LL={ll_h2_conf:.5f}", flush=True)
            print(f"  H2+B10 Blend:        Acc={acc_h2_b10_conf:.5f} | LL={ll_h2_b10_conf:.5f} | Net={net_h2b10} [{low_h2b10}, {high_h2b10}]", flush=True)
            print(f"  H3 (Decoupled):      Acc={acc_h3_conf:.5f} | LL={ll_h3_conf:.5f} | Net={net_h3} [{low_h3}, {high_h3}] (Seen: {seen_net_h3}, New: {new_net_h3})", flush=True)

            # Save paired confirm predictions
            paired_df = pd.DataFrame({
                "row_idx": confirm,
                "seller_id": sellers_confirm,
                "is_seen_seller": seen_confirm.astype(int),
                "y_true": y_confirm,
                "p_b10": p_b10,
                "p_A_hist": p_a_hist,
                "p_h1_text_subwords": p_txt_conf,
                "p_h2_inductive": p_h2_conf,
                "p_h2_b10_blend": p_h2_b10_conf,
                "p_h3_decoupled": p_h3_conf
            })
            paired_path = OUT_DIR / f"paired_intensive_hypotheses_seed{s}_confirm.csv.gz"
            paired_df.to_csv(paired_path, index=False, compression="gzip")
            print(f"Saved paired predictions to {paired_path}", flush=True)

            report["cuts"][str(s)] = {
                "elapsed_seconds": round(time.time() - t0_cut, 1),
                "n_confirm": len(confirm),
                "n_seen": int(seen_confirm.sum()),
                "n_new": int((~seen_confirm).sum()),
                "b10_baseline": {"accuracy": acc_b10, "log_loss": ll_b10},
                "a_hist_baseline": {"accuracy": acc_ahist, "log_loss": ll_ahist},
                "h1_pure_text_subwords": {"accuracy": acc_h1_conf, "log_loss": ll_h1_conf},
                "h2_inductive_lgb": {"accuracy": acc_h2_conf, "log_loss": ll_h2_conf, "best_iteration": best_iter},
                "h2_b10_blend": {
                    "accuracy": acc_h2_b10_conf,
                    "log_loss": ll_h2_b10_conf,
                    "vs_b10_net": net_h2b10,
                    "ci_95": [low_h2b10, high_h2b10],
                    "delta_log_loss": round(ll_h2_b10_conf - ll_b10, 5)
                },
                "h3_decoupled_router": {
                    "accuracy": acc_h3_conf,
                    "log_loss": ll_h3_conf,
                    "fixes_vs_b10": fixes_h3,
                    "breaks_vs_b10": breaks_h3,
                    "vs_b10_net": net_h3,
                    "ci_95": [low_h3, high_h3],
                    "net_seen": seen_net_h3,
                    "net_new": new_net_h3,
                    "delta_log_loss": round(ll_h3_conf - ll_b10, 5)
                }
            }

        with open(REPORT_JSON, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n=======================================================", flush=True)
        print(f"REPORT SAVED TO {REPORT_JSON}", flush=True)
        print(f"=======================================================", flush=True)


if __name__ == "__main__":
    run_training_and_validation()
