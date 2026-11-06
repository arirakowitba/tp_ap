"""Evaluación EXACTA en los 3 cortes de confirmación de la configuración de candidate_model_havg_plus_full70k.csv.

Cumple estrictamente CX-0019 y CX-0020:
- Cortes de validación seller-group: split_seller_groups_r151 (semillas 2026, 314159, 20261030).
- Entrenamiento estrictamente sobre fit (cero etiquetas de tune ni confirm en fit).
- Componentes de Variant C Blend:
  1. CatBoost Bagging: 3 semillas (2026, 314159, 20261030) con EXACTAMENTE 600 iteraciones cada una,
     learning_rate=0.07, depth=6, l2_leaf_reg=6.0, cat_features=CAT_FEATURES + ['seller_id'].
  2. LightGBM: EXACTAMENTE 227 estimadores (congelados desde la mediana de tune),
     learning_rate=0.04, num_leaves=63, min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
     reg_alpha=1.0, reg_lambda=5.0, random_state=2026.
  3. Variant C Blend: 0.75 * CB_Bagged + 0.25 * LGB_Exact.
- Ensamble H-AVG-Plus: 0.50 * B10 (Claude) + 0.50 * Variant C Blend.
- Umbral de decisión fijo: 0.50.
- Reporte pareado frente a:
  * B10 (Claude)
  * A_hist (Strong baseline de Claude)
  * Variant C Blend solo
- Métricas: Accuracy global/vistos/nuevos, LogLoss global/vistos/nuevos,
  Fixes/Breaks netos, e Intervalos Bootstrap de Vendedor (2.000 réplicas).
- Guarda artefactos pareados por fila: results/intensive_20261030/paired_havg_plus_exact_seed{s}_confirm.csv.gz.
"""
import hashlib
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
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark_20261030.split_utils import split_seller_groups_r151
from src.coordination_lock import ProjectLock
from src.intensive_20261030.seller_te_ablation import compute_target_encodings_ablation
from src.intensive_20261030.shared_cache import CAT_FEATURES, build_or_load_cache

OUT_DIR = ROOT / "results" / "intensive_20261030"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_JSON = OUT_DIR / "havg_plus_exact_confirm_report.json"


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


def evaluate_havg_plus_exact(seeds=[2026, 314159, 20261030]):
    lock = ProjectLock(owner_id="antigravity_eval_havg_plus_exact")
    print("Acquiring ProjectLock for H-AVG-Plus exact confirm evaluation...", flush=True)
    with lock:
        t0_total = time.time()
        print("Loading shared feature cache...", flush=True)
        cache = build_or_load_cache()
        F_all = cache["F_r151_tr"]
        cat_tr = cache["cat_tr"]
        y_all = cache["y_tr"]
        sellers_all = cache["sellers_tr"]
        ids_all = cache["ids_tr"]
        combos_all = cache["combos_hist_tr"]
        titles_all = cache["titles_tr"]

        report = {
            "evaluation": "H-AVG-Plus Exact Holdout Confirm (CX-0019 / CX-0020)",
            "model_specification": {
                "b10": "origin/claude/posterior-y-modelos-20261027:results/round141_claude_baseline/seed{s}_fit/paired_b10_vs_A_hist_seed{s}.csv.gz",
                "catboost_bagged": {
                    "seeds": [2026, 314159, 20261030],
                    "iterations": 600,
                    "learning_rate": 0.07,
                    "depth": 6,
                    "l2_leaf_reg": 6.0,
                    "cat_features": list(CAT_FEATURES) + ["seller_id"],
                },
                "lightgbm": {
                    "n_estimators": 227,
                    "learning_rate": 0.04,
                    "num_leaves": 63,
                    "min_child_samples": 30,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 1.0,
                    "reg_lambda": 5.0,
                    "random_state": 2026,
                },
                "variant_c_blend_weights": "0.75 * CB_Bagged + 0.25 * LGB_Exact",
                "havg_plus_weights": "0.50 * B10 + 0.50 * Variant_C_Blend",
                "decision_threshold": 0.50,
            },
            "cuts": {},
        }

        cb_seeds = [2026, 314159, 20261030]

        for s in seeds:
            t0_cut = time.time()
            print(f"\n=======================================================", flush=True)
            print(f"--- EVALUATING EXACT H-AVG-PLUS ON SEED {s} ---", flush=True)
            print(f"=======================================================", flush=True)

            # 1. Load Claude's B10 and A_hist confirm predictions
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
            y_co = y_all[confirm]
            sellers_fit = sellers_all[fit]
            sellers_co = sellers_all[confirm]
            ids_co = ids_all[confirm]
            is_seen_co = np.isin(sellers_co, sellers_fit).astype(int)

            n_seen = int(is_seen_co.sum())
            n_new = int((1 - is_seen_co).sum())
            print(f"Confirm set size: {len(confirm)} (seen: {n_seen}, new: {n_new})", flush=True)

            # 3. Compute text teacher on fit and project to confirm
            print("Computing txt_sparse_p teacher features...", flush=True)
            from src.benchmark_20261030.feature_extractor import compute_txt_sparse_p
            oof_sp, (p_sp_co,) = compute_txt_sparse_p(
                combos_all, titles_all, fit, [confirm], y_fit, sellers_fit, seed=s
            )

            # 4. Target Encodings (variant_a: GroupKFold by seller on fit -> constant in fit, smoothed to confirm)
            print("Computing Group Target Encodings...", flush=True)
            fit_te, tune_te, conf_te = compute_target_encodings_ablation(
                cat_tr, fit, tune, confirm, y_fit, sellers_fit, variant="variant_a"
            )

            # 5. Assemble tabular feature sets
            X_fit = F_all.iloc[fit].copy().reset_index(drop=True)
            X_co = F_all.iloc[confirm].copy().reset_index(drop=True)

            for c in ["seller_id", "category_id", "city"]:
                X_fit[f"te_{c}"] = fit_te[c]
                X_co[f"te_{c}"] = conf_te[c]

            X_fit["txt_sparse_p"] = oof_sp
            X_co["txt_sparse_p"] = p_sp_co

            cb_cat_features = list(CAT_FEATURES) + ["seller_id"]
            X_fit["seller_id"] = cat_tr["seller_id"].iloc[fit].astype(str).values
            X_co["seller_id"] = cat_tr["seller_id"].iloc[confirm].astype(str).values
            for c in CAT_FEATURES:
                X_fit[c] = cat_tr[c].iloc[fit].astype(str).values
                X_co[c] = cat_tr[c].iloc[confirm].astype(str).values

            pool_fit = Pool(X_fit, y_fit, cat_features=cb_cat_features)
            pool_co = Pool(X_co, cat_features=cb_cat_features)

            # 6. Train 3 CatBoost models with EXACT 600 iterations
            print(f"Training 3 CatBoost models (exact 600 iterations each)...", flush=True)
            p_cb_bagged_co = np.zeros(len(confirm), dtype=np.float32)
            cb_times = []

            for cb_s in cb_seeds:
                t_cb0 = time.time()
                print(f"   -> Fitting CatBoost (seed {cb_s}, 600 iterations)...", flush=True)
                cb_m = CatBoostClassifier(
                    iterations=600,
                    learning_rate=0.07,
                    depth=6,
                    l2_leaf_reg=6.0,
                    random_seed=cb_s,
                    thread_count=4,
                    allow_writing_files=False,
                    verbose=0,
                )
                cb_m.fit(pool_fit)
                p_s_co = cb_m.predict_proba(pool_co)[:, 1].astype(np.float32)
                p_cb_bagged_co += p_s_co / len(cb_seeds)
                cb_times.append(round(time.time() - t_cb0, 1))

            print(f"   CatBoost 3-seed bagging complete in {sum(cb_times):.1f}s", flush=True)

            # 7. Train LightGBM with EXACT 227 estimators
            print(f"Training LightGBM with EXACT 227 estimators...", flush=True)
            t_lgb0 = time.time()
            X_fit_lgb = X_fit.copy()
            X_co_lgb = X_co.copy()
            for c in CAT_FEATURES + ["seller_id"]:
                cats = pd.Index(X_fit_lgb[c].unique())
                X_fit_lgb[c] = pd.Categorical(X_fit_lgb[c], categories=cats)
                X_co_lgb[c] = pd.Categorical(X_co_lgb[c], categories=cats)

            lg_m = lgb.LGBMClassifier(
                n_estimators=227,
                learning_rate=0.04,
                num_leaves=63,
                min_child_samples=30,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=5.0,
                random_state=2026,
                n_jobs=4,
                verbose=-1,
            )
            lg_m.fit(X_fit_lgb, y_fit)
            p_lgb_co = lg_m.predict_proba(X_co_lgb)[:, 1].astype(np.float32)
            lgb_elapsed = round(time.time() - t_lgb0, 1)
            print(f"   LightGBM complete in {lgb_elapsed}s", flush=True)

            # 8. Variant C Blend & H-AVG-Plus
            p_vc_blend_co = 0.75 * p_cb_bagged_co + 0.25 * p_lgb_co
            p_havg_plus_co = 0.50 * p_b10 + 0.50 * p_vc_blend_co

            # Decisions at fixed 0.50 threshold
            pred_havg_plus = (p_havg_plus_co >= 0.50).astype(int)
            pred_b10 = (p_b10 >= 0.50).astype(int)
            pred_a_hist = (p_a_hist >= 0.50).astype(int)
            pred_vc_blend = (p_vc_blend_co >= 0.50).astype(int)

            # Save paired frame
            df_paired = pd.DataFrame({
                "row_idx": confirm,
                "item_id": ids_co,
                "seller_id": sellers_co,
                "y_true": y_co,
                "is_seen": is_seen_co,
                "p_b10": p_b10,
                "p_a_hist": p_a_hist,
                "p_cb_bagged": p_cb_bagged_co,
                "p_lgb": p_lgb_co,
                "p_vc_blend": p_vc_blend_co,
                "p_havg_plus": p_havg_plus_co,
                "pred_havg_plus": pred_havg_plus,
                "pred_b10": pred_b10,
                "pred_a_hist": pred_a_hist,
                "pred_vc_blend": pred_vc_blend,
            })
            paired_path = OUT_DIR / f"paired_havg_plus_exact_seed{s}_confirm.csv.gz"
            df_paired.to_csv(paired_path, index=False, compression="gzip")
            print(f"Saved paired confirm predictions to {paired_path}", flush=True)

            # 9. Compute metrics
            seen_mask = is_seen_co == 1
            new_mask = is_seen_co == 0

            acc_global = round(float(accuracy_score(y_co, pred_havg_plus)), 5)
            acc_seen = round(float(accuracy_score(y_co[seen_mask], pred_havg_plus[seen_mask])), 5)
            acc_new = round(float(accuracy_score(y_co[new_mask], pred_havg_plus[new_mask])), 5)

            ll_global = round(calc_log_loss(y_co, p_havg_plus_co), 5)
            ll_seen = round(calc_log_loss(y_co[seen_mask], p_havg_plus_co[seen_mask]), 5)
            ll_new = round(calc_log_loss(y_co[new_mask], p_havg_plus_co[new_mask]), 5)

            # B10 metrics
            b10_acc = round(float(accuracy_score(y_co, pred_b10)), 5)
            b10_ll = round(calc_log_loss(y_co, p_b10), 5)

            # A_hist metrics
            a_hist_acc = round(float(accuracy_score(y_co, pred_a_hist)), 5)
            a_hist_ll = round(calc_log_loss(y_co, p_a_hist), 5)

            # VC_blend metrics
            vc_acc = round(float(accuracy_score(y_co, pred_vc_blend)), 5)
            vc_ll = round(calc_log_loss(y_co, p_vc_blend_co), 5)

            # Paired Bootstrap comparisons
            # vs B10
            net_vs_b10, ci_low_b10, ci_high_b10 = seller_bootstrap_ci(y_co, pred_b10, pred_havg_plus, sellers_co)
            net_seen_b10, ci_low_s_b10, ci_high_s_b10 = seller_bootstrap_ci(
                y_co[seen_mask], pred_b10[seen_mask], pred_havg_plus[seen_mask], sellers_co[seen_mask]
            )
            net_new_b10, ci_low_n_b10, ci_high_n_b10 = seller_bootstrap_ci(
                y_co[new_mask], pred_b10[new_mask], pred_havg_plus[new_mask], sellers_co[new_mask]
            )

            # vs A_hist
            net_vs_ahist, ci_low_ahist, ci_high_ahist = seller_bootstrap_ci(y_co, pred_a_hist, pred_havg_plus, sellers_co)
            net_seen_ahist, _, _ = seller_bootstrap_ci(
                y_co[seen_mask], pred_a_hist[seen_mask], pred_havg_plus[seen_mask], sellers_co[seen_mask]
            )
            net_new_ahist, _, _ = seller_bootstrap_ci(
                y_co[new_mask], pred_a_hist[new_mask], pred_havg_plus[new_mask], sellers_co[new_mask]
            )

            # vs VC_blend
            net_vs_vc, ci_low_vc, ci_high_vc = seller_bootstrap_ci(y_co, pred_vc_blend, pred_havg_plus, sellers_co)
            net_seen_vc, _, _ = seller_bootstrap_ci(
                y_co[seen_mask], pred_vc_blend[seen_mask], pred_havg_plus[seen_mask], sellers_co[seen_mask]
            )
            net_new_vc, _, _ = seller_bootstrap_ci(
                y_co[new_mask], pred_vc_blend[new_mask], pred_havg_plus[new_mask], sellers_co[new_mask]
            )

            fixes_b10 = int(((pred_havg_plus == y_co) & (pred_b10 != y_co)).sum())
            breaks_b10 = int(((pred_havg_plus != y_co) & (pred_b10 == y_co)).sum())

            elapsed_cut = round(time.time() - t0_cut, 1)

            cut_result = {
                "elapsed_seconds": elapsed_cut,
                "n_confirm": len(confirm),
                "n_seen": n_seen,
                "n_new": n_new,
                "havg_plus": {
                    "accuracy": {"global": acc_global, "seen": acc_seen, "new": acc_new},
                    "log_loss": {"global": ll_global, "seen": ll_seen, "new": ll_new},
                },
                "b10_baseline": {
                    "accuracy": b10_acc,
                    "log_loss": b10_ll,
                },
                "a_hist_baseline": {
                    "accuracy": a_hist_acc,
                    "log_loss": a_hist_ll,
                },
                "vc_blend_component": {
                    "accuracy": vc_acc,
                    "log_loss": vc_ll,
                },
                "vs_b10_paired": {
                    "fixes": fixes_b10,
                    "breaks": breaks_b10,
                    "net_total": net_vs_b10,
                    "ci_95": [ci_low_b10, ci_high_b10],
                    "net_seen": net_seen_b10,
                    "ci_95_seen": [ci_low_s_b10, ci_high_s_b10],
                    "net_new": net_new_b10,
                    "ci_95_new": [ci_low_n_b10, ci_high_n_b10],
                    "delta_log_loss": round(ll_global - b10_ll, 5),
                },
                "vs_a_hist_paired": {
                    "net_total": net_vs_ahist,
                    "ci_95": [ci_low_ahist, ci_high_ahist],
                    "net_seen": net_seen_ahist,
                    "net_new": net_new_ahist,
                    "delta_log_loss": round(ll_global - a_hist_ll, 5),
                },
                "vs_vc_blend_paired": {
                    "net_total": net_vs_vc,
                    "ci_95": [ci_low_vc, ci_high_vc],
                    "net_seen": net_seen_vc,
                    "net_new": net_new_vc,
                    "delta_log_loss": round(ll_global - vc_ll, 5),
                },
            }
            report["cuts"][str(s)] = cut_result

            print(f"Results for seed {s}:", flush=True)
            print(f"  H-AVG-Plus Acc: {acc_global} (seen: {acc_seen}, new: {acc_new}) | LL: {ll_global}", flush=True)
            print(f"  B10 Acc:        {b10_acc} | LL: {b10_ll}", flush=True)
            print(f"  vs B10 Net:     {net_vs_b10} [{ci_low_b10}, {ci_high_b10}] (seen: {net_seen_b10}, new: {net_new_b10})", flush=True)
            print(f"  vs A_hist Net:  {net_vs_ahist} [{ci_low_ahist}, {ci_high_ahist}] (seen: {net_seen_ahist}, new: {net_new_ahist})", flush=True)

        elapsed_all = round(time.time() - t0_total, 1)
        report["total_elapsed_seconds"] = elapsed_all
        report["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        with open(REPORT_JSON, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"\n=======================================================", flush=True)
        print(f"EXACT EVALUATION COMPLETE IN {elapsed_all}s", flush=True)
        print(f"Report saved to {REPORT_JSON}", flush=True)
        print(f"=======================================================", flush=True)
        return report


if __name__ == "__main__":
    evaluate_havg_plus_exact()
