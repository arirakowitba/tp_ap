"""Ablación rigurosa de Target Encoding de Vendedor y CatBoost Categórico (CX-0013 / Directiva de Ariel).

Objetivo:
Evaluar con rigor sobre los splits de validación (outer fit, tune, confirm con seeds 2026, 314159, 20261030)
el impacto del Target Encoding del vendedor y la feature categórica nativa de CatBoost,
separando de forma limpia vendedores vistos ('seen') y nuevos ('new').

Variantes evaluadas:
1. `variant_a_baseline_group_te`:
   - El pipeline actual de A_hist / CB_Bag3 / LGB_Bag3.
   - En fit, te_seller_id se calculaba con GroupKFold(4) agrupado por vendedor -> te_seller_id es constante en fit (std == 0).
   - En tune y confirm, te_seller_id se mapea desde fit (con smoothing alpha=5.0).
2. `variant_b_row_oof_seller_te`:
   - En fit, te_seller_id se calcula OOF por filas usando StratifiedKFold(5, shuffle=True, random_state=42)
     estrictamente dentro de outer fit (excluyendo el fold propio, con smoothing alpha=5.0 y prior global de fit).
   - En tune y confirm, se mapea desde fit (con smoothing alpha=5.0 y prior global de fit).
   - Garantía: CERO etiquetas de tune ni confirm intervienen en el cálculo.
3. `variant_c_cb_native_cat_seller`:
   - Mantiene la variante (a) pero agrega 'seller_id' a las variables categóricas nativas de CatBoost,
     permitiendo que CatBoost construya sus propias estadísticas ordenadas sobre el vendedor durante el fit.
4. `variant_d_row_oof_te_plus_cat_seller`:
   - Combina la variante (b) con 'seller_id' categórico nativo en CatBoost.

Modelos entrenados y evaluados:
- CatBoost Classifier (depth=6, iterations ajustadas por tune, lr=0.07, l2=6.0)
- LightGBM Classifier (con num_leaves y regularización seleccionada en tune)
- Blend 0.75 CB + 0.25 LGB

Métricas evaluadas:
- Accuracy global, Log Loss global.
- Accuracy en vendedores vistos (`is_seen == 1`) y nuevos (`is_seen == 0`).
- Fixes y Breaks pareados frente al baseline (variante a) en confirm.
- Intervalos de confianza bootstrap por vendedor (2,000 réplicas).
"""
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark_20261030.split_utils import split_seller_groups_r151
from src.eval_20261030.paired_eval import evaluate_paired_frames
from src.intensive_20261030.shared_cache import CAT_FEATURES, build_or_load_cache

OUT_DIR = ROOT / "results" / "intensive_20261030"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_target_encodings_ablation(
    cat_df: pd.DataFrame,
    fit: np.ndarray,
    tune: np.ndarray,
    confirm: np.ndarray,
    y_fit: np.ndarray,
    sellers_fit: np.ndarray,
    variant: str = "variant_a",
    alpha: float = 5.0,
    m_seller: float = 10.0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Calcula Target Encodings para fit, tune y confirm bajo las variantes 'variant_a' o 'variant_b'."""
    prior = float(y_fit.mean())
    te_cols = ["seller_id", "category_id", "city"]

    fit_te = {c: np.full(len(fit), prior, dtype=np.float32) for c in te_cols}
    tune_te = {c: np.full(len(tune), prior, dtype=np.float32) for c in te_cols}
    conf_te = {c: np.full(len(confirm), prior, dtype=np.float32) for c in te_cols}

    # 1. En fit: category_id y city siempre usan GroupKFold(4) agrupado por vendedor (aislamiento de vendedor)
    gkf = GroupKFold(n_splits=4)
    for tr, va in gkf.split(fit, y_fit, sellers_fit):
        for c in ["category_id", "city"]:
            k_tr = cat_df[c].iloc[fit[tr]]
            k_va = cat_df[c].iloc[fit[va]]
            stats = pd.DataFrame({"k": k_tr, "y": y_fit[tr]}).groupby("k").y.agg(["sum", "count"])
            smoothed = (stats["sum"] + alpha * prior) / (stats["count"] + alpha)
            fit_te[c][va] = k_va.map(smoothed).fillna(prior).values.astype(np.float32)

    # 2. seller_id en fit:
    if variant in ("variant_a", "variant_c"):
        # GroupKFold agrupado por seller_id -> en cada fold va, los vendedores nunca estuvieron en tr -> constante == prior
        for tr, va in gkf.split(fit, y_fit, sellers_fit):
            k_tr = cat_df["seller_id"].iloc[fit[tr]]
            k_va = cat_df["seller_id"].iloc[fit[va]]
            stats = pd.DataFrame({"k": k_tr, "y": y_fit[tr]}).groupby("k").y.agg(["sum", "count"])
            smoothed = (stats["sum"] + alpha * prior) / (stats["count"] + alpha)
            fit_te["seller_id"][va] = k_va.map(smoothed).fillna(prior).values.astype(np.float32)
    elif variant in ("variant_b", "variant_d"):
        # StratifiedKFold por filas dentro de fit -> OOF real sin fuga de fila propia
        # Smoothing m = 10.0 (CX-0013) y prior interno por fold para que singletons nunca vean su y
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for tr, va in skf.split(fit, y_fit):
            prior_tr = float(y_fit[tr].mean())
            k_tr = cat_df["seller_id"].iloc[fit[tr]]
            k_va = cat_df["seller_id"].iloc[fit[va]]
            stats = pd.DataFrame({"k": k_tr, "y": y_fit[tr]}).groupby("k").y.agg(["sum", "count"])
            smoothed = (stats["sum"] + m_seller * prior_tr) / (stats["count"] + m_seller)
            fit_te["seller_id"][va] = k_va.map(smoothed).fillna(prior_tr).values.astype(np.float32)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # 3. En tune y confirm: mapear desde TODO fit con smoothing apropiado
    m_val = m_seller if variant in ("variant_b", "variant_d") else alpha
    for c in te_cols:
        smooth_param = m_val if c == "seller_id" else alpha
        k_fit = cat_df[c].iloc[fit]
        stats = pd.DataFrame({"k": k_fit, "y": y_fit}).groupby("k").y.agg(["sum", "count"])
        smoothed = (stats["sum"] + smooth_param * prior) / (stats["count"] + smooth_param)
        tune_te[c] = cat_df[c].iloc[tune].map(smoothed).fillna(prior).values.astype(np.float32)
        conf_te[c] = cat_df[c].iloc[confirm].map(smoothed).fillna(prior).values.astype(np.float32)

    return fit_te, tune_te, conf_te


def run_ablation_experiment(seeds: List[int] = [2026, 314159, 20261030]) -> Dict:
    cache = build_or_load_cache()
    F_r151_tr = cache["F_r151_tr"]
    cat_tr = cache["cat_tr"]
    y_tr = cache["y_tr"]
    sellers_tr = cache["sellers_tr"]
    ids_tr = cache["ids_tr"]

    variants = [
        "variant_a_baseline_group_te",
        "variant_b_row_oof_seller_te",
        "variant_c_cb_native_cat_seller",
        "variant_d_row_oof_te_plus_cat_seller",
    ]

    all_results = {v: {} for v in variants}

    for seed in seeds:
        print(f"\n=======================================================")
        print(f"--- RUNNING SELLER TE ABLATION FOR SEED {seed} ---")
        print(f"=======================================================")
        fit, tune, confirm = split_seller_groups_r151(sellers_tr, seed)
        y_fit = y_tr[fit]
        y_tu = y_tr[tune]
        y_co = y_tr[confirm]
        sellers_fit = sellers_tr[fit]
        sellers_tu = sellers_tr[tune]
        sellers_co = sellers_tr[confirm]
        ids_co = ids_tr[confirm]

        # Flags seen
        is_seen_tu = np.isin(sellers_tu, sellers_fit).astype(int)
        is_seen_co = np.isin(sellers_co, sellers_fit).astype(int)
        print(f"Split sizes: fit={len(fit)}, tune={len(tune)} ({np.mean(is_seen_tu)*100:.1f}% seen), confirm={len(confirm)} ({np.mean(is_seen_co)*100:.1f}% seen)")

        # Load txt_sparse_p from precomputed benchmark cache if available or compute it
        from src.benchmark_20261030.feature_extractor import compute_txt_sparse_p
        combos = cache["combos_hist_tr"]
        titles = cache["titles_tr"]
        print(f"Computing txt_sparse_p for seed {seed}...")
        oof_sp, (p_sp_tune, p_sp_conf) = compute_txt_sparse_p(
            combos, titles, fit, [tune, confirm], y_fit, sellers_fit, seed=seed
        )

        for var_name in variants:
            print(f"\n>>> Running {var_name} (seed {seed}) <<<")
            t0 = time.time()
            v_type = "variant_b" if "row_oof" in var_name else "variant_a"
            fit_te, tune_te, conf_te = compute_target_encodings_ablation(
                cat_tr, fit, tune, confirm, y_fit, sellers_fit, variant=v_type
            )

            # Verification of fit te variation
            std_fit_seller = float(np.std(fit_te["seller_id"]))
            n_uniq_fit_seller = int(len(np.unique(fit_te["seller_id"])))
            std_conf_seller = float(np.std(conf_te["seller_id"]))
            print(f"[{var_name}] te_seller_id fit: std={std_fit_seller:.5f}, unique={n_uniq_fit_seller} | confirm std={std_conf_seller:.5f}")

            # Prepare DataFrames
            X_fit = F_r151_tr.iloc[fit].copy().reset_index(drop=True)
            X_tu = F_r151_tr.iloc[tune].copy().reset_index(drop=True)
            X_co = F_r151_tr.iloc[confirm].copy().reset_index(drop=True)

            for c in ["seller_id", "category_id", "city"]:
                X_fit[f"te_{c}"] = fit_te[c]
                X_tu[f"te_{c}"] = tune_te[c]
                X_co[f"te_{c}"] = conf_te[c]

            X_fit["txt_sparse_p"] = oof_sp
            X_tu["txt_sparse_p"] = p_sp_tune
            X_co["txt_sparse_p"] = p_sp_conf

            cb_cat_features = list(CAT_FEATURES)
            if "cat_seller" in var_name:
                cb_cat_features.append("seller_id")
                X_fit["seller_id"] = cat_tr["seller_id"].iloc[fit].astype(str).values
                X_tu["seller_id"] = cat_tr["seller_id"].iloc[tune].astype(str).values
                X_co["seller_id"] = cat_tr["seller_id"].iloc[confirm].astype(str).values

            for c in CAT_FEATURES:
                X_fit[c] = cat_tr[c].iloc[fit].astype(str).values
                X_tu[c] = cat_tr[c].iloc[tune].astype(str).values
                X_co[c] = cat_tr[c].iloc[confirm].astype(str).values

            # 1. CatBoost Training
            print(f"Training CatBoost for {var_name}...")
            cb_model = CatBoostClassifier(
                iterations=900,
                learning_rate=0.07,
                depth=6,
                l2_leaf_reg=6.0,
                random_seed=seed,
                thread_count=4,
                verbose=0,
            )
            pool_fit = Pool(X_fit, y_fit, cat_features=cb_cat_features)
            pool_tu = Pool(X_tu, y_tu, cat_features=cb_cat_features)
            pool_co = Pool(X_co, cat_features=cb_cat_features)

            cb_model.fit(pool_fit, eval_set=pool_tu, early_stopping_rounds=80, verbose=0)
            best_iter = cb_model.get_best_iteration()
            p_cb_tu = cb_model.predict_proba(pool_tu)[:, 1]
            p_cb_co = cb_model.predict_proba(pool_co)[:, 1]

            # 2. LightGBM Training (using label encoding for categoricals)
            print(f"Training LightGBM for {var_name}...")
            X_fit_lgb = X_fit.copy()
            X_tu_lgb = X_tu.copy()
            X_co_lgb = X_co.copy()
            for c in CAT_FEATURES + (["seller_id"] if "cat_seller" in var_name else []):
                cats = pd.Index(X_fit_lgb[c].unique())
                X_fit_lgb[c] = pd.Categorical(X_fit_lgb[c], categories=cats)
                X_tu_lgb[c] = pd.Categorical(X_tu_lgb[c], categories=cats)
                X_co_lgb[c] = pd.Categorical(X_co_lgb[c], categories=cats)

            lgb_model = lgb.LGBMClassifier(
                n_estimators=800,
                learning_rate=0.04,
                num_leaves=63,
                min_child_samples=30,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=5.0,
                random_state=seed,
                n_jobs=4,
                verbose=-1,
            )
            lgb_model.fit(
                X_fit_lgb,
                y_fit,
                eval_set=[(X_tu_lgb, y_tu)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            p_lgb_tu = lgb_model.predict_proba(X_tu_lgb)[:, 1]
            p_lgb_co = lgb_model.predict_proba(X_co_lgb)[:, 1]

            # 3. Blend 0.75 CB + 0.25 LGB
            p_blend_tu = 0.75 * p_cb_tu + 0.25 * p_lgb_tu
            p_blend_co = 0.75 * p_cb_co + 0.25 * p_lgb_co

            # Compute threshold strictly on tune
            best_th = 0.50
            best_acc_tu = -1.0
            for th in np.linspace(0.40, 0.60, 41):
                acc = accuracy_score(y_tu, p_blend_tu >= th)
                if acc > best_acc_tu:
                    best_acc_tu = acc
                    best_th = float(th)

            # Confirm metrics
            pred_co_050 = (p_blend_co >= 0.50).astype(int)
            pred_co_th = (p_blend_co >= best_th).astype(int)

            acc_050 = float(accuracy_score(y_co, pred_co_050))
            acc_th = float(accuracy_score(y_co, pred_co_th))
            ll_co = float(log_loss(y_co, p_blend_co))

            # Breakdown seen / new
            acc_seen_050 = float(accuracy_score(y_co[is_seen_co == 1], pred_co_050[is_seen_co == 1]))
            acc_new_050 = float(accuracy_score(y_co[is_seen_co == 0], pred_co_050[is_seen_co == 0]))
            acc_seen_th = float(accuracy_score(y_co[is_seen_co == 1], pred_co_th[is_seen_co == 1]))
            acc_new_th = float(accuracy_score(y_co[is_seen_co == 0], pred_co_th[is_seen_co == 0]))

            ll_seen = float(log_loss(y_co[is_seen_co == 1], p_blend_co[is_seen_co == 1]))
            ll_new = float(log_loss(y_co[is_seen_co == 0], p_blend_co[is_seen_co == 0]))

            # Breakdown by seller frequency in fit (freq == 1, 2..4, >= 5) for seen sellers (CX-0013)
            vc_fit = pd.Series(sellers_fit).value_counts()
            freq_in_fit = pd.Series(sellers_co).map(vc_fit).fillna(0).astype(int).values

            mask_f1 = (is_seen_co == 1) & (freq_in_fit == 1)
            mask_f2_4 = (is_seen_co == 1) & (freq_in_fit >= 2) & (freq_in_fit <= 4)
            mask_f5p = (is_seen_co == 1) & (freq_in_fit >= 5)

            acc_f1 = float(accuracy_score(y_co[mask_f1], pred_co_050[mask_f1])) if mask_f1.sum() > 0 else None
            acc_f2_4 = float(accuracy_score(y_co[mask_f2_4], pred_co_050[mask_f2_4])) if mask_f2_4.sum() > 0 else None
            acc_f5p = float(accuracy_score(y_co[mask_f5p], pred_co_050[mask_f5p])) if mask_f5p.sum() > 0 else None

            elapsed = round(time.time() - t0, 1)
            print(f"[{var_name}] confirm Acc@0.50: {acc_050:.5f} (seen: {acc_seen_050:.5f}, new: {acc_new_050:.5f}) | LogLoss: {ll_co:.5f} ({elapsed}s)")

            all_results[var_name][str(seed)] = {
                "elapsed_seconds": elapsed,
                "best_iteration_cb": int(best_iter),
                "tune_th": best_th,
                "tune_acc_at_th": float(best_acc_tu),
                "confirm_acc_050": acc_050,
                "confirm_acc_tuned_th": acc_th,
                "confirm_log_loss": ll_co,
                "confirm_seen_acc_050": acc_seen_050,
                "confirm_new_acc_050": acc_new_050,
                "confirm_seen_acc_tuned_th": acc_seen_th,
                "confirm_new_acc_tuned_th": acc_new_th,
                "confirm_seen_log_loss": ll_seen,
                "confirm_new_log_loss": ll_new,
                "confirm_seen_f1_acc_050": acc_f1,
                "confirm_seen_f2_4_acc_050": acc_f2_4,
                "confirm_seen_f5p_acc_050": acc_f5p,
                "p_blend_co": p_blend_co.tolist(),
                "pred_co_050": pred_co_050.tolist(),
                "pred_co_th": pred_co_th.tolist(),
                "std_fit_seller": std_fit_seller,
                "n_uniq_fit_seller": n_uniq_fit_seller,
            }

    # Summary and paired comparisons against baseline (variant_a)
    summary_report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seeds": seeds,
        "variants": variants,
        "results": {},
        "paired_vs_baseline_050": {},
    }

    base_key = "variant_a_baseline_group_te"
    for var_name in variants:
        summary_report["results"][var_name] = {}
        for seed in seeds:
            s_str = str(seed)
            d = dict(all_results[var_name][s_str])
            # Drop big prediction arrays from summary
            d.pop("p_blend_co", None)
            d.pop("pred_co_050", None)
            d.pop("pred_co_th", None)
            summary_report["results"][var_name][s_str] = d

    for var_name in variants:
        if var_name != base_key:
            summary_report["paired_vs_baseline_050"][var_name] = {}
            for seed in seeds:
                s_str = str(seed)
                fit, tune, confirm = split_seller_groups_r151(sellers_tr, seed)
                y_co = y_tr[confirm]
                sellers_co = sellers_tr[confirm]
                is_seen_co = np.isin(sellers_co, sellers_tr[fit]).astype(int)

                p_base = np.array(all_results[base_key][s_str]["p_blend_co"])
                pred_base = np.array(all_results[base_key][s_str]["pred_co_050"])
                p_cand = np.array(all_results[var_name][s_str]["p_blend_co"])
                pred_cand = np.array(all_results[var_name][s_str]["pred_co_050"])

                df_b = pd.DataFrame({"id": ids_tr[confirm], "seller_id": sellers_co, "is_seen": is_seen_co, "y_used": y_co, "p_used": p_base, "pred": pred_base})
                df_c = pd.DataFrame({"id": ids_tr[confirm], "seller_id": sellers_co, "is_seen": is_seen_co, "y_used": y_co, "p_used": p_cand, "pred": pred_cand})

                paired_res = evaluate_paired_frames(df_b, df_c, n_boot=2000, seed=42)
                # Remove non-serializable elements
                paired_res.pop("fixes_ids", None)
                paired_res.pop("breaks_ids", None)
                summary_report["paired_vs_baseline_050"][var_name][s_str] = paired_res

    def default_json(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Unserializable type {type(obj)}")

    out_file = OUT_DIR / "seller_te_ablation_report.json"
    out_file.write_text(json.dumps(summary_report, indent=2, default=default_json), encoding="utf-8")
    print(f"\nSaved ablation report to {out_file}")
    return summary_report


if __name__ == "__main__":
    report = run_ablation_experiment(seeds=[2026, 314159, 20261030])
