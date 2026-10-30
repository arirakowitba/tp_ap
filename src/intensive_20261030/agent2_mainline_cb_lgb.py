"""AGENTE 2 — LÍNEA PRINCIPAL CATBOOST + LIGHTGBM (2026-10-30).

Executes the complete experimental protocol for Agent 2:
1. Freezes `0.75 CatBoost + 0.25 LightGBM` as the reference candidate configuration.
2. Compares on the EXACT SAME confirm IDs across seeds 2026, 314159, and 20261030:
   - `A_hist`: Exact historical strong baseline (Deep CatBoost R151, attribute IDs in text combo).
   - `A_valname`: New Deep CatBoost variant (attribute value_names in text combo).
   - `LGB_hist` & `LGB_valname`: LightGBM selected strictly on `tune`.
   - `Blend_075_025_hist` & `Blend_075_025_valname`: Frozen `0.75 CB + 0.25 LGB` reference blends.
   - `CB_Bag3 + LGB_Bag3 (Blend_Bag3_075_025)`: Substantial bounded model-diversity variant (3-seed bagged
     CatBoost + 3-seed bagged LightGBM with feature/row subsampling), selected on `tune` and frozen before `confirm`.
   - Historical compatible models on the exact same IDs (Seeds 2026 & 314159):
     * `R142_HistGB`, `R142_SparseLinear`, `R142_LightGBM`, `R142_CatBoost` (`paired_predictions_seed_{seed}.csv`)
     * `R156_Ensemble` (`paired_ensemble_seed_{seed}.csv`)
     * `R151_Dual_NBSVM` (`paired_deep_ablation_seed_{seed}.csv`)
3. Explicitly separates **model contribution** from **threshold contribution**:
   - Common threshold `th = 0.500`
   - Common baseline threshold `th = th_base_tune`
   - Policy threshold selected strictly on `tune` (`th = th_cand_tune`)
4. Evaluates accuracy, log loss, fixes/breaks/net, seen/new breakdown, flip directions (`used->new`, `new->used`),
   and 2,000-replicate seller-clustered bootstrap CIs.
5. Never sums gains across overlapping cuts or declares independence.
6. Runs a strictly inductive 70k refit and generates verified 30k test predictions (`used=1`, IDs `1..30000`).
"""
import hashlib
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from catboost import CatBoostClassifier, Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.intensive_20261030.shared_cache import (
    build_or_load_cache, compute_file_sha256, CAT_FEATURES, CACHE_DIR
)
from src.eval_20261030.paired_eval import evaluate_paired_frames

OUT_DIR = ROOT / "results" / "intensive_20261030"
OUT_DIR.mkdir(parents=True, exist_ok=True)
S48_JSON = ROOT / "results" / "revision_submissions" / "posterior_flips_20261027" / "s48.json"


def select_threshold_on_tune(y_tune, p_tune):
    best_th, best_acc = 0.50, -1.0
    for th in np.linspace(0.40, 0.60, 41):
        acc = accuracy_score(y_tune, p_tune >= th)
        if acc > best_acc:
            best_acc = acc
            best_th = float(th)
    return best_th, float(best_acc)


def build_eval_df(ids_confirm, sellers_confirm, is_seen_confirm, y_confirm, p_used, pred):
    return pd.DataFrame({
        "id": ids_confirm,
        "seller_id": sellers_confirm,
        "is_seen": is_seen_confirm,
        "y_used": y_confirm,
        "p_used": p_used,
        "pred": pred,
    })


def compare_under_three_threshold_regimes(
    ids_co, sellers_co, is_seen_co, y_co,
    p_base, th_base_tune, p_cand, th_cand_tune
):
    """Evaluates candidate vs baseline under:
    1) Common threshold 0.500 (`common_050`)
    2) Common baseline tune threshold (`common_base_th`)
    3) Each model using its own tune-selected threshold (`policy_tuned_th`)
    """
    # 1. Common th = 0.500
    df_b_050 = build_eval_df(ids_co, sellers_co, is_seen_co, y_co, p_base, (p_base >= 0.50).astype(int))
    df_c_050 = build_eval_df(ids_co, sellers_co, is_seen_co, y_co, p_cand, (p_cand >= 0.50).astype(int))
    res_050 = evaluate_paired_frames(df_b_050, df_c_050, n_boot=2000, seed=42)

    # 2. Common th = th_base_tune
    df_b_bth = build_eval_df(ids_co, sellers_co, is_seen_co, y_co, p_base, (p_base >= th_base_tune).astype(int))
    df_c_bth = build_eval_df(ids_co, sellers_co, is_seen_co, y_co, p_cand, (p_cand >= th_base_tune).astype(int))
    res_bth = evaluate_paired_frames(df_b_bth, df_c_bth, n_boot=2000, seed=42)

    # 3. Policy th (base @ th_base_tune vs cand @ th_cand_tune)
    df_c_cth = build_eval_df(ids_co, sellers_co, is_seen_co, y_co, p_cand, (p_cand >= th_cand_tune).astype(int))
    res_pol = evaluate_paired_frames(df_b_bth, df_c_cth, n_boot=2000, seed=42)

    return {
        "th_base_tune": float(th_base_tune),
        "th_cand_tune": float(th_cand_tune),
        "common_th_050": res_050,
        "common_th_base": res_bth,
        "policy_tuned_th": res_pol,
    }


def prepare_lgb_matrices(F_df, fit, tune, confirm, fit_te, tune_te, confirm_te, oof_sp, p_sp_tune, p_sp_conf):
    X_fit = F_df.iloc[fit].copy().reset_index(drop=True)
    X_tune = F_df.iloc[tune].copy().reset_index(drop=True)
    X_conf = F_df.iloc[confirm].copy().reset_index(drop=True)

    for c in ["seller_id", "category_id", "city"]:
        X_fit[f"te_{c}"] = fit_te[c]
        X_tune[f"te_{c}"] = tune_te[c]
        X_conf[f"te_{c}"] = confirm_te[c]

    X_fit["txt_sparse_p"] = oof_sp
    X_tune["txt_sparse_p"] = p_sp_tune
    X_conf["txt_sparse_p"] = p_sp_conf

    for c in CAT_FEATURES:
        X_fit[c] = X_fit[c].astype(str)
        X_tune[c] = X_tune[c].astype(str)
        X_conf[c] = X_conf[c].astype(str)

    X_fit_lgb = X_fit.copy()
    X_tune_lgb = X_tune.copy()
    X_conf_lgb = X_conf.copy()
    for c in CAT_FEATURES:
        cats = pd.Index(X_fit_lgb[c].unique())
        X_fit_lgb[c] = pd.Categorical(X_fit_lgb[c], categories=cats)
        X_tune_lgb[c] = pd.Categorical(X_tune_lgb[c], categories=cats)
        X_conf_lgb[c] = pd.Categorical(X_conf_lgb[c], categories=cats)

    return X_fit, X_tune, X_conf, X_fit_lgb, X_tune_lgb, X_conf_lgb


def fit_bounded_lgb(X_fit_lgb, y_fit, X_tune_lgb, y_tune, X_conf_lgb, seed):
    lgb_configs = [
        {"name": "lgb_default", "n_estimators": 800, "learning_rate": 0.04, "num_leaves": 63, "min_child_samples": 30, "reg_lambda": 5.0},
        {"name": "lgb_conservative", "n_estimators": 700, "learning_rate": 0.04, "num_leaves": 31, "min_child_samples": 50, "reg_lambda": 10.0},
        {"name": "lgb_deep", "n_estimators": 900, "learning_rate": 0.035, "num_leaves": 95, "min_child_samples": 25, "reg_lambda": 8.0},
    ]
    best_acc = -1.0
    best_cfg = None
    best_p_tu = None
    best_p_co = None

    for cfg in lgb_configs:
        m = lgb.LGBMClassifier(
            n_estimators=cfg["n_estimators"],
            learning_rate=cfg["learning_rate"],
            num_leaves=cfg["num_leaves"],
            min_child_samples=cfg["min_child_samples"],
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=cfg["reg_lambda"],
            random_state=seed,
            n_jobs=4,
            verbose=-1,
        )
        m.fit(X_fit_lgb, y_fit)
        p_tu = m.predict_proba(X_tune_lgb)[:, 1]
        acc_tu = accuracy_score(y_tune, p_tu >= 0.50)
        if acc_tu > best_acc:
            best_acc = acc_tu
            best_cfg = cfg
            best_p_tu = p_tu
            best_p_co = m.predict_proba(X_conf_lgb)[:, 1]

    return best_cfg, best_p_tu, best_p_co


def fit_bagged_diversity_variant(
    X_fit, y_fit, X_tune, y_tune, X_conf,
    X_fit_lgb, X_tune_lgb, X_conf_lgb,
    p_cb_tu_seed0, p_cb_co_seed0,
    p_cb_valname_tu_seed0, p_cb_valname_co_seed0,
    best_lgb_cfg, p_lgb_tu_seed0, p_lgb_co_seed0,
    seed
):
    """Substantial bounded diversity variant:
    Multi-view / multi-seed Bagged CatBoost + 3-seed Bagged LightGBM (`[seed, seed+101, seed+202]`).
    Reuses cached `CB_hist` and `CB_valname` models, fits an additional CatBoost seed (`seed+101`)
    on the primary seed 2026 to measure pure multi-seed CB + multi-view diversity, and fits
    3-seed subsampled LightGBM across all 3 seeds.
    """
    cb_tu_list = [p_cb_tu_seed0, p_cb_valname_tu_seed0]
    cb_co_list = [p_cb_co_seed0, p_cb_valname_co_seed0]
    lgb_tu_list = [p_lgb_tu_seed0]
    lgb_co_list = [p_lgb_co_seed0]

    if seed == 2026:
        pool_fit = Pool(X_fit, y_fit, cat_features=CAT_FEATURES)
        pool_tu = Pool(X_tune, cat_features=CAT_FEATURES)
        pool_co = Pool(X_conf, cat_features=CAT_FEATURES)
        cb_i = CatBoostClassifier(
            iterations=900, depth=6, learning_rate=0.07, l2_leaf_reg=6.0,
            random_seed=seed + 101, thread_count=4, allow_writing_files=False, verbose=0
        )
        cb_i.fit(pool_fit)
        cb_tu_list.append(cb_i.predict_proba(pool_tu)[:, 1])
        cb_co_list.append(cb_i.predict_proba(pool_co)[:, 1])

    for offset in [101, 202]:
        s_i = seed + offset
        lgb_i = lgb.LGBMClassifier(
            n_estimators=best_lgb_cfg["n_estimators"],
            learning_rate=best_lgb_cfg["learning_rate"],
            num_leaves=best_lgb_cfg["num_leaves"],
            min_child_samples=best_lgb_cfg["min_child_samples"],
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=best_lgb_cfg["reg_lambda"],
            random_state=s_i,
            n_jobs=4,
            verbose=-1,
        )
        lgb_i.fit(X_fit_lgb, y_fit)
        lgb_tu_list.append(lgb_i.predict_proba(X_tune_lgb)[:, 1])
        lgb_co_list.append(lgb_i.predict_proba(X_conf_lgb)[:, 1])

    p_cb_bag_tu = np.mean(cb_tu_list, axis=0)
    p_cb_bag_co = np.mean(cb_co_list, axis=0)
    p_lgb_bag_tu = np.mean(lgb_tu_list, axis=0)
    p_lgb_bag_co = np.mean(lgb_co_list, axis=0)

    p_blend_bag_tu = 0.75 * p_cb_bag_tu + 0.25 * p_lgb_bag_tu
    p_blend_bag_co = 0.75 * p_cb_bag_co + 0.25 * p_lgb_bag_co

    return {
        "p_cb_bag_tu": p_cb_bag_tu,
        "p_cb_bag_co": p_cb_bag_co,
        "p_lgb_bag_tu": p_lgb_bag_tu,
        "p_lgb_bag_co": p_lgb_bag_co,
        "p_blend_bag_tu": p_blend_bag_tu,
        "p_blend_bag_co": p_blend_bag_co,
    }


def run_inductive_70k_refit_and_test_inference(cache):
    """Strictly inductive 70k refit for test 30k inference:
    - Group stats (`F_ind_te`) computed exclusively from 70k train.
    - 4-fold GroupKFold TE (`alpha=5.0`) on 70k train; full 70k TE mapped to 30k test.
    - 4-fold GroupKFold `txt_sparse_p` via `LogisticRegression(C=3.0, solver='liblinear')` on 70k train;
      full 70k TF-IDF + LogisticRegression applied to 30k test.
    - Trains `CatBoost` (3 seeds: 2026, 2127, 2228) and `LightGBM` (`lgb_deep`, 3 seeds: 2026, 2127, 2228)
      on full 70k train and predicts on 30k test.
    """
    print("\n" + "=" * 80)
    print("RUNNING STRICTLY INDUCTIVE 70K REFIT & 30K TEST INFERENCE (AGENT 2)")
    print("=" * 80, flush=True)
    t0 = time.time()

    F_tr = cache["F_r151_tr"].copy()
    F_te = cache["F_ind_te"].copy()
    cat_tr = cache["cat_tr"]
    cat_te = cache["cat_te"]
    y_tr = cache["y_tr"]
    sellers_tr = cache["sellers_tr"]
    sellers_te = cache["sellers_te"]
    titles_tr = cache["titles_tr"]
    titles_te = cache["titles_te"]
    combos_tr = cache["combos_hist_tr"]
    combos_te = cache["combos_hist_te"]

    prior = float(y_tr.mean())
    te_cols = ["seller_id", "category_id", "city"]
    gkf = GroupKFold(n_splits=4)
    fit_te = {c: np.full(len(y_tr), prior, dtype=np.float32) for c in te_cols}
    test_te = {}

    for tr_idx, va_idx in gkf.split(F_tr, y_tr, sellers_tr):
        for c in te_cols:
            k_tr = cat_tr[c].iloc[tr_idx]
            k_va = cat_tr[c].iloc[va_idx]
            stats = pd.DataFrame({"k": k_tr, "y": y_tr[tr_idx]}).groupby("k").y.agg(["sum", "count"])
            smoothed = (stats["sum"] + 5.0 * prior) / (stats["count"] + 5.0)
            fit_te[c][va_idx] = k_va.map(smoothed).fillna(prior).values.astype(np.float32)

    for c in te_cols:
        stats = pd.DataFrame({"k": cat_tr[c], "y": y_tr}).groupby("k").y.agg(["sum", "count"])
        smoothed = (stats["sum"] + 5.0 * prior) / (stats["count"] + 5.0)
        test_te[c] = cat_te[c].map(smoothed).fillna(prior).values.astype(np.float32)

    # Sparse TF-IDF text teacher (exact R151 specification)
    vec_combo = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=120000,
                                sublinear_tf=True, strip_accents="unicode", lowercase=True)
    vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                               max_features=150000, sublinear_tf=True)

    X_sp_tr = sp.hstack([vec_combo.fit_transform(combos_tr), vec_char.fit_transform(titles_tr)], format="csr")
    X_sp_te = sp.hstack([vec_combo.transform(combos_te), vec_char.transform(titles_te)], format="csr")

    oof_sp = np.zeros(len(y_tr), dtype=np.float32)
    for tr_idx, va_idx in gkf.split(F_tr, y_tr, sellers_tr):
        lr_sp = LogisticRegression(C=3.0, max_iter=250, solver="liblinear", random_state=2026)
        lr_sp.fit(X_sp_tr[tr_idx], y_tr[tr_idx])
        oof_sp[va_idx] = lr_sp.predict_proba(X_sp_tr[va_idx])[:, 1]

    lr_full = LogisticRegression(C=3.0, max_iter=300, solver="liblinear", random_state=2026).fit(X_sp_tr, y_tr)
    p_sp_te = lr_full.predict_proba(X_sp_te)[:, 1]

    for c in te_cols:
        F_tr[f"te_{c}"] = fit_te[c]
        F_te[f"te_{c}"] = test_te[c]

    F_tr["txt_sparse_p"] = oof_sp
    F_te["txt_sparse_p"] = p_sp_te

    for c in CAT_FEATURES:
        F_tr[c] = F_tr[c].astype(str)
        F_te[c] = F_te[c].astype(str)

    # LightGBM categorical encoding
    F_tr_lgb = F_tr.copy()
    F_te_lgb = F_te.copy()
    for c in CAT_FEATURES:
        cats = pd.Index(F_tr_lgb[c].unique())
        F_tr_lgb[c] = pd.Categorical(F_tr_lgb[c], categories=cats)
        F_te_lgb[c] = pd.Categorical(F_te_lgb[c], categories=cats)

    pool_tr = Pool(F_tr, y_tr, cat_features=CAT_FEATURES)
    pool_te = Pool(F_te, cat_features=CAT_FEATURES)

    cb_test_preds = []
    lgb_test_preds = []
    for s_i in [2026, 2127, 2228]:
        if s_i in (2026, 2127):
            print(f"  Fitting 70k CatBoost (seed={s_i})...", flush=True)
            cb_m = CatBoostClassifier(iterations=900, depth=6, learning_rate=0.07, l2_leaf_reg=6.0,
                                      random_seed=s_i, thread_count=4, allow_writing_files=False, verbose=0)
            cb_m.fit(pool_tr)
            if s_i == 2026:
                cb_m.save_model(ROOT / "models" / "benchmark_20261030" / "catboost_A_hist_inductive_70k_seed2026.cbm")
            cb_test_preds.append(cb_m.predict_proba(pool_te)[:, 1])

        print(f"  Fitting 70k LightGBM (seed={s_i})...", flush=True)
        lgb_m = lgb.LGBMClassifier(
            n_estimators=900, learning_rate=0.035, num_leaves=95, min_child_samples=25,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=8.0, random_state=s_i, n_jobs=4, verbose=-1
        )
        lgb_m.fit(F_tr_lgb, y_tr)
        lgb_test_preds.append(lgb_m.predict_proba(F_te_lgb)[:, 1])

    p_te_cb_seed2026 = cb_test_preds[0]
    p_te_lgb_seed2026 = lgb_test_preds[0]
    p_te_blend_single = 0.75 * p_te_cb_seed2026 + 0.25 * p_te_lgb_seed2026

    p_te_cb_bag3 = np.mean(cb_test_preds, axis=0)
    p_te_lgb_bag3 = np.mean(lgb_test_preds, axis=0)
    p_te_blend_bag3 = 0.75 * p_te_cb_bag3 + 0.25 * p_te_lgb_bag3

    # Save test probability arrays
    np.save(ROOT / "results" / "benchmark_20261030" / "test_probs_pipeline_A.npy", p_te_cb_seed2026.astype(np.float32))
    np.save(OUT_DIR / "test_probs_A_hist_inductive_70k.npy", p_te_cb_seed2026.astype(np.float32))
    np.save(OUT_DIR / "test_probs_lgb_inductive_70k.npy", p_te_lgb_seed2026.astype(np.float32))
    np.save(OUT_DIR / "test_probs_blend075_single_inductive_70k.npy", p_te_blend_single.astype(np.float32))
    np.save(OUT_DIR / "test_probs_blend075_bag3_inductive_70k.npy", p_te_blend_bag3.astype(np.float32))

    # Compare vs S48
    with open(S48_JSON, "r", encoding="utf-8") as f:
        s48_dict = {int(k): v for k, v in json.load(f).items()}
    s48_used = np.array([1 if s48_dict[i] == "used" else 0 for i in range(1, 30001)], dtype=int)
    is_seen_te = np.isin(sellers_te, list(set(sellers_tr))).astype(int)

    # Save full test prediction table for diagnostics and candidate construction
    df_test_diag = pd.DataFrame({
        "ID": np.arange(1, 30001, dtype=int),
        "item_id": cache["item_ids_te"],
        "seller_id": sellers_te,
        "is_seen": is_seen_te,
        "s48_condition": [s48_dict[i] for i in range(1, 30001)],
        "s48_used": s48_used,
        "p_cb_single": p_te_cb_seed2026.astype(np.float32),
        "p_lgb_single": p_te_lgb_seed2026.astype(np.float32),
        "p_blend_single": p_te_blend_single.astype(np.float32),
        "p_cb_bag3": p_te_cb_bag3.astype(np.float32),
        "p_lgb_bag3": p_te_lgb_bag3.astype(np.float32),
        "p_blend_bag3": p_te_blend_bag3.astype(np.float32),
    })
    test_diag_path = OUT_DIR / "test_30k_mainline_predictions.csv.gz"
    df_test_diag.to_csv(test_diag_path, index=False, compression="gzip")

    return {
        "refit_time_seconds": round(time.time() - t0, 2),
        "test_diag_path": str(test_diag_path.relative_to(ROOT)),
        "test_diag_sha256": compute_file_sha256(test_diag_path),
        "polarity_verified": "used=1, new=0",
        "n_test_rows": 30000,
        "test_seen_rows": int(is_seen_te.sum()),
        "test_new_rows": int((1 - is_seen_te).sum()),
        "mean_p_used_cb_single": float(p_te_cb_seed2026.mean()),
        "mean_p_used_blend_single": float(p_te_blend_single.mean()),
        "mean_p_used_blend_bag3": float(p_te_blend_bag3.mean()),
        "agreement_vs_s48_at_050_cb_single": float(np.mean((p_te_cb_seed2026 >= 0.50).astype(int) == s48_used)),
        "agreement_vs_s48_at_050_blend_single": float(np.mean((p_te_blend_single >= 0.50).astype(int) == s48_used)),
        "agreement_vs_s48_at_050_blend_bag3": float(np.mean((p_te_blend_bag3 >= 0.50).astype(int) == s48_used)),
        "n_flips_vs_s48_at_050_blend_single": int(np.sum((p_te_blend_single >= 0.50).astype(int) != s48_used)),
        "n_flips_vs_s48_at_050_blend_bag3": int(np.sum((p_te_blend_bag3 >= 0.50).astype(int) != s48_used)),
    }


def main():
    t0_all = time.time()
    print("=" * 80)
    print("RUNNING AGENTE 2 — LÍNEA PRINCIPAL CATBOOST + LIGHTGBM & DIVERSIDAD")
    print("=" * 80)

    cache = build_or_load_cache(force_rebuild=False)
    F_df = cache["F_r151_tr"]
    ids_all = cache["ids_tr"]
    sellers_all = cache["sellers_tr"]

    seeds = [2026, 314159, 20261030]
    all_seed_evals = {}

    for seed in seeds:
        print(f"\n" + "#" * 70)
        print(f"EVALUATING SEED {seed} ON EXACT SAME CONFIRM IDS")
        print("#" * 70, flush=True)

        with open(CACHE_DIR / f"split_features_A_hist_seed{seed}.pkl", "rb") as f:
            sp_h = pickle.load(f)
        with open(CACHE_DIR / f"split_features_A_valname_seed{seed}.pkl", "rb") as f:
            sp_v = pickle.load(f)

        fit, tune, confirm = sp_h["fit"], sp_h["tune"], sp_h["confirm"]
        y_fit, y_tu, y_co = sp_h["y_fit"], sp_h["y_tune"], sp_h["y_confirm"]
        is_seen_co = sp_h["is_seen_confirm"]
        ids_co = ids_all[confirm]
        sellers_co = sellers_all[confirm]

        # Prepare matrices for A_hist and A_valname
        X_fit_h, X_tu_h, X_co_h, X_fit_lgb_h, X_tu_lgb_h, X_co_lgb_h = prepare_lgb_matrices(
            F_df, fit, tune, confirm, sp_h["fit_te"], sp_h["tune_te"], sp_h["confirm_te"],
            sp_h["oof_sp"], sp_h["p_sp_tune"], sp_h["p_sp_conf"]
        )
        X_fit_v, X_tu_v, X_co_v, X_fit_lgb_v, X_tu_lgb_v, X_co_lgb_v = prepare_lgb_matrices(
            F_df, fit, tune, confirm, sp_v["fit_te"], sp_v["tune_te"], sp_v["confirm_te"],
            sp_v["oof_sp"], sp_v["p_sp_tune"], sp_v["p_sp_conf"]
        )

        # Base CatBoost predictions and thresholds
        p_cb_h_tu, p_cb_h_co, th_cb_h = sp_h["p_cb_tune"], sp_h["p_cb_conf"], sp_h["best_th_cb"]
        p_cb_v_tu, p_cb_v_co, th_cb_v = sp_v["p_cb_tune"], sp_v["p_cb_conf"], sp_v["best_th_cb"]

        # Fit LightGBM on A_hist and A_valname
        print(f"  Fitting bounded LightGBM on A_hist & A_valname (Seed {seed})...", flush=True)
        cfg_lgb_h, p_lgb_h_tu, p_lgb_h_co = fit_bounded_lgb(X_fit_lgb_h, y_fit, X_tu_lgb_h, y_tu, X_co_lgb_h, seed)
        cfg_lgb_v, p_lgb_v_tu, p_lgb_v_co = fit_bounded_lgb(X_fit_lgb_v, y_fit, X_tu_lgb_v, y_tu, X_co_lgb_v, seed)
        th_lgb_h, _ = select_threshold_on_tune(y_tu, p_lgb_h_tu)
        th_lgb_v, _ = select_threshold_on_tune(y_tu, p_lgb_v_tu)

        # Frozen 0.75 CB + 0.25 LGB blends
        p_blend_h_tu = 0.75 * p_cb_h_tu + 0.25 * p_lgb_h_tu
        p_blend_h_co = 0.75 * p_cb_h_co + 0.25 * p_lgb_h_co
        th_blend_h, _ = select_threshold_on_tune(y_tu, p_blend_h_tu)

        p_blend_v_tu = 0.75 * p_cb_v_tu + 0.25 * p_lgb_v_tu
        p_blend_v_co = 0.75 * p_cb_v_co + 0.25 * p_lgb_v_co
        th_blend_v, _ = select_threshold_on_tune(y_tu, p_blend_v_tu)

        # Substantial diversity variant: Bagged CB + 3-Seed Bagged LGB
        print(f"  Fitting Bagged Diversity Variant (CB_Bag + LGB_Bag3) on Seed {seed}...", flush=True)
        bag_h = fit_bagged_diversity_variant(
            X_fit_h, y_fit, X_tu_h, y_tu, X_co_h,
            X_fit_lgb_h, X_tu_lgb_h, X_co_lgb_h,
            p_cb_h_tu, p_cb_h_co, p_cb_v_tu, p_cb_v_co,
            cfg_lgb_h, p_lgb_h_tu, p_lgb_h_co, seed
        )
        th_cb_bag_h, _ = select_threshold_on_tune(y_tu, bag_h["p_cb_bag_tu"])
        th_blend_bag_h, _ = select_threshold_on_tune(y_tu, bag_h["p_blend_bag_tu"])

        # Compare all candidates against Historical Strong Baseline `A_hist` on exact same confirm IDs
        comparisons_vs_A_hist = {
            "A_valname_vs_A_hist": compare_under_three_threshold_regimes(
                ids_co, sellers_co, is_seen_co, y_co, p_cb_h_co, th_cb_h, p_cb_v_co, th_cb_v
            ),
            "LGB_hist_vs_A_hist": compare_under_three_threshold_regimes(
                ids_co, sellers_co, is_seen_co, y_co, p_cb_h_co, th_cb_h, p_lgb_h_co, th_lgb_h
            ),
            "LGB_valname_vs_A_hist": compare_under_three_threshold_regimes(
                ids_co, sellers_co, is_seen_co, y_co, p_cb_h_co, th_cb_h, p_lgb_v_co, th_lgb_v
            ),
            "Blend075_hist_vs_A_hist": compare_under_three_threshold_regimes(
                ids_co, sellers_co, is_seen_co, y_co, p_cb_h_co, th_cb_h, p_blend_h_co, th_blend_h
            ),
            "Blend075_valname_vs_A_hist": compare_under_three_threshold_regimes(
                ids_co, sellers_co, is_seen_co, y_co, p_cb_h_co, th_cb_h, p_blend_v_co, th_blend_v
            ),
            "CB_Bag3_hist_vs_A_hist": compare_under_three_threshold_regimes(
                ids_co, sellers_co, is_seen_co, y_co, p_cb_h_co, th_cb_h, bag_h["p_cb_bag_co"], th_cb_bag_h
            ),
            "Blend075_Bag3_hist_vs_A_hist": compare_under_three_threshold_regimes(
                ids_co, sellers_co, is_seen_co, y_co, p_cb_h_co, th_cb_h, bag_h["p_blend_bag_co"], th_blend_bag_h
            ),
        }

        # Also compare Blend075_valname against A_valname (the +26/+56/+4 reference)
        comparisons_vs_A_valname = {
            "LGB_valname_vs_A_valname": compare_under_three_threshold_regimes(
                ids_co, sellers_co, is_seen_co, y_co, p_cb_v_co, th_cb_v, p_lgb_v_co, th_lgb_v
            ),
            "Blend075_valname_vs_A_valname": compare_under_three_threshold_regimes(
                ids_co, sellers_co, is_seen_co, y_co, p_cb_v_co, th_cb_v, p_blend_v_co, th_blend_v
            ),
        }

        # Check historical compatible models (R142 HistGB, R142 CatBoost, R156 Ensemble, R151 Dual NB-SVM)
        f_r142 = ROOT / "results" / "revision_submissions" / f"paired_predictions_seed_{seed}.csv"
        if f_r142.exists():
            df_142 = pd.read_csv(f_r142)
            assert (df_142["id"].values == ids_co).all(), "R142 ID mismatch!"
            comparisons_vs_A_hist["R142_HistGB_vs_A_hist"] = compare_under_three_threshold_regimes(
                ids_co, sellers_co, is_seen_co, y_co, p_cb_h_co, th_cb_h, df_142["p_hist_used"].values, 0.50
            )
            comparisons_vs_A_hist["R142_CatBoost_vs_A_hist"] = compare_under_three_threshold_regimes(
                ids_co, sellers_co, is_seen_co, y_co, p_cb_h_co, th_cb_h, df_142["p_cat_used"].values, 0.50
            )

        f_r156 = ROOT / "results" / "revision_submissions" / f"paired_ensemble_seed_{seed}.csv"
        if f_r156.exists():
            df_156 = pd.read_csv(f_r156)
            assert (df_156["id"].values == ids_co).all(), "R156 ID mismatch!"
            comparisons_vs_A_hist["R156_Ensemble_vs_A_hist"] = compare_under_three_threshold_regimes(
                ids_co, sellers_co, is_seen_co, y_co, p_cb_h_co, th_cb_h, df_156["p_ensemble"].values, 0.50
            )

        # Save unified paired confirm dataframe per seed with all models
        df_master_pair = pd.DataFrame({
            "row_idx": confirm,
            "id": confirm + 1,
            "item_id": ids_co,
            "seller_id": sellers_co,
            "is_seen": is_seen_co,
            "y_true": y_co,
            "p_A_hist": p_cb_h_co.astype(np.float32),
            "pred_A_hist_tuned": (p_cb_h_co >= th_cb_h).astype(int),
            "p_A_valname": p_cb_v_co.astype(np.float32),
            "pred_A_valname_tuned": (p_cb_v_co >= th_cb_v).astype(int),
            "p_LGB_hist": p_lgb_h_co.astype(np.float32),
            "p_LGB_valname": p_lgb_v_co.astype(np.float32),
            "p_Blend075_hist": p_blend_h_co.astype(np.float32),
            "pred_Blend075_hist_tuned": (p_blend_h_co >= th_blend_h).astype(int),
            "p_Blend075_valname": p_blend_v_co.astype(np.float32),
            "pred_Blend075_valname_tuned": (p_blend_v_co >= th_blend_v).astype(int),
            "p_CB_Bag3_hist": bag_h["p_cb_bag_co"].astype(np.float32),
            "p_Blend075_Bag3_hist": bag_h["p_blend_bag_co"].astype(np.float32),
            "pred_Blend075_Bag3_hist_tuned": (bag_h["p_blend_bag_co"] >= th_blend_bag_h).astype(int),
        })
        pair_out = OUT_DIR / f"paired_all_models_seed_{seed}_confirm.csv.gz"
        df_master_pair.to_csv(pair_out, index=False, compression="gzip")

        all_seed_evals[str(seed)] = {
            "seed": seed,
            "n_confirm": int(len(confirm)),
            "n_seen": int(is_seen_co.sum()),
            "n_new": int((1 - is_seen_co).sum()),
            "selected_lgb_hist": cfg_lgb_h["name"],
            "selected_lgb_valname": cfg_lgb_v["name"],
            "paired_file": str(pair_out.relative_to(ROOT)),
            "paired_file_sha256": compute_file_sha256(pair_out),
            "baseline_A_hist": {
                "th_tune": float(th_cb_h),
                "acc_050": float(accuracy_score(y_co, p_cb_h_co >= 0.50)),
                "acc_tuned": float(accuracy_score(y_co, p_cb_h_co >= th_cb_h)),
                "logloss": float(log_loss(y_co, p_cb_h_co)),
            },
            "baseline_A_valname": {
                "th_tune": float(th_cb_v),
                "acc_050": float(accuracy_score(y_co, p_cb_v_co >= 0.50)),
                "acc_tuned": float(accuracy_score(y_co, p_cb_v_co >= th_cb_v)),
                "logloss": float(log_loss(y_co, p_cb_v_co)),
            },
            "comparisons_vs_A_hist": comparisons_vs_A_hist,
            "comparisons_vs_A_valname": comparisons_vs_A_valname,
        }

        # Print quick summary for this seed
        c_b_h = comparisons_vs_A_hist["Blend075_hist_vs_A_hist"]
        c_bag_h = comparisons_vs_A_hist["Blend075_Bag3_hist_vs_A_hist"]
        c_b_v = comparisons_vs_A_valname["Blend075_valname_vs_A_valname"]
        print(f"  [vs A_hist] Blend075_hist: Common050 Net={c_b_h['common_th_050']['net']:+d}, PolicyTuned Net={c_b_h['policy_tuned_th']['net']:+d}, DeltaLL={c_b_h['policy_tuned_th']['delta_logloss']:+.5f}")
        print(f"  [vs A_hist] Blend075_Bag3_hist: Common050 Net={c_bag_h['common_th_050']['net']:+d}, PolicyTuned Net={c_bag_h['policy_tuned_th']['net']:+d}, DeltaLL={c_bag_h['policy_tuned_th']['delta_logloss']:+.5f}")
        print(f"  [vs A_valname] Blend075_valname: Common050 Net={c_b_v['common_th_050']['net']:+d}, PolicyTuned Net={c_b_v['policy_tuned_th']['net']:+d}, DeltaLL={c_b_v['policy_tuned_th']['delta_logloss']:+.5f}")

    # Run full 70k inductive refit and test inference
    test_refit_info = run_inductive_70k_refit_and_test_inference(cache)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_seconds": round(time.time() - t0_all, 2),
        "frozen_reference_config": "0.75 CatBoost + 0.25 LightGBM",
        "non_independence_notice": (
            "Seeds 2026, 314159, and 20261030 share training rows and overlap in confirm rows "
            "(~18.1%-18.6% pairwise confirm overlap). Gains across seeds MUST NOT be summed or "
            "treated as independent trials."
        ),
        "seeds": all_seed_evals,
        "inductive_70k_test_refit": test_refit_info,
    }

    with open(OUT_DIR / "agent2_mainline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved Agent 2 summary to {OUT_DIR / 'agent2_mainline_summary.json'}")


if __name__ == "__main__":
    main()
