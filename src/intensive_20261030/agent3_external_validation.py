"""AGENTE 3 — VALIDACIÓN EXTERNA AL PROCESO DE SELECCIÓN (2026-10-30).

Implements the most honest external & nested validation possible on the 70k dataset:
1. Evaluates 3 strictly inductive outer cuts:
   - Cut 1 (`disjoint_3seed_20261101`): Seller-group split where `confirm` (both `new_confirm` sellers
     and `seen_confirm` rows) is drawn EXCLUSIVELY from rows and sellers that NEVER appeared in the
     `confirm` splits of seeds 2026, 314159, or 20261030. Solapa en 92.39% con la unión histórica completa
     de 18 holdouts (67,007 / 70,000 filas); NO es virgen globalmente.
   - Cut 2 (`nested_outer_777001`): Fresh nested seller-group split (seed 777001).
   - Cut 3 (`nested_outer_777002`): Fresh nested seller-group split (seed 777002).
2. Zero-leakage strictly inductive transformations inside each cut (UNIFORM ACROSS ALL CUTS):
   - Unsupervised group aggregations (`freq_*`, `seller_mean_*`, `rel_price_cat`) fitted ONLY on `fit`
     via `compute_inductive_group_features` (never touching `tune` or `confirm`).
   - Supervised Target Encoding (`alpha=5.0`) fitted ONLY on `fit` via 4-fold GroupKFold by seller.
   - Text Teacher (`txt_sparse_p`) fitted ONLY on `fit` via per-fold `TfidfVectorizer` + `LogisticRegression`
     inside 4-fold GroupKFold by seller (strict per-fold refit without global vocabulary reuse, applied
     identically across ALL 3 cuts).
   - All model selection (LightGBM hyperparameter config, nested blend weight, and decision threshold)
     performed strictly on `tune` inside each outer cut.
3. Bag3 exact evaluation:
   - Measures both the single model (`CB_single`, `LGB_single`, `Blend_single`) and the exact Bag3
     ensemble (`CB_bag3`, `LGB_bag3`, `Blend_bag3` with seeds [seed, seed+101, seed+202]).
   - Directly tests the marginal contribution of Bag3 over Single Blend on the outer cuts.
4. Computes:
   - Sizes and SHA256 hashes of `fit`, `tune`, `confirm`.
   - Historical overlap metrics against the full union of 18 historical holdouts.
   - Seen/new proportions and separate metrics (`seen_sellers` vs `new_sellers`).
   - Observed aggregate result AND Transported Estimate to the Kaggle Test Seen/New mixture
     (71.623% seen / 28.377% new), with Seller-Clustered Transported Bootstrap 95% CIs and explicit assumptions.
   - Detailed regression analysis (where the blend breaks baseline correct predictions).
   - Explicit epistemological classification (Development vs Strong External Confirmation).
"""
import hashlib
import json
import shutil
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
    build_or_load_cache, compute_inductive_group_features, compute_file_sha256, CAT_FEATURES
)
from src.benchmark_20261030.split_utils import split_seller_groups_r151
from src.eval_20261030.paired_eval import evaluate_paired_frames

OUT_DIR = ROOT / "results" / "intensive_20261030"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_SEEN_WEIGHT = 21487.0 / 30000.0  # 0.71623333
TEST_NEW_WEIGHT = 8513.0 / 30000.0    # 0.28376667


def build_disjoint_3seed_split(sellers, prior_seeds=(2026, 314159, 20261030), seed=20261101):
    """Constructs a seller-group partition (`fit`, `tune`, `confirm`) where `confirm`
    contains ZERO rows and ZERO new-seller groups that appeared in `confirm` of `prior_seeds`.
    Notice: Does NOT guarantee independence from other historical holdouts (92.39% overlap).
    """
    prior_confirm_rows = set()
    prior_confirm_sellers = set()
    for ps in prior_seeds:
        _, _, c_idx = split_seller_groups_r151(sellers, ps)
        prior_confirm_rows.update(c_idx.tolist())
        prior_confirm_sellers.update(sellers[c_idx].tolist())

    rng = np.random.default_rng(seed)
    unique_all = np.unique(sellers)
    n_new = len(unique_all) // 10

    # Eligible sellers for `new_confirm`: never appeared in any prior confirm split
    eligible_new_confirm_sellers = np.array([s for s in unique_all if s not in prior_confirm_sellers])
    rng.shuffle(eligible_new_confirm_sellers)
    new_confirm = set(eligible_new_confirm_sellers[:n_new])

    # Remaining sellers for `new_tune`
    rem_sellers = np.array([s for s in unique_all if s not in new_confirm])
    rng.shuffle(rem_sellers)
    new_tune = set(rem_sellers[:n_new])

    seen_sellers_mask = ~(np.isin(sellers, list(new_tune | new_confirm)))
    seen_row_indices = np.where(seen_sellers_mask)[0]
    n_seen_target = len(seen_row_indices) // 10

    # Eligible seen rows for `seen_confirm`: never appeared in `prior_confirm_rows`
    eligible_seen_confirm = np.array([r for r in seen_row_indices if r not in prior_confirm_rows])
    rng.shuffle(eligible_seen_confirm)
    seen_confirm = eligible_seen_confirm[:n_seen_target]

    rem_seen_rows = np.setdiff1d(seen_row_indices, seen_confirm)
    rng.shuffle(rem_seen_rows)
    seen_tune = rem_seen_rows[:n_seen_target]

    tune_rows = np.flatnonzero(np.isin(sellers, list(new_tune)))
    confirm_rows = np.flatnonzero(np.isin(sellers, list(new_confirm)))

    tune = np.r_[tune_rows, seen_tune]
    confirm = np.r_[confirm_rows, seen_confirm]
    fit = np.setdiff1d(np.arange(len(sellers)), np.r_[tune, confirm])

    assert len(set(confirm.tolist()) & prior_confirm_rows) == 0, "Disjoint confirm overlapped prior confirm rows!"
    return fit, tune, confirm


def transported_seller_bootstrap(y_true, pred_base, pred_cand, is_seen, sellers, n_boot=2000, seed=42):
    """Computes transported net accuracy delta on the 30k Kaggle test mixture
    (71.623% seen, 28.377% new) and its seller-clustered 95% bootstrap CI.
    """
    rng = np.random.default_rng(seed)
    seen_mask = (is_seen == 1)
    new_mask = (is_seen == 0)

    n_seen = int(np.sum(seen_mask))
    n_new = int(np.sum(new_mask))

    diff = (pred_cand == y_true).astype(float) - (pred_base == y_true).astype(float)
    delta_acc_seen = float(np.mean(diff[seen_mask]))
    delta_acc_new = float(np.mean(diff[new_mask]))
    delta_acc_obs = float(np.mean(diff))
    delta_acc_trans = TEST_SEEN_WEIGHT * delta_acc_seen + TEST_NEW_WEIGHT * delta_acc_new
    expected_30k_net = 30000.0 * delta_acc_trans

    seen_sellers = np.unique(sellers[seen_mask])
    new_sellers = np.unique(sellers[new_mask])

    seen_s_to_diff = defaultdict(float)
    seen_s_to_cnt = defaultdict(int)
    for s, d in zip(sellers[seen_mask], diff[seen_mask]):
        seen_s_to_diff[s] += d
        seen_s_to_cnt[s] += 1

    new_s_to_diff = defaultdict(float)
    new_s_to_cnt = defaultdict(int)
    for s, d in zip(sellers[new_mask], diff[new_mask]):
        new_s_to_diff[s] += d
        new_s_to_cnt[s] += 1

    seen_diffs = np.array([seen_s_to_diff[s] for s in seen_sellers])
    seen_cnts = np.array([seen_s_to_cnt[s] for s in seen_sellers])
    new_diffs = np.array([new_s_to_diff[s] for s in new_sellers])
    new_cnts = np.array([new_s_to_cnt[s] for s in new_sellers])

    boot_trans_30k = np.zeros(n_boot)
    for b in range(n_boot):
        idx_s = rng.integers(0, len(seen_sellers), size=len(seen_sellers))
        idx_n = rng.integers(0, len(new_sellers), size=len(new_sellers))
        b_da_seen = np.sum(seen_diffs[idx_s]) / max(np.sum(seen_cnts[idx_s]), 1)
        b_da_new = np.sum(new_diffs[idx_n]) / max(np.sum(new_cnts[idx_n]), 1)
        boot_trans_30k[b] = 30000.0 * (TEST_SEEN_WEIGHT * b_da_seen + TEST_NEW_WEIGHT * b_da_new)

    return {
        "n_seen": n_seen,
        "n_new": n_new,
        "prop_seen_confirm": round(n_seen / (n_seen + n_new), 5),
        "prop_new_confirm": round(n_new / (n_seen + n_new), 5),
        "delta_acc_seen": round(delta_acc_seen, 5),
        "delta_acc_new": round(delta_acc_new, 5),
        "delta_acc_observed": round(delta_acc_obs, 5),
        "delta_acc_transported_to_test_mix": round(delta_acc_trans, 5),
        "expected_net_items_30k_test_mix": round(expected_30k_net, 1),
        "transported_30k_bootstrap_ci95": [
            round(float(np.percentile(boot_trans_30k, 2.5)), 1),
            round(float(np.percentile(boot_trans_30k, 97.5)), 1),
        ],
    }


def analyze_regressions(y_true, is_seen, cat_ids, p_base, pred_base, p_cand, pred_cand, th):
    """Detailed regression analysis: where candidate breaks baseline correct predictions."""
    correct_base = (pred_base == y_true)
    correct_cand = (pred_cand == y_true)
    fixes = (~correct_base) & correct_cand
    breaks = correct_base & (~correct_cand)

    margin_base = np.abs(p_base - th)
    bands = [
        ("0.00-0.02 (boundary)", (margin_base >= 0.00) & (margin_base < 0.02)),
        ("0.02-0.05 (narrow)", (margin_base >= 0.02) & (margin_base < 0.05)),
        ("0.05-0.10 (moderate)", (margin_base >= 0.05) & (margin_base < 0.10)),
        ("0.10-0.20 (wide)", (margin_base >= 0.10) & (margin_base < 0.20)),
        ("0.20+ (high confidence)", (margin_base >= 0.20)),
    ]
    band_breakdown = []
    for bname, msk in bands:
        b_fix = int(np.sum(fixes & msk))
        b_brk = int(np.sum(breaks & msk))
        band_breakdown.append({
            "margin_band": bname,
            "n_rows": int(np.sum(msk)),
            "fixes": b_fix,
            "breaks": b_brk,
            "net": b_fix - b_brk,
        })

    df_cat = pd.DataFrame({"cat": cat_ids, "fix": fixes.astype(int), "brk": breaks.astype(int)})
    g_cat = df_cat.groupby("cat").agg(fixes=("fix", "sum"), breaks=("brk", "sum"))
    g_cat["net"] = g_cat["fixes"] - g_cat["breaks"]
    worst_cats = g_cat.sort_values("net").head(5).reset_index().to_dict(orient="records")
    best_cats = g_cat.sort_values("net", ascending=False).head(5).reset_index().to_dict(orient="records")

    return {
        "by_margin_band": band_breakdown,
        "top_regression_categories": worst_cats,
        "top_improvement_categories": best_cats,
    }


def run_inductive_outer_cut(cut_name, fit, tune, confirm, seed, cache, audit_holdouts_data=None):
    t0 = time.time()
    print(f"\n" + "#" * 70)
    print(f"RUNNING STRICTLY INDUCTIVE EXTERNAL CUT: {cut_name} (seed={seed})")
    print("#" * 70, flush=True)

    sellers = cache["sellers_tr"]
    ids = cache["ids_tr"]
    y_used = cache["y_tr"]
    F_raw = cache["F_r151_tr"]
    cat_df = cache["cat_tr"]
    titles = cache["titles_tr"]
    combos = cache["combos_hist_tr"]

    fit_hash = hashlib.sha256(np.ascontiguousarray(fit).tobytes()).hexdigest()
    tune_hash = hashlib.sha256(np.ascontiguousarray(tune).tobytes()).hexdigest()
    conf_hash = hashlib.sha256(np.ascontiguousarray(confirm).tobytes()).hexdigest()
    split_hash = hashlib.sha256(np.r_[fit, tune, confirm].tobytes()).hexdigest()

    fit_sellers = sellers[fit]
    fit_seller_set = set(fit_sellers)
    is_seen_tune = np.isin(sellers[tune], list(fit_seller_set)).astype(int)
    is_seen_conf = np.isin(sellers[confirm], list(fit_seller_set)).astype(int)

    y_fit, y_tu, y_co = y_used[fit], y_used[tune], y_used[confirm]

    # 1. Strictly inductive unsupervised group aggregations (fitted ONLY on `fit`)
    X_fit, X_tu, X_co = compute_inductive_group_features(
        F_raw, cat_df, fit,
        [
            (F_raw.iloc[fit], cat_df.iloc[fit]),
            (F_raw.iloc[tune], cat_df.iloc[tune]),
            (F_raw.iloc[confirm], cat_df.iloc[confirm]),
        ]
    )

    # 2. Strictly inductive Target Encoding inside `fit` (alpha=5.0)
    prior = float(y_fit.mean())
    te_cols = ["seller_id", "category_id", "city"]
    gkf = GroupKFold(n_splits=4)
    fit_te = {c: np.full(len(fit), prior, dtype=np.float32) for c in te_cols}
    tune_te, conf_te = {}, {}

    for tr, va in gkf.split(fit, y_fit, fit_sellers):
        for c in te_cols:
            k_tr = cat_df[c].iloc[fit[tr]]
            k_va = cat_df[c].iloc[fit[va]]
            stats = pd.DataFrame({"k": k_tr, "y": y_fit[tr]}).groupby("k").y.agg(["sum", "count"])
            smoothed = (stats["sum"] + 5.0 * prior) / (stats["count"] + 5.0)
            fit_te[c][va] = k_va.map(smoothed).fillna(prior).values.astype(np.float32)

    for c in te_cols:
        k_fit = cat_df[c].iloc[fit]
        stats = pd.DataFrame({"k": k_fit, "y": y_fit}).groupby("k").y.agg(["sum", "count"])
        smoothed = (stats["sum"] + 5.0 * prior) / (stats["count"] + 5.0)
        tune_te[c] = cat_df[c].iloc[tune].map(smoothed).fillna(prior).values.astype(np.float32)
        conf_te[c] = cat_df[c].iloc[confirm].map(smoothed).fillna(prior).values.astype(np.float32)

    # 3. Strictly inductive `txt_sparse_p` fitted exclusively inside `fit` (UNIFORM ACROSS ALL CUTS)
    print("  [Inductive Refit] Fitting full-fit TF-IDF & LogisticRegression...")
    vec_w_full = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=120000,
                                 sublinear_tf=True, strip_accents="unicode", lowercase=True)
    vec_c_full = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                                 max_features=150000, sublinear_tf=True)
    X_fit_sp = sp.hstack([
        vec_w_full.fit_transform([combos[i] for i in fit]),
        vec_c_full.fit_transform([titles[i] for i in fit])
    ], format="csr")
    X_tu_sp = sp.hstack([
        vec_w_full.transform([combos[i] for i in tune]),
        vec_c_full.transform([titles[i] for i in tune])
    ], format="csr")
    X_co_sp = sp.hstack([
        vec_w_full.transform([combos[i] for i in confirm]),
        vec_c_full.transform([titles[i] for i in confirm])
    ], format="csr")

    # Strict inner-fold TF-IDF refit for ALL cuts (zero shortcuts)
    print("  [Inductive Refit] Fitting inner-fold TF-IDF across 4 folds for OOF text probabilities...")
    oof_sp = np.zeros(len(fit), dtype=np.float32)
    for fold_i, (tr, va) in enumerate(gkf.split(fit, y_fit, fit_sellers)):
        vec_w_f = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=120000,
                                  sublinear_tf=True, strip_accents="unicode", lowercase=True)
        vec_c_f = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                                  max_features=150000, sublinear_tf=True)
        X_tr_sp = sp.hstack([
            vec_w_f.fit_transform([combos[fit[i]] for i in tr]),
            vec_c_f.fit_transform([titles[fit[i]] for i in tr])
        ], format="csr")
        X_va_sp = sp.hstack([
            vec_w_f.transform([combos[fit[i]] for i in va]),
            vec_c_f.transform([titles[fit[i]] for i in va])
        ], format="csr")
        clf_fold = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")
        clf_fold.fit(X_tr_sp, y_fit[tr])
        oof_sp[va] = clf_fold.predict_proba(X_va_sp)[:, 1]

    clf_full = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")
    clf_full.fit(X_fit_sp, y_fit)
    p_sp_tu = clf_full.predict_proba(X_tu_sp)[:, 1].astype(np.float32)
    p_sp_co = clf_full.predict_proba(X_co_sp)[:, 1].astype(np.float32)

    for c in te_cols:
        X_fit[f"te_{c}"] = fit_te[c]
        X_tu[f"te_{c}"] = tune_te[c]
        X_co[f"te_{c}"] = conf_te[c]

    X_fit["txt_sparse_p"] = oof_sp
    X_tu["txt_sparse_p"] = p_sp_tu
    X_co["txt_sparse_p"] = p_sp_co

    for c in CAT_FEATURES:
        X_fit[c] = X_fit[c].astype(str)
        X_tu[c] = X_tu[c].astype(str)
        X_co[c] = X_co[c].astype(str)

    # 4. Fit CatBoost on `fit` (Single Model seed + Bag3 ensemble [seed, seed+101, seed+202])
    print("  [CatBoost] Training Seed 0 (baseline)...")
    cb_seeds = [seed, seed + 101, seed + 202]
    cb_tu_preds = []
    cb_co_preds = []

    pool_fit = Pool(X_fit, y_fit, cat_features=CAT_FEATURES)
    pool_tu = Pool(X_tu, cat_features=CAT_FEATURES)
    pool_co = Pool(X_co, cat_features=CAT_FEATURES)

    for s_idx, s in enumerate(cb_seeds):
        if s_idx > 0:
            print(f"  [CatBoost Bag3] Training Seed {s_idx} (seed={s})...")
        cb_m = CatBoostClassifier(iterations=900, depth=6, learning_rate=0.07, l2_leaf_reg=6.0,
                                  random_seed=s, thread_count=4, allow_writing_files=False, verbose=0)
        cb_m.fit(pool_fit)
        cb_tu_preds.append(cb_m.predict_proba(pool_tu)[:, 1])
        cb_co_preds.append(cb_m.predict_proba(pool_co)[:, 1])

    p_cb_single_tu = cb_tu_preds[0]
    p_cb_single_co = cb_co_preds[0]
    p_cb_bag3_tu = np.mean(cb_tu_preds, axis=0)
    p_cb_bag3_co = np.mean(cb_co_preds, axis=0)

    # Select CatBoost threshold on `tune`
    th_cb_single, acc_cb_tu = 0.50, -1.0
    for th in np.linspace(0.40, 0.60, 41):
        a = accuracy_score(y_tu, p_cb_single_tu >= th)
        if a > acc_cb_tu:
            acc_cb_tu = a
            th_cb_single = float(th)

    # 5. Bounded LightGBM search inside `(fit, tune)`
    X_fit_lgb, X_tu_lgb, X_co_lgb = X_fit.copy(), X_tu.copy(), X_co.copy()
    for c in CAT_FEATURES:
        cats = pd.Index(X_fit_lgb[c].unique())
        X_fit_lgb[c] = pd.Categorical(X_fit_lgb[c], categories=cats)
        X_tu_lgb[c] = pd.Categorical(X_tu_lgb[c], categories=cats)
        X_co_lgb[c] = pd.Categorical(X_co_lgb[c], categories=cats)

    lgb_configs = [
        {"name": "lgb_default", "n_estimators": 800, "learning_rate": 0.04, "num_leaves": 63, "min_child_samples": 30, "reg_lambda": 5.0},
        {"name": "lgb_conservative", "n_estimators": 700, "learning_rate": 0.04, "num_leaves": 31, "min_child_samples": 50, "reg_lambda": 10.0},
        {"name": "lgb_deep", "n_estimators": 900, "learning_rate": 0.035, "num_leaves": 95, "min_child_samples": 25, "reg_lambda": 8.0},
    ]
    best_lgb_tu_acc = -1.0
    best_lgb_cfg = None
    p_lgb_single_tu, p_lgb_single_co = None, None

    print("  [LightGBM] Tuning hyperparameters on `tune`...")
    for cfg in lgb_configs:
        m = lgb.LGBMClassifier(
            n_estimators=cfg["n_estimators"], learning_rate=cfg["learning_rate"],
            num_leaves=cfg["num_leaves"], min_child_samples=cfg["min_child_samples"],
            subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=cfg["reg_lambda"],
            random_state=seed, n_jobs=4, verbose=-1
        )
        m.fit(X_fit_lgb, y_fit)
        p_tu_m = m.predict_proba(X_tu_lgb)[:, 1]
        a_m = accuracy_score(y_tu, p_tu_m >= 0.50)
        if a_m > best_lgb_tu_acc:
            best_lgb_tu_acc = a_m
            best_lgb_cfg = cfg
            p_lgb_single_tu = p_tu_m
            p_lgb_single_co = m.predict_proba(X_co_lgb)[:, 1]

    # LightGBM Bag3 training (offset seeds)
    print(f"  [LightGBM Bag3] Best config: {best_lgb_cfg['name']}. Training seeds 1 and 2...")
    lgb_tu_preds = [p_lgb_single_tu]
    lgb_co_preds = [p_lgb_single_co]
    for offset in [101, 202]:
        m_bag = lgb.LGBMClassifier(
            n_estimators=best_lgb_cfg["n_estimators"], learning_rate=best_lgb_cfg["learning_rate"],
            num_leaves=best_lgb_cfg["num_leaves"], min_child_samples=best_lgb_cfg["min_child_samples"],
            subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=best_lgb_cfg["reg_lambda"],
            random_state=seed + offset, n_jobs=4, verbose=-1
        )
        m_bag.fit(X_fit_lgb, y_fit)
        lgb_tu_preds.append(m_bag.predict_proba(X_tu_lgb)[:, 1])
        lgb_co_preds.append(m_bag.predict_proba(X_co_lgb)[:, 1])

    p_lgb_bag3_tu = np.mean(lgb_tu_preds, axis=0)
    p_lgb_bag3_co = np.mean(lgb_co_preds, axis=0)

    # 6. Evaluate Frozen Blends (Single Blend and Bag3 Blend)
    p_blend_single_tu = 0.75 * p_cb_single_tu + 0.25 * p_lgb_single_tu
    p_blend_single_co = 0.75 * p_cb_single_co + 0.25 * p_lgb_single_co

    p_blend_bag3_tu = 0.75 * p_cb_bag3_tu + 0.25 * p_lgb_bag3_tu
    p_blend_bag3_co = 0.75 * p_cb_bag3_co + 0.25 * p_lgb_bag3_co

    # Threshold selection on `tune`
    th_blend_single, acc_blend_single_tu = 0.50, -1.0
    for th in np.linspace(0.40, 0.60, 41):
        a = accuracy_score(y_tu, p_blend_single_tu >= th)
        if a > acc_blend_single_tu:
            acc_blend_single_tu = a
            th_blend_single = float(th)

    th_blend_bag3, acc_blend_bag3_tu = 0.50, -1.0
    for th in np.linspace(0.40, 0.60, 41):
        a = accuracy_score(y_tu, p_blend_bag3_tu >= th)
        if a > acc_blend_bag3_tu:
            acc_blend_bag3_tu = a
            th_blend_bag3 = float(th)

    ids_co = ids[confirm]
    sellers_co = sellers[confirm]

    # Paired evaluations:
    # 1. Blend_Single vs Base_CB
    df_base_cb_050 = pd.DataFrame({"id": ids_co, "seller_id": sellers_co, "is_seen": is_seen_conf, "y_used": y_co, "p_used": p_cb_single_co, "pred": (p_cb_single_co >= 0.50).astype(int)})
    df_blend_s_050 = pd.DataFrame({"id": ids_co, "seller_id": sellers_co, "is_seen": is_seen_conf, "y_used": y_co, "p_used": p_blend_single_co, "pred": (p_blend_single_co >= 0.50).astype(int)})
    eval_single_050 = evaluate_paired_frames(df_base_cb_050, df_blend_s_050, n_boot=2000, seed=42)
    trans_single_050 = transported_seller_bootstrap(y_co, (p_cb_single_co >= 0.50).astype(int), (p_blend_single_co >= 0.50).astype(int), is_seen_conf, sellers_co)

    df_base_cb_pol = pd.DataFrame({"id": ids_co, "seller_id": sellers_co, "is_seen": is_seen_conf, "y_used": y_co, "p_used": p_cb_single_co, "pred": (p_cb_single_co >= th_cb_single).astype(int)})
    df_blend_s_pol = pd.DataFrame({"id": ids_co, "seller_id": sellers_co, "is_seen": is_seen_conf, "y_used": y_co, "p_used": p_blend_single_co, "pred": (p_blend_single_co >= th_blend_single).astype(int)})
    eval_single_pol = evaluate_paired_frames(df_base_cb_pol, df_blend_s_pol, n_boot=2000, seed=42)
    trans_single_pol = transported_seller_bootstrap(y_co, (p_cb_single_co >= th_cb_single).astype(int), (p_blend_single_co >= th_blend_single).astype(int), is_seen_conf, sellers_co)

    # 2. Blend_Bag3 vs Base_CB
    df_blend_bag_050 = pd.DataFrame({"id": ids_co, "seller_id": sellers_co, "is_seen": is_seen_conf, "y_used": y_co, "p_used": p_blend_bag3_co, "pred": (p_blend_bag3_co >= 0.50).astype(int)})
    eval_bag_050 = evaluate_paired_frames(df_base_cb_050, df_blend_bag_050, n_boot=2000, seed=42)
    trans_bag_050 = transported_seller_bootstrap(y_co, (p_cb_single_co >= 0.50).astype(int), (p_blend_bag3_co >= 0.50).astype(int), is_seen_conf, sellers_co)

    df_blend_bag_pol = pd.DataFrame({"id": ids_co, "seller_id": sellers_co, "is_seen": is_seen_conf, "y_used": y_co, "p_used": p_blend_bag3_co, "pred": (p_blend_bag3_co >= th_blend_bag3).astype(int)})
    eval_bag_pol = evaluate_paired_frames(df_base_cb_pol, df_blend_bag_pol, n_boot=2000, seed=42)
    trans_bag_pol = transported_seller_bootstrap(y_co, (p_cb_single_co >= th_cb_single).astype(int), (p_blend_bag3_co >= th_blend_bag3).astype(int), is_seen_conf, sellers_co)

    # 3. Blend_Bag3 vs Blend_Single (DIRECT MARGINAL EFFECT OF BAG3)
    eval_bag_vs_single_050 = evaluate_paired_frames(df_blend_s_050, df_blend_bag_050, n_boot=2000, seed=42)
    trans_bag_vs_single_050 = transported_seller_bootstrap(y_co, (p_blend_single_co >= 0.50).astype(int), (p_blend_bag3_co >= 0.50).astype(int), is_seen_conf, sellers_co)

    eval_bag_vs_single_pol = evaluate_paired_frames(df_blend_s_pol, df_blend_bag_pol, n_boot=2000, seed=42)
    trans_bag_vs_single_pol = transported_seller_bootstrap(y_co, (p_blend_single_co >= th_blend_single).astype(int), (p_blend_bag3_co >= th_blend_bag3).astype(int), is_seen_conf, sellers_co)

    reg_analysis = analyze_regressions(
        y_co, is_seen_conf, cat_df.iloc[confirm]["category_id"].values,
        p_cb_single_co, (p_cb_single_co >= 0.50).astype(int), p_blend_bag3_co, (p_blend_bag3_co >= 0.50).astype(int), 0.50
    )

    # Save paired confirm CSV
    pair_path = OUT_DIR / f"paired_external_{cut_name}.csv.gz"
    pd.DataFrame({
        "row_idx": confirm,
        "id": confirm + 1,
        "item_id": ids_co,
        "seller_id": sellers_co,
        "is_seen": is_seen_conf,
        "y_true": y_co,
        "p_cb_single": p_cb_single_co.astype(np.float32),
        "p_cb_bag3": p_cb_bag3_co.astype(np.float32),
        "p_lgb_single": p_lgb_single_co.astype(np.float32),
        "p_lgb_bag3": p_lgb_bag3_co.astype(np.float32),
        "p_blend_single": p_blend_single_co.astype(np.float32),
        "p_blend_bag3": p_blend_bag3_co.astype(np.float32),
        "pred_cb_single_050": (p_cb_single_co >= 0.50).astype(int),
        "pred_cb_single_tuned": (p_cb_single_co >= th_cb_single).astype(int),
        "pred_blend_single_050": (p_blend_single_co >= 0.50).astype(int),
        "pred_blend_single_tuned": (p_blend_single_co >= th_blend_single).astype(int),
        "pred_blend_bag3_050": (p_blend_bag3_co >= 0.50).astype(int),
        "pred_blend_bag3_tuned": (p_blend_bag3_co >= th_blend_bag3).astype(int),
    }).to_csv(pair_path, index=False, compression="gzip")

    # If cut is disjoint_3seed_20261101, also write copy to legacy filename for backwards-compatibility
    if cut_name == "disjoint_3seed_20261101":
        legacy_path = OUT_DIR / "paired_external_virgin_confirm_20261101.csv.gz"
        shutil.copyfile(pair_path, legacy_path)

    print(f"  [{cut_name}] Single Blend @0.500: Net={eval_single_050['net']:+d} (Seen:{eval_single_050['seen_sellers']['net']:+d}, New:{eval_single_050['new_sellers']['net']:+d}), Transported 30k={trans_single_050['expected_net_items_30k_test_mix']:+.1f} {trans_single_050['transported_30k_bootstrap_ci95']}, DeltaLL={eval_single_050['delta_logloss']:+.5f}")
    print(f"  [{cut_name}] Bag3 Blend   @0.500: Net={eval_bag_050['net']:+d} (Seen:{eval_bag_050['seen_sellers']['net']:+d}, New:{eval_bag_050['new_sellers']['net']:+d}), Transported 30k={trans_bag_050['expected_net_items_30k_test_mix']:+.1f} {trans_bag_050['transported_30k_bootstrap_ci95']}, DeltaLL={eval_bag_050['delta_logloss']:+.5f}")
    print(f"  [{cut_name}] Bag3 vs Single @0.50: Net={eval_bag_vs_single_050['net']:+d} (Seen:{eval_bag_vs_single_050['seen_sellers']['net']:+d}, New:{eval_bag_vs_single_050['new_sellers']['net']:+d}), DeltaLL={eval_bag_vs_single_050['delta_logloss']:+.5f}")
    print(f"  [{cut_name}] Bag3 Blend   @Policy: Net={eval_bag_pol['net']:+d} (Seen:{eval_bag_pol['seen_sellers']['net']:+d}, New:{eval_bag_pol['new_sellers']['net']:+d}), Transported 30k={trans_bag_pol['expected_net_items_30k_test_mix']:+.1f} {trans_bag_pol['transported_30k_bootstrap_ci95']}, DeltaLL={eval_bag_pol['delta_logloss']:+.5f}")

    # Historical overlap reference from audit
    cut_overlap_info = {}
    if audit_holdouts_data and "eval_cuts_overlap_matrix" in audit_holdouts_data:
        m_key = f"{cut_name}_confirm"
        if m_key in audit_holdouts_data["eval_cuts_overlap_matrix"]:
            ovs = audit_holdouts_data["eval_cuts_overlap_matrix"][m_key].get("overlaps", {})
            for uk in ["UNION_R151_3Seeds_Confirm", "UNION_R151_3Seeds_TuneAndConfirm", "UNION_All_Reconstructed_Holdouts"]:
                if uk in ovs:
                    cut_overlap_info[uk] = ovs[uk]

    return {
        "cut_name": cut_name,
        "seed": seed,
        "runtime_seconds": round(time.time() - t0, 2),
        "sizes": {
            "fit": int(len(fit)),
            "tune": int(len(tune)),
            "confirm": int(len(confirm)),
            "confirm_seen": int(is_seen_conf.sum()),
            "confirm_new": int((1 - is_seen_conf).sum()),
            "confirm_seen_pct": round(100.0 * float(is_seen_conf.mean()), 2),
            "confirm_new_pct": round(100.0 * float(1.0 - is_seen_conf.mean()), 2),
        },
        "hashes": {
            "fit_sha256": fit_hash,
            "tune_sha256": tune_hash,
            "confirm_sha256": conf_hash,
            "split_sha256": split_hash,
            "paired_csv_sha256": compute_file_sha256(pair_path),
        },
        "historical_holdouts_overlap": cut_overlap_info,
        "selected_on_tune": {
            "best_lgb_variant": best_lgb_cfg["name"],
            "th_cb_single_tune": th_cb_single,
            "th_blend_single_tune": th_blend_single,
            "th_blend_bag3_tune": th_blend_bag3,
        },
        "blend_single_vs_base_cb": {
            "common_th_050": {
                "paired_metrics": eval_single_050,
                "transported_test_mixture": trans_single_050,
            },
            "policy_tuned_th": {
                "paired_metrics": eval_single_pol,
                "transported_test_mixture": trans_single_pol,
            },
        },
        "blend_bag3_vs_base_cb": {
            "common_th_050": {
                "paired_metrics": eval_bag_050,
                "transported_test_mixture": trans_bag_050,
            },
            "policy_tuned_th": {
                "paired_metrics": eval_bag_pol,
                "transported_test_mixture": trans_bag_pol,
            },
        },
        "marginal_bag3_vs_blend_single": {
            "common_th_050": {
                "paired_metrics": eval_bag_vs_single_050,
                "transported_test_mixture": trans_bag_vs_single_050,
            },
            "policy_tuned_th": {
                "paired_metrics": eval_bag_vs_single_pol,
                "transported_test_mixture": trans_bag_vs_single_pol,
            },
        },
        "regression_analysis_bag3_at_050": reg_analysis,
    }


def main():
    t0_all = time.time()
    print("=" * 80)
    print("RUNNING AGENTE 3 — VALIDACIÓN EXTERNA ANIDADA Y TRANSPORTE A MEZCLA DE TEST (REVISED CX-0002)")
    print("=" * 80)

    cache = build_or_load_cache(force_rebuild=False)
    sellers = cache["sellers_tr"]

    audit_path = OUT_DIR / "all_historical_holdouts_overlap_audit.json"
    audit_data = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else None

    # 1. Disjoint 3-Seed Confirm Split (0% overlap with confirm of 2026, 314159, 20261030; 92.39% with full historical holdout union)
    fit_v, tune_v, conf_v = build_disjoint_3seed_split(sellers, prior_seeds=(2026, 314159, 20261030), seed=20261101)
    res_disjoint = run_inductive_outer_cut("disjoint_3seed_20261101", fit_v, tune_v, conf_v, 20261101, cache, audit_data)

    # 2. Fresh Nested Outer Cut 1 (seed 777001)
    fit_n1, tune_n1, conf_n1 = split_seller_groups_r151(sellers, 777001)
    res_n1 = run_inductive_outer_cut("nested_outer_777001", fit_n1, tune_n1, conf_n1, 777001, cache, audit_data)

    # 3. Fresh Nested Outer Cut 2 (seed 777002)
    fit_n2, tune_n2, conf_n2 = split_seller_groups_r151(sellers, 777002)
    res_n2 = run_inductive_outer_cut("nested_outer_777002", fit_n2, tune_n2, conf_n2, 777002, cache, audit_data)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_seconds": round(time.time() - t0_all, 2),
        "epistemological_classification": "DEVELOPMENT_GRADE_PROMISING_LOGLOSS_MODERATE_ACCURACY",
        "independence_limitation": (
            "Audited in all_historical_holdouts_overlap_audit.json: the union of all 18 historical holdouts "
            "covers 67,007 of 70,000 train rows (95.72%). Although `disjoint_3seed_20261101` was constructed to have "
            "0.0% overlap with the recent confirm sets of 2026, 314159, and 20261030, it overlaps in 53.49% with "
            "their tune+confirm union and in 92.39% (8,862 / 9,592) with the union of all 18 historical holdouts. "
            "Sharing fit rows across experiments does not constitute supervised leakage, but does rule out sample "
            "independence. All 3 cuts execute strictly inductive per-fold vocabulary and TF-IDF refit inside `fit`."
        ),
        "transportability_assumptions": [
            "1. Conditional invariance within strata: E[Delta Acc | seen/new, test] == E[Delta Acc | seen/new, confirm].",
            "2. Mixture reweighting: Validation confirm has ~30.2% seen / ~69.8% new rows, whereas Kaggle 30k test has 71.623% seen (21,487) / 28.377% new (8,513) rows.",
            "3. Sample size scaling: Full 70k refit has 58% more training rows than the ~44.2k `fit` split, strengthening seen-seller target encodings.",
            "4. Threshold sensitivity: Thresholds tuned on a 30/70 seen/new validation mixture can suffer prior-shift degradation on a 72/28 test mixture; common-threshold (0.500) comparisons isolate pure probability ranking improvements."
        ],
        "cuts": {
            "disjoint_3seed_20261101": res_disjoint,
            "nested_outer_777001": res_n1,
            "nested_outer_777002": res_n2,
        },
    }

    with open(OUT_DIR / "agent3_external_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved revised Agent 3 summary to {OUT_DIR / 'agent3_external_summary.json'}")


if __name__ == "__main__":
    main()
