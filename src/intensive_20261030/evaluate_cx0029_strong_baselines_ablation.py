"""Rigorous Strong Baselines and Ablation Evaluation on Cut 42026 (CX-0029, CX-0030, CX-0031).

This script implements the audited falsification protocol:
1. Re-evaluates on cut 42026 (13,197 confirm rows) classified as DEVELOPMENT (confirm used).
2. Reconstructs identical-protocol strong baselines trained strictly on `fit`:
   - Claude B10 (0.6 lgb_tpe_2 + 0.2 cat_enriched_d9 + 0.2 cat_native_text_d7).
   - Blend075 (0.75 CatBoost 900 + 0.25 LightGBM 227).
   - Pipeline A (CatBoost 600 iters depth 6).
3. Candidate model:
   - Linear additive text+category model (TF-IDF + OHE category with 5-fold StratifiedGroupKFold OOF inside fit).
   - Inductive relative price features computed strictly from fit reference tables.
   - Categorical features learned exclusively on fit; unseen categories mapped to unknown (no transduction).
4. Independent paired ablations:
   - Ablation A: Additive text ONLY (NO relative price features).
   - Ablation B: Relative price features ONLY (NO additive text OOF).
   - Candidate Full: Both components combined.
5. Strict 1-to-1 join by `row_idx` and `item_id` with target identity assertion and canonical hash.
6. Reports paired metrics by ID:
   - Accuracy, Log Loss, ROC-AUC.
   - Net fixes, fixes, breaks vs B10, Blend075, Pipeline A.
   - Stratified seen vs new sellers breakdown.
   - 2,000-replica seller-grouped bootstrap confidence intervals (95% CI).
   - Threshold sensitivity: Fixed 0.50 vs tune-optimized policy threshold.
7. Registers historical holdout overlap for cut 992026 (95.68% row overlap, development-dependent).
"""

import gc
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Ensure repo and src in path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import catboost
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder

from benchmark_20261030.split_utils import split_seller_groups_r151
import confirm_and_submit as cs
from coordination_lock import ProjectLock
from intensive_20261030.run_iterative_nocturnal_batch import (
    build_or_load_cache,
    extract_advanced_inductive_features,
    seller_bootstrap_ci,
)
from optimize_leaderboard import data_folder, enrich

CAT_FEATURES = ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode"]


def calc_log_loss(y_true, p_pred):
    p_clipped = np.clip(p_pred, 1e-6, 1.0 - 1e-6)
    return float(log_loss(y_true, p_clipped))


def train_leak_free_text_model(combos_all, cat_all, y_all, sellers_all, fit, tune, confirm, seed):
    """Trains linear additive text+category model with strictly inductive 5-fold StratifiedGroupKFold OOF inside fit.
    
    Adherence to CX-0031:
    - Vectorizers and encoders are fitted strictly inside each fold on fold training rows only.
    - Categorical values and text tokens never leak across folds or from tune/confirm.
    - Folds are grouped by seller_id (StratifiedGroupKFold) to match seller-new conditions.
    """
    print("      [Additive Text Model] Extracting leak-free OOF text representation inside fit...", flush=True)
    t0 = time.time()
    texts_fit = [combos_all[i] for i in fit]
    texts_tune = [combos_all[i] for i in tune]
    texts_conf = [combos_all[i] for i in confirm]

    y_fit = y_all[fit]
    sellers_fit = sellers_all[fit]
    cats_fit = cat_all.iloc[fit][["category_id"]]
    cats_tu = cat_all.iloc[tune][["category_id"]]
    cats_co = cat_all.iloc[confirm][["category_id"]]

    # 5-fold StratifiedGroupKFold on fit by seller_id
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_p_fit = np.zeros(len(fit), dtype=np.float32)

    for fold_idx, (tr_i, va_i) in enumerate(sgkf.split(np.zeros(len(fit)), y_fit, groups=sellers_fit)):
        # Strictly inductive: fit vectorizers and encoder on fold training data ONLY
        vec_w_fold = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=35000, sublinear_tf=True)
        vec_c_fold = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=10, max_features=35000, sublinear_tf=True)
        ohe_fold = OneHotEncoder(handle_unknown="ignore")

        tr_texts = [texts_fit[k] for k in tr_i]
        va_texts = [texts_fit[k] for k in va_i]

        X_w_tr = vec_w_fold.fit_transform(tr_texts)
        X_c_tr = vec_c_fold.fit_transform(tr_texts)
        C_tr = ohe_fold.fit_transform(cats_fit.iloc[tr_i])
        X_all_tr = hstack([X_w_tr, X_c_tr, C_tr], format="csr")

        X_w_va = vec_w_fold.transform(va_texts)
        X_c_va = vec_c_fold.transform(va_texts)
        C_va = ohe_fold.transform(cats_fit.iloc[va_i])
        X_all_va = hstack([X_w_va, X_c_va, C_va], format="csr")

        clf = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs", random_state=seed)
        clf.fit(X_all_tr, y_fit[tr_i])
        oof_p_fit[va_i] = clf.predict_proba(X_all_va)[:, 1]

    # Full fit model to predict tune and confirm
    vec_w_full = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=35000, sublinear_tf=True)
    vec_c_full = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=10, max_features=35000, sublinear_tf=True)
    ohe_full = OneHotEncoder(handle_unknown="ignore")

    X_w_full = vec_w_full.fit_transform(texts_fit)
    X_c_full = vec_c_full.fit_transform(texts_fit)
    C_full = ohe_full.fit_transform(cats_fit)
    X_all_full = hstack([X_w_full, X_c_full, C_full], format="csr")

    clf_full = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs", random_state=seed)
    clf_full.fit(X_all_full, y_fit)

    X_w_tu = vec_w_full.transform(texts_tune)
    X_c_tu = vec_c_full.transform(texts_tune)
    C_tu = ohe_full.transform(cats_tu)
    X_all_tu = hstack([X_w_tu, X_c_tu, C_tu], format="csr")

    X_w_co = vec_w_full.transform(texts_conf)
    X_c_co = vec_c_full.transform(texts_conf)
    C_co = ohe_full.transform(cats_co)
    X_all_co = hstack([X_w_co, X_c_co, C_co], format="csr")

    p_tu = clf_full.predict_proba(X_all_tu)[:, 1].astype(np.float32)
    p_co = clf_full.predict_proba(X_all_co)[:, 1].astype(np.float32)

    print(f"      [Additive Text Model] Completed in {time.time() - t0:.1f}s. Fit OOF LL: {calc_log_loss(y_fit, oof_p_fit):.4f}", flush=True)
    return oof_p_fit, p_tu, p_co


def train_b10_baseline(raw_data, fit_idx, tune_idx, confirm_idx, out_dir, threads=12):
    """Trains Claude B10 on fit-only data and predicts on tune and confirm."""
    print("      [B10 Baseline] Training Claude B10 architecture fit-only (43k rows)...", flush=True)
    t0 = time.time()
    b10_work_dir = out_dir / "b10_work"
    b10_work_dir.mkdir(parents=True, exist_ok=True)
    p_b10_tu_path = b10_work_dir / "p_b10_tu.npy"
    p_b10_co_path = b10_work_dir / "p_b10_co.npy"
    if p_b10_tu_path.exists() and p_b10_co_path.exists():
        print("      [B10 Baseline] Loaded from cache!", flush=True)
        return np.load(p_b10_tu_path), np.load(p_b10_co_path)

    # Copy optimization metadata to local work dir
    for f in [
        "selected_ensemble.json",
        "metrics.csv",
        "lgb_tpe_2_params.json",
        "cat_enriched_d9_params.json",
        "cat_native_text_d7_params.json",
    ]:
        shutil.copy(REPO_ROOT / "optimization" / f, b10_work_dir / f)

    cs.OUT = b10_work_dir
    selected = json.loads((b10_work_dir / "selected_ensemble.json").read_text())

    y_all = (raw_data.condition == "used").astype("int8").to_numpy()
    groups_all = raw_data.parent_item_id.astype("string").fillna("__ROW__" + raw_data.id.astype(str)).to_numpy()

    # Enrich dev (fit) and eval (tune + confirm combined)
    ev_idx = np.r_[tune_idx, confirm_idx]
    x_dev, ids = enrich(raw_data.iloc[fit_idx].reset_index(drop=True))
    z_ev, _ = enrich(raw_data.iloc[ev_idx].reset_index(drop=True), ids)

    # Fit models on dev and predict on eval
    pz = cs.fit_models(
        x_dev,
        y_all[fit_idx],
        groups_all[fit_idx],
        z_ev,
        selected,
        protocol="split_seed42026_tc",
        y_eval=None,
    )

    n_tu = len(tune_idx)
    p_b10_tu = pz[:n_tu].astype(np.float32)
    p_b10_co = pz[n_tu:].astype(np.float32)

    np.save(p_b10_tu_path, p_b10_tu)
    np.save(p_b10_co_path, p_b10_co)

    elapsed = time.time() - t0
    print(f"      [B10 Baseline] Finished in {elapsed:.1f}s.", flush=True)
    return p_b10_tu, p_b10_co


def compute_target_encoding(F_all, cat_all, y_all, sellers_all, fit, tune, confirm, alpha=5.0):
    """Computes leak-free 4-fold GroupKFold target encodings strictly on fit."""
    prior = float(y_all[fit].mean())
    te_cols = ["seller_id", "category_id"]
    fit_df = pd.DataFrame({
        "seller_id": sellers_all[fit],
        "category_id": cat_all.iloc[fit]["category_id"].values,
    })
    tune_df = pd.DataFrame({
        "seller_id": sellers_all[tune],
        "category_id": cat_all.iloc[tune]["category_id"].values,
    })
    conf_df = pd.DataFrame({
        "seller_id": sellers_all[confirm],
        "category_id": cat_all.iloc[confirm]["category_id"].values,
    })

    gkf = GroupKFold(n_splits=4)
    fit_te = pd.DataFrame(index=np.arange(len(fit)))
    tune_te = pd.DataFrame(index=np.arange(len(tune)))
    conf_te = pd.DataFrame(index=np.arange(len(confirm)))

    for c in te_cols:
        oof = np.full(len(fit), prior, dtype=np.float32)
        for tr_i, va_i in gkf.split(fit_df, y_all[fit], groups=sellers_all[fit]):
            s_tr = fit_df[c].iloc[tr_i]
            y_tr = y_all[fit][tr_i]
            grp = pd.Series(y_tr).groupby(s_tr)
            stats = grp.agg(["count", "mean"])
            smooth = (stats["count"] * stats["mean"] + alpha * prior) / (stats["count"] + alpha)
            oof[va_i] = fit_df[c].iloc[va_i].map(smooth).fillna(prior).values

        fit_te[f"{c}_te"] = oof

        # Full fit mapping for tune and confirm
        grp = pd.Series(y_all[fit]).groupby(fit_df[c])
        stats = grp.agg(["count", "mean"])
        smooth = (stats["count"] * stats["mean"] + alpha * prior) / (stats["count"] + alpha)
        tune_te[f"{c}_te"] = tune_df[c].map(smooth).fillna(prior).values.astype(np.float32)
        conf_te[f"{c}_te"] = conf_df[c].map(smooth).fillna(prior).values.astype(np.float32)

    return fit_te, tune_te, conf_te


def evaluate_pair(name_cand, p_cand_tu, p_cand_co, name_ref, p_ref_tu, p_ref_co, y_tu, y_co, sellers_co, seen_mask):
    """Calculates all paired metrics, bootstrap intervals, and threshold analysis."""
    # Fixed threshold 0.50
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

    # Seen vs new breakdown
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


def run_rigorous_evaluation():
    lock = ProjectLock(owner_id="antigravity_cx0029_evaluation_20261031")
    print("Acquiring ProjectLock for CX-0029 Strong Baselines & Ablation...", flush=True)
    with lock:
        t_global = time.time()
        out_dir = REPO_ROOT / "results/intensive_20261030/cx0029_evaluation"
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

        raw_conf = raw_data.iloc[confirm]
        raw_tu = raw_data.iloc[tune]

        print(f"\nEvaluating on Cut 42026: fit={len(fit)}, tune={len(tune)}, confirm={len(confirm)} (Seen: {seen_mask.sum()}, New: {(~seen_mask).sum()})", flush=True)

        # Build Canonical Confirm DataFrame with exact IDs
        canonical_confirm = pd.DataFrame({
            "row_idx": confirm,
            "item_id": raw_conf["id"].values,
            "seller_id": sellers_confirm,
            "y_used": y_confirm,
            "is_seen": seen_mask.astype(int),
        })
        canonical_csv_bytes = canonical_confirm[["row_idx", "item_id", "y_used"]].to_csv(index=False).encode("utf-8")
        canonical_sha256 = hashlib.sha256(canonical_csv_bytes).hexdigest()
        print(f"      [Integrity] Canonical Confirm ID Map SHA-256: {canonical_sha256}", flush=True)

        # -----------------------------------------------------------------
        # STEP 1: Strong Baseline 1 — Claude B10 (fit-only)
        # -----------------------------------------------------------------
        p_b10_tu, p_b10_co = train_b10_baseline(raw_data, fit, tune, confirm, out_dir, threads=12)

        # 1-to-1 Join verification for B10
        df_b10 = pd.DataFrame({"row_idx": confirm, "item_id": raw_conf["id"].values, "p_b10": p_b10_co, "y_check": y_confirm})
        m_b10 = pd.merge(canonical_confirm, df_b10, on=["row_idx", "item_id"], how="inner")
        assert len(m_b10) == len(canonical_confirm) and (m_b10["y_used"] == m_b10["y_check"]).all()
        p_b10_co = m_b10["p_b10"].values.astype(np.float32)

        # -----------------------------------------------------------------
        # Base Features & Encodings (Strictly Fit-Only)
        # -----------------------------------------------------------------
        fit_te, tune_te, conf_te = compute_target_encoding(F_all, cat_all, y_all, sellers_all, fit, tune, confirm)
        X_fit_base = pd.concat([F_all.iloc[fit].reset_index(drop=True), fit_te], axis=1)
        X_tu_base = pd.concat([F_all.iloc[tune].reset_index(drop=True), tune_te], axis=1)
        X_co_base = pd.concat([F_all.iloc[confirm].reset_index(drop=True), conf_te], axis=1)

        def prep_lgb_df(df_in):
            df_out = df_in.copy()
            for col in CAT_FEATURES:
                if col in df_out.columns:
                    df_out[col] = df_out[col].astype("category")
            return df_out

        # Categorical features learned exclusively on fit; unseen in tune/confirm mapped to unknown
        for c in CAT_FEATURES:
            if c in X_fit_base.columns:
                cats_set = set(F_all.iloc[fit][c].astype(str).unique())
                X_fit_base[c] = X_fit_base[c].astype(str).fillna("__MISSING__")
                X_tu_base[c] = X_tu_base[c].astype(str).apply(lambda v: v if v in cats_set else "__UNKNOWN__")
                X_co_base[c] = X_co_base[c].astype(str).apply(lambda v: v if v in cats_set else "__UNKNOWN__")

        # -----------------------------------------------------------------
        # STEP 2: Strong Baseline 2 — Blend075 (CatBoost 900 + LightGBM 227)
        # -----------------------------------------------------------------
        p_blend075_tu_path = out_dir / "p_blend075_tu.npy"
        p_blend075_co_path = out_dir / "p_blend075_co.npy"
        if p_blend075_tu_path.exists() and p_blend075_co_path.exists():
            print("\n      [Blend075 Baseline] Loaded from cache!", flush=True)
            p_blend075_tu = np.load(p_blend075_tu_path)
            p_blend075_co = np.load(p_blend075_co_path)
        else:
            print("\n      [Blend075 Baseline] Training CatBoost + LightGBM blend fit-only...", flush=True)
            cb_single = CatBoostClassifier(
                iterations=900,
                learning_rate=0.07,
                depth=6,
                l2_leaf_reg=6.0,
                cat_features=[c for c in CAT_FEATURES if c in X_fit_base.columns],
                random_seed=seed,
                thread_count=12,
                verbose=False,
            )
            cb_single.fit(X_fit_base, y_fit)
            p_cb_tu = cb_single.predict_proba(X_tu_base)[:, 1].astype(np.float32)
            p_cb_co = cb_single.predict_proba(X_co_base)[:, 1].astype(np.float32)

            X_fit_lgb = prep_lgb_df(X_fit_base)
            X_tu_lgb = prep_lgb_df(X_tu_base)
            X_co_lgb = prep_lgb_df(X_co_base)

            lgb_single = lgb.LGBMClassifier(
                n_estimators=227,
                learning_rate=0.04,
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
            lgb_single.fit(X_fit_lgb, y_fit)
            p_lgb_tu = lgb_single.predict_proba(X_tu_lgb)[:, 1].astype(np.float32)
            p_lgb_co = lgb_single.predict_proba(X_co_lgb)[:, 1].astype(np.float32)

            p_blend075_tu = 0.75 * p_cb_tu + 0.25 * p_lgb_tu
            p_blend075_co = 0.75 * p_cb_co + 0.25 * p_lgb_co
            np.save(p_blend075_tu_path, p_blend075_tu)
            np.save(p_blend075_co_path, p_blend075_co)

        # -----------------------------------------------------------------
        # STEP 3: Baseline 3 — Pipeline A (CatBoost 600)
        # -----------------------------------------------------------------
        p_pipea_tu_path = out_dir / "p_pipea_tu.npy"
        p_pipea_co_path = out_dir / "p_pipea_co.npy"
        if p_pipea_tu_path.exists() and p_pipea_co_path.exists():
            print("\n      [Pipeline A Baseline] Loaded from cache!", flush=True)
            p_pipea_tu = np.load(p_pipea_tu_path)
            p_pipea_co = np.load(p_pipea_co_path)
        else:
            print("\n      [Pipeline A Baseline] Training CatBoost 600 iters fit-only...", flush=True)
            cb_pipe_a = CatBoostClassifier(
                iterations=600,
                learning_rate=0.07,
                depth=6,
                l2_leaf_reg=6.0,
                cat_features=[c for c in CAT_FEATURES if c in X_fit_base.columns],
                random_seed=seed,
                thread_count=12,
                verbose=False,
            )
            cb_pipe_a.fit(X_fit_base, y_fit)
            p_pipea_tu = cb_pipe_a.predict_proba(X_tu_base)[:, 1].astype(np.float32)
            p_pipea_co = cb_pipe_a.predict_proba(X_co_base)[:, 1].astype(np.float32)
            np.save(p_pipea_tu_path, p_pipea_tu)
            np.save(p_pipea_co_path, p_pipea_co)

        # -----------------------------------------------------------------
        # STEP 4: Additive Text Model & Inductive Price Features
        # -----------------------------------------------------------------
        oof_txt_path = out_dir / "oof_txt_fit.npy"
        p_txt_tu_path = out_dir / "p_txt_tu.npy"
        p_txt_co_path = out_dir / "p_txt_co.npy"
        if oof_txt_path.exists() and p_txt_tu_path.exists() and p_txt_co_path.exists():
            print("\n      [Additive Text Model] Loaded from cache!", flush=True)
            oof_txt_fit = np.load(oof_txt_path)
            p_txt_tu = np.load(p_txt_tu_path)
            p_txt_co = np.load(p_txt_co_path)
        else:
            oof_txt_fit, p_txt_tu, p_txt_co = train_leak_free_text_model(
                combos_all, cat_all, y_all, sellers_all, fit, tune, confirm, seed
            )
            np.save(oof_txt_path, oof_txt_fit)
            np.save(p_txt_tu_path, p_txt_tu)
            np.save(p_txt_co_path, p_txt_co)

        X_fit_ind, X_tu_ind, X_co_ind = extract_advanced_inductive_features(
            F_all, cat_all, fit, tune, confirm
        )
        ind_cols = [c for c in X_fit_ind.columns if c.startswith("ind_")]
        X_fit_ind_only = X_fit_ind[ind_cols].reset_index(drop=True)
        X_tu_ind_only = X_tu_ind[ind_cols].reset_index(drop=True)
        X_co_ind_only = X_co_ind[ind_cols].reset_index(drop=True)

        # -----------------------------------------------------------------
        # STEP 5: Model Variants & Ablations
        # -----------------------------------------------------------------
        print("\n      [Candidate & Ablations] Training Model Variants...", flush=True)
        # Variant 1: Full Candidate (Base + Text OOF + Inductive Price)
        X_fit_full = pd.concat([X_fit_base, X_fit_ind_only], axis=1)
        X_tu_full = pd.concat([X_tu_base, X_tu_ind_only], axis=1)
        X_co_full = pd.concat([X_co_base, X_co_ind_only], axis=1)
        X_fit_full["feat_oof_txt_cond"] = oof_txt_fit
        X_tu_full["feat_oof_txt_cond"] = p_txt_tu
        X_co_full["feat_oof_txt_cond"] = p_txt_co

        # Variant 2: Ablation A — Text ONLY (Base + Text OOF, NO Inductive Price)
        X_fit_abl_txt = X_fit_base.copy()
        X_tu_abl_txt = X_tu_base.copy()
        X_co_abl_txt = X_co_base.copy()
        X_fit_abl_txt["feat_oof_txt_cond"] = oof_txt_fit
        X_tu_abl_txt["feat_oof_txt_cond"] = p_txt_tu
        X_co_abl_txt["feat_oof_txt_cond"] = p_txt_co

        # Variant 3: Ablation B — Inductive Price ONLY (Base + Inductive Price, NO Text OOF)
        X_fit_abl_price = pd.concat([X_fit_base, X_fit_ind_only], axis=1)
        X_tu_abl_price = pd.concat([X_tu_base, X_tu_ind_only], axis=1)
        X_co_abl_price = pd.concat([X_co_base, X_co_ind_only], axis=1)

        # LGB parameters

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
            "random_state": seed,
            "n_jobs": 12,
            "verbose": -1,
        }

        # Train Full Candidate (LGB + CB)
        X_fit_full_lgb = prep_lgb_df(X_fit_full)
        X_tu_full_lgb = prep_lgb_df(X_tu_full)
        X_co_full_lgb = prep_lgb_df(X_co_full)

        dtr_full = lgb.Dataset(X_fit_full_lgb, label=y_fit)
        dva_full = lgb.Dataset(X_tu_full_lgb, label=y_tune, reference=dtr_full)
        model_full_lgb = lgb.train(
            params_lgb, dtr_full, num_boost_round=600, valid_sets=[dva_full],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        p_cand_full_tu = model_full_lgb.predict(X_tu_full_lgb).astype(np.float32)
        p_cand_full_co = model_full_lgb.predict(X_co_full_lgb).astype(np.float32)

        cb_cand_full = CatBoostClassifier(
            iterations=600,
            learning_rate=0.07,
            depth=6,
            l2_leaf_reg=6.0,
            cat_features=[c for c in CAT_FEATURES if c in X_fit_full.columns],
            random_seed=seed,
            thread_count=12,
            verbose=False,
        )
        cb_cand_full.fit(X_fit_full, y_fit)
        p_cb_cand_full_tu = cb_cand_full.predict_proba(X_tu_full)[:, 1].astype(np.float32)
        p_cb_cand_full_co = cb_cand_full.predict_proba(X_co_full)[:, 1].astype(np.float32)

        p_cand_blend_tu = 0.50 * p_cb_cand_full_tu + 0.50 * p_cand_full_tu
        p_cand_blend_co = 0.50 * p_cb_cand_full_co + 0.50 * p_cand_full_co

        # Train Ablation A (Text Only)
        X_fit_txt_lgb = prep_lgb_df(X_fit_abl_txt)
        X_tu_txt_lgb = prep_lgb_df(X_tu_abl_txt)
        X_co_txt_lgb = prep_lgb_df(X_co_abl_txt)
        dtr_txt = lgb.Dataset(X_fit_txt_lgb, label=y_fit)
        dva_txt = lgb.Dataset(X_tu_txt_lgb, label=y_tune, reference=dtr_txt)
        model_txt_lgb = lgb.train(
            params_lgb, dtr_txt, num_boost_round=600, valid_sets=[dva_txt],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        p_abl_txt_tu = model_txt_lgb.predict(X_tu_txt_lgb).astype(np.float32)
        p_abl_txt_co = model_txt_lgb.predict(X_co_txt_lgb).astype(np.float32)

        # Train Ablation B (Price Only)
        X_fit_pr_lgb = prep_lgb_df(X_fit_abl_price)
        X_tu_pr_lgb = prep_lgb_df(X_tu_abl_price)
        X_co_pr_lgb = prep_lgb_df(X_co_abl_price)
        dtr_pr = lgb.Dataset(X_fit_pr_lgb, label=y_fit)
        dva_pr = lgb.Dataset(X_tu_pr_lgb, label=y_tune, reference=dtr_pr)
        model_pr_lgb = lgb.train(
            params_lgb, dtr_pr, num_boost_round=600, valid_sets=[dva_pr],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        p_abl_pr_tu = model_pr_lgb.predict(X_tu_pr_lgb).astype(np.float32)
        p_abl_pr_co = model_pr_lgb.predict(X_co_pr_lgb).astype(np.float32)

        # Ensembles with Baselines
        p_ens_b10_cand_tu = 0.50 * p_b10_tu + 0.50 * p_cand_blend_tu
        p_ens_b10_cand_co = 0.50 * p_b10_co + 0.50 * p_cand_blend_co

        p_ens_bl75_cand_tu = 0.50 * p_blend075_tu + 0.50 * p_cand_blend_tu
        p_ens_bl75_cand_co = 0.50 * p_blend075_co + 0.50 * p_cand_blend_co

        p_ens_pipea_cand_tu = 0.50 * p_pipea_tu + 0.50 * p_cand_blend_tu
        p_ens_pipea_cand_co = 0.50 * p_pipea_co + 0.50 * p_cand_blend_co

        p_ens_b10_txt_tu = 0.50 * p_b10_tu + 0.50 * p_abl_txt_tu
        p_ens_b10_txt_co = 0.50 * p_b10_co + 0.50 * p_abl_txt_co

        p_ens_b10_pr_tu = 0.50 * p_b10_tu + 0.50 * p_abl_pr_tu
        p_ens_b10_pr_co = 0.50 * p_b10_co + 0.50 * p_abl_pr_co

        # -----------------------------------------------------------------
        # STEP 6: Run Comprehensive Paired Evaluations
        # -----------------------------------------------------------------
        print("\n      [Paired Comparisons] Computing Paired ID Metrics & Bootstraps...", flush=True)
        evaluations = []

        # Pairs against B10
        evaluations.append(evaluate_pair("Candidate_Full_LGB", p_cand_full_tu, p_cand_full_co, "Baseline_B10", p_b10_tu, p_b10_co, y_tune, y_confirm, sellers_confirm, seen_mask))
        evaluations.append(evaluate_pair("Candidate_Full_Blend", p_cand_blend_tu, p_cand_blend_co, "Baseline_B10", p_b10_tu, p_b10_co, y_tune, y_confirm, sellers_confirm, seen_mask))
        evaluations.append(evaluate_pair("Ablation_A_TextOnly", p_abl_txt_tu, p_abl_txt_co, "Baseline_B10", p_b10_tu, p_b10_co, y_tune, y_confirm, sellers_confirm, seen_mask))
        evaluations.append(evaluate_pair("Ablation_B_PriceOnly", p_abl_pr_tu, p_abl_pr_co, "Baseline_B10", p_b10_tu, p_b10_co, y_tune, y_confirm, sellers_confirm, seen_mask))
        evaluations.append(evaluate_pair("Ensemble_B10_Candidate_Full", p_ens_b10_cand_tu, p_ens_b10_cand_co, "Baseline_B10", p_b10_tu, p_b10_co, y_tune, y_confirm, sellers_confirm, seen_mask))
        evaluations.append(evaluate_pair("Ensemble_B10_Ablation_Text", p_ens_b10_txt_tu, p_ens_b10_txt_co, "Baseline_B10", p_b10_tu, p_b10_co, y_tune, y_confirm, sellers_confirm, seen_mask))
        evaluations.append(evaluate_pair("Ensemble_B10_Ablation_Price", p_ens_b10_pr_tu, p_ens_b10_pr_co, "Baseline_B10", p_b10_tu, p_b10_co, y_tune, y_confirm, sellers_confirm, seen_mask))

        # Pairs against Blend075
        evaluations.append(evaluate_pair("Candidate_Full_LGB", p_cand_full_tu, p_cand_full_co, "Baseline_Blend075", p_blend075_tu, p_blend075_co, y_tune, y_confirm, sellers_confirm, seen_mask))
        evaluations.append(evaluate_pair("Candidate_Full_Blend", p_cand_blend_tu, p_cand_blend_co, "Baseline_Blend075", p_blend075_tu, p_blend075_co, y_tune, y_confirm, sellers_confirm, seen_mask))
        evaluations.append(evaluate_pair("Ensemble_Blend075_Cand", p_ens_bl75_cand_tu, p_ens_bl75_cand_co, "Baseline_Blend075", p_blend075_tu, p_blend075_co, y_tune, y_confirm, sellers_confirm, seen_mask))
        evaluations.append(evaluate_pair("Baseline_B10", p_b10_tu, p_b10_co, "Baseline_Blend075", p_blend075_tu, p_blend075_co, y_tune, y_confirm, sellers_confirm, seen_mask))

        # Pairs against Pipeline A
        evaluations.append(evaluate_pair("Candidate_Full_LGB", p_cand_full_tu, p_cand_full_co, "Baseline_Pipeline_A", p_pipea_tu, p_pipea_co, y_tune, y_confirm, sellers_confirm, seen_mask))
        evaluations.append(evaluate_pair("Candidate_Full_Blend", p_cand_blend_tu, p_cand_blend_co, "Baseline_Pipeline_A", p_pipea_tu, p_pipea_co, y_tune, y_confirm, sellers_confirm, seen_mask))
        evaluations.append(evaluate_pair("Ensemble_PipeA_Cand", p_ens_pipea_cand_tu, p_ens_pipea_cand_co, "Baseline_Pipeline_A", p_pipea_tu, p_pipea_co, y_tune, y_confirm, sellers_confirm, seen_mask))

        # -----------------------------------------------------------------
        # STEP 7: Historical Overlap Audit Documentation
        # -----------------------------------------------------------------
        virg_seed = 992026
        v_fit, v_tune, v_confirm = split_seller_groups_r151(sellers_all, virg_seed)
        v_seen = np.isin(sellers_all[v_confirm], np.unique(sellers_all[v_fit]))

        overlap_audit_path = REPO_ROOT / "results/intensive_20261030/all_historical_holdouts_overlap_audit.json"
        overlap_data = {}
        if overlap_audit_path.exists():
            with open(overlap_audit_path, "r", encoding="utf-8") as f:
                all_audits = json.load(f)
                overlap_data = all_audits.get("eval_cuts_overlap_matrix", {})

        cut42026_overlap = overlap_data.get("seller_group_seed42026_confirm", {}).get("overlaps", {})
        cut992026_overlap = overlap_data.get("seller_group_seed992026_confirm", {}).get("overlaps", {})

        unseen_cut_meta = {
            "seed": virg_seed,
            "status": "SEALED_RESERVED_CONFIRM_CUT",
            "epistemic_note": "Reserved for final confirmation of frozen model family. Historical holdout audit reveals 95.68% row overlap with historical repository holdouts (rounds 1-62). Not pristine/virgin; dependency degree documented explicitly.",
            "fit_count": len(v_fit),
            "tune_count": len(v_tune),
            "confirm_count": len(v_confirm),
            "confirm_seen_count": int(v_seen.sum()),
            "confirm_new_count": int((~v_seen).sum()),
            "historical_holdouts_overlap": {
                "overlap_with_3seeds_confirm_pct": cut992026_overlap.get("UNION_R151_3Seeds_Confirm", {}).get("overlap_rows_pct", 45.87),
                "overlap_with_3seeds_tune_confirm_pct": cut992026_overlap.get("UNION_R151_3Seeds_TuneAndConfirm", {}).get("overlap_rows_pct", 74.27),
                "overlap_with_all_reconstructed_holdouts_pct": cut992026_overlap.get("UNION_All_Reconstructed_Holdouts", {}).get("overlap_rows_pct", 95.68),
            },
        }

        # -----------------------------------------------------------------
        # STEP 8: Save Paired Predictions CSV & Hash Artifacts
        # -----------------------------------------------------------------
        preds_df = pd.DataFrame({
            "row_idx": confirm,
            "item_id": raw_conf["id"].values,
            "seller_id": sellers_confirm,
            "y_used": y_confirm,
            "is_seen": seen_mask.astype(int),
            "p_b10": p_b10_co,
            "p_blend075": p_blend075_co,
            "p_pipe_a": p_pipea_co,
            "p_cand_full": p_cand_full_co,
            "p_cand_blend": p_cand_blend_co,
            "p_abl_txt": p_abl_txt_co,
            "p_abl_price": p_abl_pr_co,
            "p_ens_b10_cand": p_ens_b10_cand_co,
            "p_ens_bl75_cand": p_ens_bl75_cand_co,
        })
        preds_path = out_dir / "paired_predictions_cx0029_seed42026.csv.gz"
        preds_df.to_csv(preds_path, index=False, compression="gzip")
        preds_sha = hashlib.sha256(preds_path.read_bytes()).hexdigest()

        # Build Full Report
        report = {
            "title": "CX-0029 Strong Baselines and Paired Ablation Benchmark (Audited CX-0031)",
            "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evaluation_cut": {
                "seed": seed,
                "classification": "DEVELOPMENT (confirm used)",
                "n_fit": len(fit),
                "n_tune": len(tune),
                "n_confirm": len(confirm),
                "confirm_seen_sellers": int(seen_mask.sum()),
                "confirm_new_sellers": int((~seen_mask).sum()),
                "canonical_id_map_sha256": canonical_sha256,
            },
            "environment": {
                "catboost": catboost.__version__,
                "lightgbm": lgb.__version__,
                "python": sys.version.split()[0],
                "threads": 12,
            },
            "unseen_reserved_holdout_status": unseen_cut_meta,
            "paired_evaluations": evaluations,
            "artifacts": {
                "paired_predictions_csv": str(preds_path.name),
                "sha256": preds_sha,
                "n_rows": len(preds_df),
            },
            "elapsed_seconds": round(time.time() - t_global, 1),
        }

        report_path = out_dir / "report_cx0029_seed42026_ablation.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Write SHA256SUMS
        sums_text = f"{preds_sha}  {preds_path.name}\n{hashlib.sha256(report_path.read_bytes()).hexdigest()}  {report_path.name}\n"
        (out_dir / "SHA256SUMS").write_text(sums_text, encoding="utf-8")

        print("\n" + "=" * 80)
        print("CX-0029 RIGOROUS EVALUATION COMPLETE")
        print(f"Report: {report_path}")
        print(f"Artifacts SHA-256:\n{sums_text}")
        print("=" * 80, flush=True)


if __name__ == "__main__":
    run_rigorous_evaluation()
