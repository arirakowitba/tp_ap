"""Execute pristine audit pipeline: Fit (46k) -> Tune (12k) -> Frozen Virgin Confirm (12k).

Author: Antigravity
Date: 2026-11-09
Protocol: Zero-leakage, dual-regime, strong EVT-132 analogue comparison, frozen single-rule confirm.
"""

import json
import time
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
from catboost import CatBoostClassifier

ROOT = Path(".")
DATA_DIR = ROOT / "data/ap_2026_q_2/raw"
OUT_DIR = ROOT / "results/pristine_audit_20261109"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PUB = 20998
N_TEST = 30000
BASE_HITS = 19896  # EVT-132 anchor hits
GAMMA = 2.955747   # Prequential stress factor


def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def build_enriched(row):
    parts = [str(row.get("title") or "")]
    w = row.get("warranty")
    if w and str(w).strip():
        parts.append(f"garantia_{w}")
    tags = row.get("tags") or []
    for t in tags:
        parts.append(f"tag_{t}")
    attrs = row.get("attributes") or []
    for a in attrs:
        if isinstance(a, dict):
            name = a.get("id") or a.get("name") or ""
            val = a.get("value_name") or a.get("value_id") or ""
            if name or val:
                parts.append(f"attr_{name}_{val}")
    return " ".join(parts).lower()


def main():
    print("=" * 90)
    print("PRISTINE AUDIT PIPELINE: STRONG EVT-132 ANALOGUE & VIRGIN CONFIRM")
    print("=" * 90, flush=True)

    t0 = time.time()

    # 1. Load Partition Indices
    with open(OUT_DIR / "pristine_partitions.json", "r") as f:
        idx_data = json.load(f)
    fit_idx = np.array(idx_data["fit_idx"])
    tune_idx = np.array(idx_data["tune_idx"])
    virgin_confirm_idx = np.array(idx_data["virgin_confirm_idx"])
    seen_confirm_idx = np.array(idx_data["seen_confirm_idx"])
    unseen_confirm_idx = np.array(idx_data["unseen_confirm_idx"])

    print(f"Loaded partitions: Fit={len(fit_idx)}, Tune={len(tune_idx)}, Virgin Confirm={len(virgin_confirm_idx)}")

    # 2. Load Datasets
    print("\n[1/6] Loading Datasets and Strong EVT-132 Analogue...")
    df_train = pd.read_json(DATA_DIR / "train_data.jsonlines", lines=True)
    df_test = pd.read_json(DATA_DIR / "test_data.jsonlines", lines=True)

    # Load strong EVT-132 OOF baseline
    p_podium_oof = ROOT / "results/ultimate_podium_engine_23h/oof_predictions_podium_engine.csv"
    df_podium_oof = pd.read_csv(p_podium_oof)
    base_pred_all = (df_podium_oof["oof_blend"].values >= 0.5).astype(np.int8)
    base_prob_all = df_podium_oof["oof_blend"].values

    # Test baseline
    p_podium_test = ROOT / "results/ultimate_podium_engine_23h/test_probabilities_podium_engine.csv"
    df_podium_test = pd.read_csv(p_podium_test)
    base_test_prob = df_podium_test["p_optimal_blend"].values
    base_test_pred = (base_test_prob >= 0.5).astype(np.int8)

    y_all = (df_train["condition"] == "used").astype(np.int8).values
    y_fit = y_all[fit_idx]
    y_tune = y_all[tune_idx]
    y_confirm = y_all[virgin_confirm_idx]

    df_fit = df_train.iloc[fit_idx].reset_index(drop=True)
    df_tune = df_train.iloc[tune_idx].reset_index(drop=True)
    df_confirm = df_train.iloc[virgin_confirm_idx].reset_index(drop=True)

    acc_base_tune = accuracy_score(y_tune, base_pred_all[tune_idx])
    acc_base_confirm = accuracy_score(y_confirm, base_pred_all[virgin_confirm_idx])
    print(f"  Strong EVT-132 Analogue Baseline Accuracy:")
    print(f"    Tune Pool (12,000):           {acc_base_tune:.5f} ({np.sum(base_pred_all[tune_idx] == y_tune)} / {len(tune_idx)})")
    print(f"    Virgin Confirm (12,000):      {acc_base_confirm:.5f} ({np.sum(base_pred_all[virgin_confirm_idx] == y_confirm)} / {len(virgin_confirm_idx)})")

    # 3. Extract Fold-Local Enriched Text Features on Fit Pool
    print("\n[2/6] Building Enriched Multimodal Text Representations...")
    texts_fit = df_fit.apply(build_enriched, axis=1)
    texts_tune = df_tune.apply(build_enriched, axis=1)
    texts_confirm = df_confirm.apply(build_enriched, axis=1)
    texts_test = df_test.apply(build_enriched, axis=1)

    print("  Fitting 5-fold cross-validation Ridge Text on 46k Fit Pool...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_ridge_fit = np.zeros(len(df_fit), dtype=np.float32)

    vec_word = TfidfVectorizer(max_features=35000, ngram_range=(1, 2), sublinear_tf=True)
    vec_char = TfidfVectorizer(max_features=25000, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(df_fit, y_fit, df_fit["seller_id"].values)):
        w_tr = vec_word.fit_transform(texts_fit.iloc[tr_idx])
        c_tr = vec_char.fit_transform(texts_fit.iloc[tr_idx])
        X_tr = sparse.hstack([w_tr, c_tr], format="csr")

        w_va = vec_word.transform(texts_fit.iloc[va_idx])
        c_va = vec_char.transform(texts_fit.iloc[va_idx])
        X_va = sparse.hstack([w_va, c_va], format="csr")

        m_rdg = RidgeClassifier(alpha=2.0, random_state=42)
        m_rdg.fit(X_tr, y_fit[tr_idx])
        scores = m_rdg.decision_function(X_va)
        oof_ridge_fit[va_idx] = 1.0 / (1.0 + np.exp(-scores))

    print(f"  OOF Ridge Acc on Fit Pool: {accuracy_score(y_fit, (oof_ridge_fit >= 0.5)):.5f} | LogLoss: {log_loss(y_fit, oof_ridge_fit):.5f}")

    # Full fit of Ridge on Fit Pool
    w_fit = vec_word.fit_transform(texts_fit)
    c_fit = vec_char.fit_transform(texts_fit)
    X_fit_full = sparse.hstack([w_fit, c_fit], format="csr")

    m_rdg_full = RidgeClassifier(alpha=2.0, random_state=42)
    m_rdg_full.fit(X_fit_full, y_fit)

    # Transform Tune, Confirm, Test
    X_tune_sparse = sparse.hstack([vec_word.transform(texts_tune), vec_char.transform(texts_tune)], format="csr")
    p_tune_ridge = 1.0 / (1.0 + np.exp(-m_rdg_full.decision_function(X_tune_sparse)))

    X_confirm_sparse = sparse.hstack([vec_word.transform(texts_confirm), vec_char.transform(texts_confirm)], format="csr")
    p_confirm_ridge = 1.0 / (1.0 + np.exp(-m_rdg_full.decision_function(X_confirm_sparse)))

    X_test_sparse = sparse.hstack([vec_word.transform(texts_test), vec_char.transform(texts_test)], format="csr")
    p_test_ridge = 1.0 / (1.0 + np.exp(-m_rdg_full.decision_function(X_test_sparse)))

    # 4. Train Meta-Stackers Conditioned on Regime
    print("\n[3/6] Training Regime-Conditioned Meta-Stacker on Fit Pool...")
    # Base model predictions on fit pool from podium engine
    oof_lgb_fit = df_podium_oof.iloc[fit_idx]["oof_lgb"].values
    oof_cb_fit = df_podium_oof.iloc[fit_idx]["oof_cb"].values

    seller_counts_fit = df_fit["seller_id"].value_counts()
    is_multi_fit = df_fit["seller_id"].map(seller_counts_fit > 1).values

    meta_X_fit = np.column_stack([oof_cb_fit, oof_lgb_fit, oof_ridge_fit])
    meta_seen = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200, random_state=42)
    meta_seen.fit(meta_X_fit[is_multi_fit], y_fit[is_multi_fit])
    print(f"  Meta-Seen Weights [CB, LGB, EnrichedRidge]: {meta_seen.coef_[0].round(3)} | Intercept: {meta_seen.intercept_[0]:.3f}")

    meta_unseen = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200, random_state=42)
    meta_unseen.fit(meta_X_fit[~is_multi_fit], y_fit[~is_multi_fit])
    print(f"  Meta-Unseen Weights [CB, LGB, EnrichedRidge]: {meta_unseen.coef_[0].round(3)} | Intercept: {meta_unseen.intercept_[0]:.3f}")

    # Predict Meta on Tune Pool
    oof_lgb_tune = df_podium_oof.iloc[tune_idx]["oof_lgb"].values
    oof_cb_tune = df_podium_oof.iloc[tune_idx]["oof_cb"].values
    meta_X_tune = np.column_stack([oof_cb_tune, oof_lgb_tune, p_tune_ridge])

    is_seen_tune = np.isin(df_tune["seller_id"].values, df_fit["seller_id"].unique())
    p_tune_meta = np.zeros(len(df_tune), dtype=np.float32)
    p_tune_meta[is_seen_tune] = meta_seen.predict_proba(meta_X_tune[is_seen_tune])[:, 1]
    p_tune_meta[~is_seen_tune] = meta_unseen.predict_proba(meta_X_tune[~is_seen_tune])[:, 1]

    # 5. Tune Margin delta ONCE on Tune Pool (DO NOT TOUCH VIRGIN CONFIRM)
    print("\n[4/6] Tuning Margin delta on Tune Pool (against Strong EVT-132 Analogue)...")
    base_tune_pred = base_pred_all[tune_idx]
    sellers_tune = df_tune["seller_id"].values
    unique_sellers_tune = np.unique(sellers_tune)
    seller_to_idx_tune = {s: np.where(sellers_tune == s)[0] for s in unique_sellers_tune}

    margins = [0.00, 0.05, 0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]
    best_delta = 0.0
    best_adj_p5 = -999.0
    rng = np.random.RandomState(42)
    B = 2000

    print(f"{'Margin':6s} | {'Flips':6s} | {'Fixes':5s} | {'Breaks':6s} | {'Net':5s} | {'Prec%':6s} | {'Raw P5':7s} | {'Raw P50':7s} | {'Adj P5':7s}")
    print("-" * 75)

    for m in margins:
        cand_tune = base_tune_pred.copy()
        flip_mask = (np.abs(p_tune_meta - 0.5) >= m) & ((p_tune_meta >= 0.5).astype(np.int8) != base_tune_pred)
        cand_tune[flip_mask] = (p_tune_meta[flip_mask] >= 0.5).astype(np.int8)

        fixes = int(np.sum((base_tune_pred != y_tune) & (cand_tune == y_tune)))
        breaks = int(np.sum((base_tune_pred == y_tune) & (cand_tune != y_tune)))
        net = fixes - breaks
        prec = (fixes / max(fixes + breaks, 1)) * 100.0

        item_gains = ((base_tune_pred != y_tune) & (cand_tune == y_tune)).astype(np.int8) - \
                     ((base_tune_pred == y_tune) & (cand_tune != y_tune)).astype(np.int8)

        boot_gains = np.zeros(B, dtype=np.float32)
        for b in range(B):
            s_sellers = rng.choice(unique_sellers_tune, size=len(unique_sellers_tune), replace=True)
            s_idx = np.concatenate([seller_to_idx_tune[s] for s in s_sellers])
            boot_gains[b] = np.sum(item_gains[s_idx]) * (N_PUB / len(s_idx))

        p5_raw = float(np.percentile(boot_gains, 5))
        p50_raw = float(np.percentile(boot_gains, 50))
        p95_raw = float(np.percentile(boot_gains, 95))
        half_width_adj = ((p95_raw - p5_raw) / 2.0) * GAMMA
        p5_adj = p50_raw - half_width_adj

        print(f"{m:6.2f} | {int(np.sum(flip_mask)):6d} | {fixes:5d} | {breaks:6d} | {net:+5d} | {prec:5.1f}% | {p5_raw:+7.1f} | {p50_raw:+7.1f} | {p5_adj:+7.1f}")

        if p5_adj > best_adj_p5:
            best_adj_p5 = p5_adj
            best_delta = m

    print("-" * 75)
    print(f"Optimal Rule Frozen from Tune Pool: delta* = {best_delta:.2f} (Adj P5 on Tune = {best_adj_p5:+.1f})")

    # 6. OPEN VIRGIN CONFIRM EXACTLY ONCE WITH FROZEN RULE
    print("\n[5/6] OPENING VIRGIN CONFIRM (11,999 items) EXACTLY ONCE WITH FROZEN delta* = {:.2f}...".format(best_delta))
    oof_lgb_confirm = df_podium_oof.iloc[virgin_confirm_idx]["oof_lgb"].values
    oof_cb_confirm = df_podium_oof.iloc[virgin_confirm_idx]["oof_cb"].values
    meta_X_confirm = np.column_stack([oof_cb_confirm, oof_lgb_confirm, p_confirm_ridge])

    is_seen_confirm = np.isin(df_confirm["seller_id"].values, df_fit["seller_id"].unique())
    p_confirm_meta = np.zeros(len(df_confirm), dtype=np.float32)
    p_confirm_meta[is_seen_confirm] = meta_seen.predict_proba(meta_X_confirm[is_seen_confirm])[:, 1]
    p_confirm_meta[~is_seen_confirm] = meta_unseen.predict_proba(meta_X_confirm[~is_seen_confirm])[:, 1]

    base_confirm_pred = base_pred_all[virgin_confirm_idx]
    cand_confirm = base_confirm_pred.copy()
    flip_confirm = (np.abs(p_confirm_meta - 0.5) >= best_delta) & ((p_confirm_meta >= 0.5).astype(np.int8) != base_confirm_pred)
    cand_confirm[flip_confirm] = (p_confirm_meta[flip_confirm] >= 0.5).astype(np.int8)

    fixes_confirm = int(np.sum((base_confirm_pred != y_confirm) & (cand_confirm == y_confirm)))
    breaks_confirm = int(np.sum((base_confirm_pred == y_confirm) & (cand_confirm != y_confirm)))
    net_confirm = fixes_confirm - breaks_confirm
    prec_confirm = (fixes_confirm / max(fixes_confirm + breaks_confirm, 1)) * 100.0

    print(f"  Virgin Confirm Flips:   {int(np.sum(flip_confirm)):4d}")
    print(f"  Virgin Confirm Fixes:   {fixes_confirm:4d}")
    print(f"  Virgin Confirm Breaks:  {breaks_confirm:4d}")
    print(f"  Virgin Confirm Net:     {net_confirm:+4d}")
    print(f"  Virgin Confirm Prec:    {prec_confirm:5.2f}% ({fixes_confirm/max(breaks_confirm, 1):.2f} fixes per break)")

    # Clustered Bootstrap on Virgin Confirm
    sellers_confirm = df_confirm["seller_id"].values
    unique_sellers_confirm = np.unique(sellers_confirm)
    seller_to_idx_confirm = {s: np.where(sellers_confirm == s)[0] for s in unique_sellers_confirm}

    item_gains_confirm = ((base_confirm_pred != y_confirm) & (cand_confirm == y_confirm)).astype(np.int8) - \
                         ((base_confirm_pred == y_confirm) & (cand_confirm != y_confirm)).astype(np.int8)

    boot_gains_confirm = np.zeros(B, dtype=np.float32)
    for b in range(B):
        s_sellers = rng.choice(unique_sellers_confirm, size=len(unique_sellers_confirm), replace=True)
        s_idx = np.concatenate([seller_to_idx_confirm[s] for s in s_sellers])
        boot_gains_confirm[b] = np.sum(item_gains_confirm[s_idx]) * (N_PUB / len(s_idx))

    p1_raw = float(np.percentile(boot_gains_confirm, 1))
    p5_raw = float(np.percentile(boot_gains_confirm, 5))
    p50_raw = float(np.percentile(boot_gains_confirm, 50))
    p95_raw = float(np.percentile(boot_gains_confirm, 95))

    half_width_adj = ((p95_raw - p5_raw) / 2.0) * GAMMA
    p5_adj = p50_raw - half_width_adj
    p1_adj = p50_raw - (half_width_adj / 1.645) * 2.326

    sigma_adj = max(half_width_adj / 1.645, 0.01)
    prob_53 = float(1.0 - norm.cdf((53.0 - p50_raw) / sigma_adj))
    prob_podium = float(1.0 - norm.cdf((31.0 - p50_raw) / sigma_adj))

    score_p50 = (BASE_HITS + p50_raw) / N_PUB
    score_p5_adj = (BASE_HITS + p5_adj) / N_PUB
    score_p1_adj = (BASE_HITS + p1_adj) / N_PUB

    print("\n--- FINAL VIRGIN CONFIRM CERTIFICATE ---")
    print(f"  Raw P5:                  {p5_raw:+.1f} hits | Raw P50: {p50_raw:+.1f} hits | Raw P95: {p95_raw:+.1f} hits")
    print(f"  P5 Stressed (gamma=2.96):{p5_adj:+.1f} hits (Target: >= +53.0)")
    print(f"  P1 Stressed (gamma=2.96):{p1_adj:+.1f} hits")
    print(f"  Score Guaranteed 95%:    {score_p5_adj:.5f}")
    print(f"  Score Guaranteed 99%:    {score_p1_adj:.5f}")
    print(f"  Expected Public Score:   {score_p50:.5f}")
    print(f"  P(Gain >= +53 | Stress): {prob_53*100:.2f}% (Target: >= 95.0%)")
    print(f"  P(Gain >= +31 Podium):   {prob_podium*100:.2f}%")

    status = "QUALIFIED" if (p5_adj >= 53.0 and prob_53 >= 0.95) else "NO_QUALIFY"
    print(f"\nFINAL VERDICT: >>> {status} <<<")

    # 7. Generate Test Submission with Frozen Rule
    print("\n[6/6] Generating Test Submission with Frozen delta*...")
    test_lgb = df_podium_test["p_lgb"].values
    test_cb = df_podium_test["p_cb"].values
    meta_X_test = np.column_stack([test_cb, test_lgb, p_test_ridge])

    is_seen_test = np.isin(df_test["seller_id"].values, df_fit["seller_id"].unique())
    p_test_meta = np.zeros(len(df_test), dtype=np.float32)
    p_test_meta[is_seen_test] = meta_seen.predict_proba(meta_X_test[is_seen_test])[:, 1]
    p_test_meta[~is_seen_test] = meta_unseen.predict_proba(meta_X_test[~is_seen_test])[:, 1]

    # Load EVT-132 test predictions
    p_evt = ROOT / "submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv"
    df_evt = pd.read_csv(p_evt)
    y_evt_test = (df_evt["condition"] == "used").astype(np.int8).values

    cand_test = y_evt_test.copy()
    flip_test = (np.abs(p_test_meta - 0.5) >= best_delta) & ((p_test_meta >= 0.5).astype(np.int8) != y_evt_test)
    cand_test[flip_test] = (p_test_meta[flip_test] >= 0.5).astype(np.int8)

    print(f"  Test Flips against EVT-132: {int(np.sum(flip_test)):4d}")

    df_cand = pd.DataFrame({
        "ID": np.arange(1, len(df_test) + 1),
        "condition": np.where(cand_test == 1, "used", "new")
    })
    csv_path = OUT_DIR / "candidate_pristine_audit_final.csv"
    df_cand.to_csv(csv_path, index=False)
    c_hash = sha256_file(csv_path)
    print(f"  Saved Final Candidate: {csv_path}")
    print(f"  SHA-256: {c_hash}")

    # Save Certificate
    cert_data = {
        "certificate_id": "CERT-20261109-PRISTINE-FINAL-AP2026Q2",
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "verdict": status,
        "delta_frozen": best_delta,
        "strong_baseline": "oof_predictions_podium_engine.csv (OOF 0.94230)",
        "virgin_confirm": {
            "total_items": len(virgin_confirm_idx),
            "seen_items": len(seen_confirm_idx),
            "unseen_items": len(unseen_confirm_idx),
            "flips": int(np.sum(flip_confirm)),
            "fixes": fixes_confirm,
            "breaks": breaks_confirm,
            "net": net_confirm,
            "precision_percent": prec_confirm,
        },
        "uncertainty_stressed": {
            "p5_adj": p5_adj,
            "p1_adj": p1_adj,
            "score_p5_adj": score_p5_adj,
            "score_p1_adj": score_p1_adj,
            "score_p50": score_p50,
            "prob_gain_ge_53": prob_53,
            "prob_podium": prob_podium,
        },
        "candidate": {
            "csv_path": str(csv_path),
            "sha256": c_hash,
            "flips_vs_evt132": int(np.sum(flip_test)),
        }
    }
    cert_path = OUT_DIR / "pristine_audit_certificate.json"
    with open(cert_path, "w") as f:
        json.dump(cert_data, f, indent=2)
    print(f"  Saved Certificate: {cert_path}")
    print(f"\nExecution finished in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
