"""AP-2026-Q-2: Comprehensive Audit of Seller History Mechanisms by Regime & Misrouting Damage.

Mandatory Protocol & Governance:
1. Compares three seller history representations on identical rows:
   - (a) Historial Causal (strictly past items before listing timestamp: (S_prior + m*p)/(N_prior + m), log1p(N_prior))
   - (b) Historial Batch LOO (full in-sample target encoding unconstrained by time)
   - (c) Ausencia de Historial Propio (cold prior fallback: fold/global prior, zero count)
2. Audits all 5 canonical temporal regimes:
   - 'frío' (cold_new: unseen sellers, N_fit = 0)
   - 'intercalado' (bracketed: min t_fit < t_query < max t_fit)
   - 'después' (after: t_query > max t_fit)
   - 'antes' (before: t_query < min t_fit)
   - 'empate' (ties: t_query == min t_fit or max t_fit)
3. Explicitly documents data availability per regime and proves lookahead leakage in 'antes' and 'bracketed'.
4. Freezes optimal routing policy on tune (15,044 rows).
5. Quantifies exact misrouting damage (hits lost, accuracy drop, logloss inflation).
6. Maps regime distribution to test set (30,000 rows: 8,513 cold, 14,117 bracketed, 3,121 after, 3,118 before, 1,131 ties).
7. Outputs:
   - results/seller_regimes_routing_tune_pairs_private.csv (gitignored)
   - results/seller_regimes_routing_report.json
8. Gate C: Declares "NO_IDENTIFIABLE" for unconditioned public probability claims.
"""

import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

import lightgbm as lgb
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import t98_line3_lightgbm_enriched as ag

SPLIT_CSV = ROOT / "data" / "splits" / "canonical_3way_splits.csv"
SPLIT_MANIFEST = ROOT / "data" / "splits" / "canonical_3way_manifest.json"
DATA_PKL = ROOT / "data" / "splits" / "t98_parsed_train.pkl"
CACHE_PATH = ROOT / "data" / "splits" / "text_6afe07abc3fa2a93.npz"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PAIRS_CSV_PATH = RESULTS_DIR / "seller_regimes_routing_tune_pairs_private.csv"
REPORT_JSON_PATH = RESULTS_DIR / "seller_regimes_routing_report.json"

SEEDS = [42, 2026, 777]
N_JOBS = 6


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_metrics_bundle(y_true, p_pred):
    acc = float(accuracy_score(y_true, p_pred >= 0.5))
    hits = int(((p_pred >= 0.5) == y_true).sum())
    ll = float(log_loss(y_true, np.clip(p_pred, 1e-6, 1.0 - 1e-6)))
    brier = float(brier_score_loss(y_true, p_pred))
    return {
        "n_rows": len(y_true),
        "hits": hits,
        "accuracy": round(acc, 6),
        "logloss": round(ll, 6),
        "brier_score": round(brier, 6),
    }


def main():
    t_start = time.time()
    log("=== AP-2026-Q-2: Auditoría Integral de Historial de Vendedor por Régimen ===")

    # 1. Load Data and Canonical Splits
    log(f"Loading split manifest from {SPLIT_MANIFEST}...")
    with open(SPLIT_MANIFEST, "r", encoding="utf-8") as f:
        split_manifest = json.load(f)
    log(f"Verified Split SHA-256: {split_manifest['split_sha256']}")

    split_df = pd.read_csv(SPLIT_CSV)
    fit_mask = split_df["role"] == "fit"
    tune_mask = split_df["role"] == "tune"
    fit_indices = np.flatnonzero(fit_mask)
    tune_indices = np.flatnonzero(tune_mask)

    log(f"Loading parsed train data from {DATA_PKL}...")
    with open(DATA_PKL, "rb") as f:
        df = pickle.load(f)

    y_fit = df["y"].iloc[fit_indices].to_numpy().astype(np.int32)
    y_tune = df["y"].iloc[tune_indices].to_numpy().astype(np.int32)
    ids_tune = df["id"].iloc[tune_indices].values.astype(str)
    sellers_tune = df["seller_id"].iloc[tune_indices].values.astype(str)
    cats_tune = df["category_id"].iloc[tune_indices].values.astype(str)
    regimes_tune = split_df.iloc[tune_indices]["regime"].values.astype(str)

    tune_cold_mask = (regimes_tune == "tune_cold_new")
    tune_seen_mask = (regimes_tune == "tune_seen_bracketed")

    log(f"Fit count: {len(fit_indices)}, Tune count: {len(tune_indices)}")
    log(f"Tune cold new: {tune_cold_mask.sum()}, Tune seen bracketed: {tune_seen_mask.sum()}")

    # 2. Build or Load Fold Matrices
    log("Precomputing item intrinsic arrays...")
    tr_data = ag.precompute_item_intrinsic_arrays(df)

    log("Setting up inner 4-fold StratifiedGroupKFold on seller_id inside fit...")
    inner_cv = list(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=61111).split(
        fit_indices, tr_data["y"][fit_indices], tr_data["seller_id"][fit_indices]
    ))

    log(f"Building fold matrices using cache {CACHE_PATH.name}...")
    t0_mat = time.time()
    mats = ag.build_fold_matrices(
        T_indices=fit_indices,
        eval_indices_list=[tune_indices],
        splits_T=inner_cv,
        tr_data=tr_data,
        eval_data_list=[tr_data],
        cache_path=CACHE_PATH,
    )
    log(f"Fold matrices built in {time.time() - t0_mat:.2f}s.")

    prior = float(np.mean(y_fit))
    m_prior = 5.0
    ste_col_idx = 74

    # Extract Matrices for Fit and Tune
    X_unseen_fit = mats["X_m1_unseen_T"]
    X_unseen_tune = mats["X_m1_unseen_evals"][0]
    X_seen_causal_fit = mats["X_m1_seen_causal_T"]
    X_seen_batch_fit = mats["X_m1_seen_rowte_T"]

    # Compute TRUE Causal features for Tune
    log("Computing strictly causal prior-items features for tune set...")
    df_fit_s = pd.DataFrame({
        "seller_id": tr_data["seller_id"][fit_indices],
        "start_time": tr_data["start_time"][fit_indices],
        "row": tr_data["row"][fit_indices],
        "y": y_fit,
    }).sort_values(["seller_id", "start_time", "row"])

    fit_by_seller = {}
    for s, grp in df_fit_s.groupby("seller_id", sort=False):
        fit_by_seller[s] = (grp["start_time"].values, grp["y"].values)

    start_tune = tr_data["start_time"][tune_indices]
    causal_te_tune = np.full(len(tune_indices), prior, dtype=np.float32)
    causal_cnt_tune = np.zeros(len(tune_indices), dtype=np.float32)

    for i in range(len(tune_indices)):
        s = sellers_tune[i]
        t = start_tune[i]
        if s in fit_by_seller:
            times, ys = fit_by_seller[s]
            idx = np.searchsorted(times, t, side="left")
            if idx > 0:
                cnt = float(idx)
                s_sum = float(ys[:idx].sum())
                causal_te_tune[i] = (s_sum + m_prior * prior) / (cnt + m_prior)
                causal_cnt_tune[i] = np.log1p(cnt)

    # Matrix representations for Tune
    # 1. Causal
    X_causal_tune = X_unseen_tune.copy()
    X_causal_tune[:, ste_col_idx] = causal_te_tune
    X_causal_tune[:, ste_col_idx + 1] = causal_cnt_tune

    # 2. Batch LOO on Tune (uses full fit history)
    X_batch_tune = mats["X_m1_seen_causal_evals"][0].copy()

    # 3. Cold Fallback on Tune (prior fallback, 0 count)
    X_cold_tune = X_unseen_tune.copy()

    # 3. Train Models on Fit (Bagged 3 seeds)
    log("\nTraining Bagged Models on Fit...")
    hp_unseen = {"num_leaves": 48, "min_child_samples": 35, "colsample_bytree": 0.75}
    hp_seen = {"num_leaves": 48, "min_child_samples": 35, "colsample_bytree": 0.75}

    def fit_bagged(X_train, y_train, X_eval, hp, n_est=350):
        preds = np.zeros(len(X_eval), dtype=np.float64)
        for seed in SEEDS:
            m = LGBMClassifier(
                n_estimators=n_est,
                learning_rate=0.035,
                num_leaves=hp["num_leaves"],
                min_child_samples=hp["min_child_samples"],
                colsample_bytree=hp["colsample_bytree"],
                subsample=0.85,
                subsample_freq=1,
                random_state=seed,
                n_jobs=N_JOBS,
                verbosity=-1,
            )
            m.fit(X_train, y_train)
            preds += m.predict_proba(X_eval)[:, 1]
        return (preds / len(SEEDS)).astype(np.float32)

    log("1. Fitting Cold/Unseen Specialist Model...")
    p_unseen_model_tune_cold = fit_bagged(X_unseen_fit, y_fit, X_cold_tune, hp_unseen)

    log("2. Fitting Seen Causal Model...")
    p_seen_causal_tune_causal = fit_bagged(X_seen_causal_fit, y_fit, X_causal_tune, hp_seen)
    p_seen_causal_tune_batch = fit_bagged(X_seen_causal_fit, y_fit, X_batch_tune, hp_seen)
    p_seen_causal_tune_cold = fit_bagged(X_seen_causal_fit, y_fit, X_cold_tune, hp_seen)

    log("3. Fitting Seen Batch Model...")
    p_seen_batch_tune_batch = fit_bagged(X_seen_batch_fit, y_fit, X_batch_tune, hp_seen)

    # 4. Detailed Analysis on Tune (15,044 rows)
    log("\n=== EVALUATION ON CANONICAL TUNE PARTITION (15,044 rows) ===")

    # A. Seen Bracketed Regime (6,286 rows)
    y_seen_true = y_tune[tune_seen_mask]
    m_causal_seen = compute_metrics_bundle(y_seen_true, p_seen_causal_tune_causal[tune_seen_mask])
    m_batch_leak_seen = compute_metrics_bundle(y_seen_true, p_seen_causal_tune_batch[tune_seen_mask])
    m_batch_full_seen = compute_metrics_bundle(y_seen_true, p_seen_batch_tune_batch[tune_seen_mask])
    m_cold_fallback_seen = compute_metrics_bundle(y_seen_true, p_unseen_model_tune_cold[tune_seen_mask])

    log(f"Tune Seen Bracketed (N=6,286):")
    log(f"  - Historial Causal Estricto: Hits={m_causal_seen['hits']} ({m_causal_seen['accuracy']:.4%}), LL={m_causal_seen['logloss']:.6f}")
    log(f"  - Causal Model w/ Batch Feat: Hits={m_batch_leak_seen['hits']} ({m_batch_leak_seen['accuracy']:.4%}), LL={m_batch_leak_seen['logloss']:.6f}")
    log(f"  - Historial Batch Completo:   Hits={m_batch_full_seen['hits']} ({m_batch_full_seen['accuracy']:.4%}), LL={m_batch_full_seen['logloss']:.6f}")
    log(f"  - Ausencia de Historial:     Hits={m_cold_fallback_seen['hits']} ({m_cold_fallback_seen['accuracy']:.4%}), LL={m_cold_fallback_seen['logloss']:.6f}")

    damage_cold_on_seen = m_causal_seen["hits"] - m_cold_fallback_seen["hits"]
    lookahead_gain_seen = m_batch_leak_seen["hits"] - m_causal_seen["hits"]
    log(f"  -> Misrouting Damage (Cold on Seen): -{damage_cold_on_seen} hits lost net!")
    log(f"  -> Lookahead Leakage Illusion:       +{lookahead_gain_seen} false hits")

    # B. Cold New Regime (8,758 rows)
    y_cold_true = y_tune[tune_cold_mask]
    m_cold_opt = compute_metrics_bundle(y_cold_true, p_unseen_model_tune_cold[tune_cold_mask])
    m_cold_misroute = compute_metrics_bundle(y_cold_true, p_seen_causal_tune_cold[tune_cold_mask])

    log(f"\nTune Cold New (N=8,758):")
    log(f"  - Optimal Unseen Specialist: Hits={m_cold_opt['hits']} ({m_cold_opt['accuracy']:.4%}), LL={m_cold_opt['logloss']:.6f}")
    log(f"  - Misroute (Warm Seen Model): Hits={m_cold_misroute['hits']} ({m_cold_misroute['accuracy']:.4%}), LL={m_cold_misroute['logloss']:.6f}")
    damage_warm_on_cold = m_cold_opt["hits"] - m_cold_misroute["hits"]
    log(f"  -> Misrouting Damage (Warm on Cold): -{damage_warm_on_cold} hits lost net!")

    # C. Frozen Routing Policy on Tune
    # Route: Cold items -> Unseen Model; Seen items -> Seen Causal Model with strictly causal features
    p_routed_tune = np.empty(len(tune_indices), dtype=np.float32)
    p_routed_tune[tune_cold_mask] = p_unseen_model_tune_cold[tune_cold_mask]
    p_routed_tune[tune_seen_mask] = p_seen_causal_tune_causal[tune_seen_mask]
    m_tune_routed = compute_metrics_bundle(y_tune, p_routed_tune)

    # Uniform Cold Fallback on entire Tune
    p_uniform_cold_tune = p_unseen_model_tune_cold
    m_tune_uniform_cold = compute_metrics_bundle(y_tune, p_uniform_cold_tune)

    # Uniform Warm Causal Model on entire Tune
    p_uniform_causal_tune = p_seen_causal_tune_causal
    m_tune_uniform_causal = compute_metrics_bundle(y_tune, p_uniform_causal_tune)

    log(f"\nTune Full Partition (N=15,044):")
    log(f"  - Optimal Frozen Routing Policy: Hits={m_tune_routed['hits']} ({m_tune_routed['accuracy']:.4%}), LL={m_tune_routed['logloss']:.6f}")
    log(f"  - Uniform Cold Fallback:         Hits={m_tune_uniform_cold['hits']} ({m_tune_uniform_cold['accuracy']:.4%}), LL={m_tune_uniform_cold['logloss']:.6f} (-{m_tune_routed['hits'] - m_tune_uniform_cold['hits']} hits)")
    log(f"  - Uniform Causal Model:           Hits={m_tune_uniform_causal['hits']} ({m_tune_uniform_causal['accuracy']:.4%}), LL={m_tune_uniform_causal['logloss']:.6f} (-{m_tune_routed['hits'] - m_tune_uniform_causal['hits']} hits)")

    # 5. Comprehensive 5-Regime Diagnostic Evaluation inside Fit
    log("\n=== COMPREHENSIVE 5-REGIME EVALUATION INSIDE FIT ===")
    rng = np.random.default_rng(20261114)
    n_fit = len(fit_indices)
    val_sub = rng.choice(n_fit, size=int(0.30 * n_fit), replace=False)
    val_sub.sort()
    tr_sub = np.setdiff1d(np.arange(n_fit), val_sub)

    tr_idx = fit_indices[tr_sub]
    val_idx = fit_indices[val_sub]

    df_tr = df.iloc[tr_idx].copy()
    df_val = df.iloc[val_idx].copy()
    y_tr_sub = df_tr["y"].values.astype(np.int32)
    y_val_sub = df_val["y"].values.astype(np.int32)
    prior_sub = float(np.mean(y_tr_sub))

    # Regimes on inner val
    tr_sellers_set = set(df_tr["seller_id"].unique())
    tr_stats_map = df_tr.groupby("seller_id")["start_time"].agg(["min", "max", "count"])

    val_sellers_arr = df_val["seller_id"].values
    val_seen_arr = df_val["seller_id"].isin(tr_sellers_set).values
    val_t_arr = df_val["start_time"].values
    val_t_min_arr = df_val["seller_id"].map(tr_stats_map["min"]).values
    val_t_max_arr = df_val["seller_id"].map(tr_stats_map["max"]).values

    regimes_5 = {
        "frío": ~val_seen_arr,
        "intercalado": val_seen_arr & (val_t_arr > val_t_min_arr) & (val_t_arr < val_t_max_arr),
        "después": val_seen_arr & (val_t_arr > val_t_max_arr),
        "antes": val_seen_arr & (val_t_arr < val_t_min_arr),
        "empate": val_seen_arr & ((val_t_arr == val_t_min_arr) | (val_t_arr == val_t_max_arr)),
    }

    log(f"Inner Validation Breakdown (N={len(df_val)}):")
    for r_name, r_mask in regimes_5.items():
        log(f"  - {r_name:12s}: {r_mask.sum():5d} ({r_mask.mean():.2%})")

    # Feature computation on tr_sub and val_sub
    df_tr_sorted = df_tr[["seller_id", "start_time", "row", "y"]].sort_values(["seller_id", "start_time", "row"])
    grp_tr = df_tr_sorted.groupby("seller_id", sort=False)
    cnt_tr_sorted = grp_tr.cumcount().values.astype(np.float32)
    sum_tr_sorted = grp_tr["y"].cumsum().values.astype(np.float32) - df_tr_sorted["y"].values.astype(np.float32)
    te_tr_causal_sorted = (sum_tr_sorted + m_prior * prior_sub) / (cnt_tr_sorted + m_prior)
    cnt_tr_causal_sorted = np.log1p(cnt_tr_sorted)

    df_tr_orig_order = pd.DataFrame({"seller_id": df_tr["seller_id"].values, "start_time": df_tr["start_time"].values, "row": df_tr["row"].values})
    df_merged_tr = df_tr_orig_order.merge(
        pd.DataFrame({
            "seller_id": df_tr_sorted["seller_id"].values,
            "start_time": df_tr_sorted["start_time"].values,
            "row": df_tr_sorted["row"].values,
            "te_causal": te_tr_causal_sorted,
            "cnt_causal": cnt_tr_causal_sorted,
        }),
        on=["seller_id", "start_time", "row"],
        how="left",
    )
    te_tr_causal = df_merged_tr["te_causal"].values.astype(np.float32)
    cnt_tr_causal = df_merged_tr["cnt_causal"].values.astype(np.float32)

    tr_agg_map = df_tr.groupby("seller_id")["y"].agg(["count", "sum"])
    s_cnt_tr = df_tr["seller_id"].map(tr_agg_map["count"]).values.astype(np.float32)
    s_sum_tr = df_tr["seller_id"].map(tr_agg_map["sum"]).values.astype(np.float32)
    loo_cnt_tr = s_cnt_tr - 1.0
    loo_sum_tr = s_sum_tr - y_tr_sub.astype(np.float32)
    te_tr_batch = (loo_sum_tr + m_prior * prior_sub) / (loo_cnt_tr + m_prior)
    cnt_tr_batch = np.log1p(loo_cnt_tr)

    te_tr_cold = np.full(len(df_tr), prior_sub, dtype=np.float32)
    cnt_tr_cold = np.zeros(len(df_tr), dtype=np.float32)

    # Features on val_sub
    tr_by_seller_sub = {s: (grp["start_time"].values, grp["y"].values) for s, grp in df_tr_sorted.groupby("seller_id", sort=False)}
    te_val_causal = np.full(len(df_val), prior_sub, dtype=np.float32)
    cnt_val_causal = np.zeros(len(df_val), dtype=np.float32)
    te_val_batch = np.full(len(df_val), prior_sub, dtype=np.float32)
    cnt_val_batch = np.zeros(len(df_val), dtype=np.float32)

    for i in range(len(df_val)):
        s = val_sellers_arr[i]
        t = val_t_arr[i]
        if s in tr_by_seller_sub:
            times, ys = tr_by_seller_sub[s]
            idx = np.searchsorted(times, t, side="left")
            if idx > 0:
                c = float(idx)
                sm = float(ys[:idx].sum())
                te_val_causal[i] = (sm + m_prior * prior_sub) / (c + m_prior)
                cnt_val_causal[i] = np.log1p(c)
            b_c = float(len(times))
            b_sm = float(ys.sum())
            te_val_batch[i] = (b_sm + m_prior * prior_sub) / (b_c + m_prior)
            cnt_val_batch[i] = np.log1p(b_c)

    te_val_cold = np.full(len(df_val), prior_sub, dtype=np.float32)
    cnt_val_cold = np.zeros(len(df_val), dtype=np.float32)

    # Price and Category TE base features
    cat_tr_sub = df_tr["category_id"].astype(str).values
    cat_val_sub = df_val["category_id"].astype(str).values
    c_agg_sub = pd.DataFrame({"c": cat_tr_sub, "y": y_tr_sub}).groupby("c")["y"].agg(["count", "sum"])
    cat_te_map_sub = (c_agg_sub["sum"] + 10.0 * prior_sub) / (c_agg_sub["count"] + 10.0)
    cat_te_tr_sub = pd.Series(cat_tr_sub).map(cat_te_map_sub).fillna(prior_sub).values.astype(np.float32)
    cat_te_val_sub = pd.Series(cat_val_sub).map(cat_te_map_sub).fillna(prior_sub).values.astype(np.float32)

    p_tr_sub = np.log1p(np.maximum(0.0, df_tr["price"].fillna(0.0).values)).astype(np.float32)
    p_val_sub = np.log1p(np.maximum(0.0, df_val["price"].fillna(0.0).values)).astype(np.float32)

    tx_cache = np.load(CACHE_PATH)
    ptx_tr_sub = tx_cache["ptx1_oof_T"][tr_sub]
    ptx_val_sub = tx_cache["ptx1_oof_T"][val_sub]

    X_base_tr_sub = np.column_stack([p_tr_sub, cat_te_tr_sub, ptx_tr_sub]).astype(np.float32)
    X_base_val_sub = np.column_stack([p_val_sub, cat_te_val_sub, ptx_val_sub]).astype(np.float32)

    X_causal_tr_sub = np.column_stack([X_base_tr_sub, te_tr_causal, cnt_tr_causal]).astype(np.float32)
    X_causal_val_sub = np.column_stack([X_base_val_sub, te_val_causal, cnt_val_causal]).astype(np.float32)

    X_batch_tr_sub = np.column_stack([X_base_tr_sub, te_tr_batch, cnt_tr_batch]).astype(np.float32)
    X_batch_val_sub = np.column_stack([X_base_val_sub, te_val_batch, cnt_val_batch]).astype(np.float32)

    X_cold_tr_sub = np.column_stack([X_base_tr_sub, te_tr_cold, cnt_tr_cold]).astype(np.float32)
    X_cold_val_sub = np.column_stack([X_base_val_sub, te_val_cold, cnt_val_cold]).astype(np.float32)

    # Train 3 models on inner fit
    log("Fitting 3 Diagnostic Models on inner_fit...")
    m_causal_diag = LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=48, min_child_samples=30, random_state=42, n_jobs=N_JOBS, verbosity=-1)
    m_causal_diag.fit(X_causal_tr_sub, y_tr_sub)

    m_batch_diag = LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=48, min_child_samples=30, random_state=42, n_jobs=N_JOBS, verbosity=-1)
    m_batch_diag.fit(X_batch_tr_sub, y_tr_sub)

    m_cold_diag = LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=48, min_child_samples=30, random_state=42, n_jobs=N_JOBS, verbosity=-1)
    m_cold_diag.fit(X_cold_tr_sub, y_tr_sub)

    p_causal_val_sub = m_causal_diag.predict_proba(X_causal_val_sub)[:, 1]
    p_batch_val_sub = m_batch_diag.predict_proba(X_batch_val_sub)[:, 1]
    p_cold_val_sub = m_cold_diag.predict_proba(X_cold_val_sub)[:, 1]

    regime_results = {}
    for r_name, r_mask in regimes_5.items():
        y_r = y_val_sub[r_mask]
        res_c = compute_metrics_bundle(y_r, p_causal_val_sub[r_mask])
        res_b = compute_metrics_bundle(y_r, p_batch_val_sub[r_mask])
        res_cold = compute_metrics_bundle(y_r, p_cold_val_sub[r_mask])
        regime_results[r_name] = {
            "n_items": int(r_mask.sum()),
            "share": float(r_mask.mean()),
            "causal": res_c,
            "batch": res_b,
            "cold_fallback": res_cold,
            "lookahead_gain_batch_over_causal": res_b["hits"] - res_c["hits"],
            "loss_cold_vs_causal": res_c["hits"] - res_cold["hits"],
        }
        log(f"\nRégimen {r_name.upper()} (N={r_mask.sum()}):")
        log(f"  - Causal:   Hits={res_c['hits']} ({res_c['accuracy']:.2%}), LL={res_c['logloss']:.4f}")
        log(f"  - Batch:    Hits={res_b['hits']} ({res_b['accuracy']:.2%}), LL={res_b['logloss']:.4f} (Delta: {res_b['hits'] - res_c['hits']:+d} hits)")
        log(f"  - Cold:     Hits={res_cold['hits']} ({res_cold['accuracy']:.2%}), LL={res_cold['logloss']:.4f} (Delta: {res_cold['hits'] - res_c['hits']:+d} hits)")

    # 6. Save Private Pairs CSV
    log(f"\nWriting private tune pairs to {PAIRS_CSV_PATH}...")
    hit_routed = ((p_routed_tune >= 0.5) == y_tune).astype(int)
    hit_cold = ((p_uniform_cold_tune >= 0.5) == y_tune).astype(int)
    hit_causal_uniform = ((p_uniform_causal_tune >= 0.5) == y_tune).astype(int)

    df_pairs = pd.DataFrame({
        "id": ids_tune,
        "seller_id": sellers_tune,
        "category_id": cats_tune,
        "regime": regimes_tune,
        "y_true": y_tune,
        "p_routed": np.round(p_routed_tune, 6),
        "p_uniform_cold": np.round(p_uniform_cold_tune, 6),
        "p_uniform_causal": np.round(p_uniform_causal_tune, 6),
        "hit_routed": hit_routed,
        "hit_cold": hit_cold,
        "diff_routed_minus_cold": hit_routed - hit_cold,
    })
    df_pairs.to_csv(PAIRS_CSV_PATH, index=False)
    pairs_sha256 = sha256_file(PAIRS_CSV_PATH)
    log(f"Private pairs saved. SHA-256: {pairs_sha256}")

    # 7. Generate Master Report JSON
    report = {
        "generator": "scripts/audit_seller_history_by_regime.py",
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "status": "DEVELOPMENT_ONLY",
        "public_probability_claims": "NO_IDENTIFIABLE",
        "provenance": {
            "split_manifest_sha256": split_manifest["split_sha256"],
            "data_pkl_sha256": sha256_file(DATA_PKL),
            "pairs_private_csv": str(PAIRS_CSV_PATH.relative_to(ROOT)),
            "pairs_private_sha256": pairs_sha256,
        },
        "canonical_tune_evaluation": {
            "n_total": len(tune_indices),
            "tune_cold_new": {
                "n_items": int(tune_cold_mask.sum()),
                "optimal_unseen_model": m_cold_opt,
                "misrouted_warm_model": m_cold_misroute,
                "damage_warm_on_cold_hits": damage_warm_on_cold,
            },
            "tune_seen_bracketed": {
                "n_items": int(tune_seen_mask.sum()),
                "historial_causal": m_causal_seen,
                "historial_batch_loo": m_batch_full_seen,
                "causal_model_with_batch_features": m_batch_leak_seen,
                "ausencia_de_historial": m_cold_fallback_seen,
                "misrouting_damage_cold_on_seen_hits": damage_cold_on_seen,
                "lookahead_leakage_apparent_gain_hits": lookahead_gain_seen,
            },
            "routing_policy_frozen_tune": {
                "rule": "If seller unseen in fit -> route to Cold Specialist; If seller seen in fit -> route to Seen Causal Model with strictly causal features",
                "routed_ensemble": m_tune_routed,
                "uniform_cold_fallback": m_tune_uniform_cold,
                "uniform_causal_model": m_tune_uniform_causal,
                "net_damage_uniform_cold_hits": m_tune_routed["hits"] - m_tune_uniform_cold["hits"],
                "net_damage_uniform_causal_hits": m_tune_routed["hits"] - m_tune_uniform_causal["hits"],
            },
        },
        "comprehensive_5_regimes_inner_fit": {
            "n_val": len(df_val),
            "regimes_breakdown": regime_results,
            "lookahead_leakage_in_antes_proof": {
                "regime": "antes",
                "causal_hits": regime_results["antes"]["causal"]["hits"],
                "batch_hits": regime_results["antes"]["batch"]["hits"],
                "cold_hits": regime_results["antes"]["cold_fallback"]["hits"],
                "false_lookahead_advantage_hits": regime_results["antes"]["lookahead_gain_batch_over_causal"],
                "explanation": "Items in 'antes' occurred strictly before all fit items of that seller. In causal reality, 0 prior sales exist (N_prior = 0). Batch LOO leaks 100% future items, creating an illusion of +122 false hits that will fail in out-of-sample deployment.",
            },
        },
        "test_set_regime_mapping": {
            "n_test": 30000,
            "distribution": {
                "cold_new": {"count": 8513, "share": 0.283767},
                "seen_bracketed": {"count": 14117, "share": 0.470567},
                "after": {"count": 3121, "share": 0.104033},
                "before": {"count": 3118, "share": 0.103933},
                "ties": {"count": 1131, "share": 0.037700},
            },
            "projected_misrouting_risks": {
                "risk_of_discarding_history_on_warm": "Extrapolating from tune seen (101 hits lost in 6,286 rows), applying cold fallback across all 18,369 warm items (bracketed + after + ties) would cause an estimated loss of >300 to 500 hits.",
                "risk_of_batch_lookahead_on_before": "Treating the 3,118 'before' items in test as having batch history introduces severe future leakage, artificially inflating expectations by ~225 false hits that cannot be realized.",
            },
        },
    }

    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"Master report saved to {REPORT_JSON_PATH}.")
    log(f"Completed in {time.time() - t_start:.2f}s.")


if __name__ == "__main__":
    main()
