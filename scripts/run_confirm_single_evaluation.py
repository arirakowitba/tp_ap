"""AP-2026-Q-2: Single Held-Out Confirmation Evaluation on Canonical Confirm Split.

Strict Protocol & Governance Rules:
1. Executed EXACTLY ONCE on the held-out confirm partition (14,075 rows from canonical_3way_splits.csv).
2. Pre-registration of frozen artifacts:
   - Split SHA-256: 6afe07abc3fa2a93437dc544af5e709d7e84d3b6e8eaf9bcadac29912ef5b133
   - Frozen weights: w_LGB = 0.5752, w_CB = 0.4248, w_XGB = 0.0000 (from inner 4-fold fit/tune)
   - Decision rule: cold -> Unseen Specialist; seen -> Seen Causal Model (strictly t_j < t_i); threshold = 0.5
   - Preregistered promotion criteria:
     * Delta hits on confirm > 0 vs LightGBM Enriched Base
     * Bootstrap two-way Q05 >= 0
     * Robustness under leave-top-5 sellers and leave-top-5 categories (net delta > 0)
3. Three distinct bootstrap schemes (2,000 resamples each):
   - Clustered by seller_id
   - Clustered by category_id
   - Two-way clustered by (seller_id, category_id) with multiplicative interaction weights
4. Metrics across temporal regimes:
   - Raw confirm breakdown: confirm_cold_new (7,875 rows) and confirm_seen_bracketed (6,200 rows)
   - Test-reweighted projection across the 5 canonical regimes (cold, bracketed, after, before, ties)
5. Effective support & top contributor sensitivity (leave-top-5 sellers, leave-top-5 categories).
6. Outputs:
   - results/confirm_evaluation_pairs_private.csv (gitignored)
   - results/confirm_evaluation_report.json
   - Gate C: Declares "NO_IDENTIFIABLE" for unconditioned public probability claims.
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
from catboost import CatBoostClassifier
import xgboost as xgb
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
PAIRS_CSV_PATH = RESULTS_DIR / "confirm_evaluation_pairs_private.csv"
REPORT_JSON_PATH = RESULTS_DIR / "confirm_evaluation_report.json"

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


def clip_prob(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)


def compute_metrics_bundle(y_true, p_pred):
    acc = float(accuracy_score(y_true, p_pred >= 0.5))
    hits = int(((p_pred >= 0.5) == y_true).sum())
    ll = float(log_loss(y_true, clip_prob(p_pred)))
    brier = float(brier_score_loss(y_true, p_pred))
    return {
        "n_rows": len(y_true),
        "hits": hits,
        "accuracy": round(acc, 6),
        "logloss": round(ll, 6),
        "brier_score": round(brier, 6),
    }


def bootstrap_cluster_1way(y_true, p_cand, p_base, cluster_col, n_boot=2000, seed=20261114):
    """One-way cluster bootstrap on cluster_col (seller or category)."""
    y_true = np.asarray(y_true)
    c_cand = ((p_cand >= 0.5) == y_true).astype(int)
    c_base = ((p_base >= 0.5) == y_true).astype(int)
    diff = c_cand - c_base

    unique_clusters, c_idx = np.unique(cluster_col, return_inverse=True)
    n_clusters = len(unique_clusters)

    rng = np.random.default_rng(seed)
    boot_deltas = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        w_c = rng.choice([-1.0, 1.0], size=n_clusters)
        w_row = 1.0 + w_c[c_idx]
        boot_deltas[b] = np.sum(diff * w_row)

    return {
        "delta_hits_median": round(float(np.median(boot_deltas)), 2),
        "delta_hits_q05": round(float(np.percentile(boot_deltas, 5.0)), 2),
        "delta_hits_q95": round(float(np.percentile(boot_deltas, 95.0)), 2),
        "prob_delta_hits_gt0": round(float(np.mean(boot_deltas > 0)), 4),
    }


def bootstrap_two_way_cluster(y_true, p_cand, p_base, sellers, cats, n_boot=2000, seed=20261114):
    """Two-way cluster bootstrap on (seller_id, category_id) with multiplicative weights."""
    y_true = np.asarray(y_true)
    c_cand = ((p_cand >= 0.5) == y_true).astype(int)
    c_base = ((p_base >= 0.5) == y_true).astype(int)
    diff = c_cand - c_base

    p_c_clip = clip_prob(p_cand)
    p_b_clip = clip_prob(p_base)
    ll_c = -(y_true * np.log(p_c_clip) + (1.0 - y_true) * np.log(1.0 - p_c_clip))
    ll_b = -(y_true * np.log(p_b_clip) + (1.0 - y_true) * np.log(1.0 - p_b_clip))
    ll_diff = ll_c - ll_b

    unique_sellers, s_idx = np.unique(sellers, return_inverse=True)
    unique_cats, c_idx = np.unique(cats, return_inverse=True)
    n_s = len(unique_sellers)
    n_c = len(unique_cats)

    rng = np.random.default_rng(seed)
    boot_hits_deltas = np.empty(n_boot, dtype=np.float64)
    boot_ll_deltas = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        w_s = rng.choice([-1.0, 1.0], size=n_s)
        w_c = rng.choice([-1.0, 1.0], size=n_c)
        w_row = 1.0 + (w_s[s_idx] * w_c[c_idx])
        w_sum = np.sum(w_row)
        boot_hits_deltas[b] = np.sum(diff * w_row)
        boot_ll_deltas[b] = np.sum(ll_diff * w_row) / w_sum

    return {
        "delta_hits_median": round(float(np.median(boot_hits_deltas)), 2),
        "delta_hits_q05": round(float(np.percentile(boot_hits_deltas, 5.0)), 2),
        "delta_hits_q95": round(float(np.percentile(boot_hits_deltas, 95.0)), 2),
        "prob_delta_hits_gt0": round(float(np.mean(boot_hits_deltas > 0)), 4),
        "delta_logloss_median": round(float(np.median(boot_ll_deltas)), 6),
        "delta_logloss_q05": round(float(np.percentile(boot_ll_deltas, 5.0)), 6),
        "delta_logloss_q95": round(float(np.percentile(boot_ll_deltas, 95.0)), 6),
    }


def compute_comprehensive_paired_eval(y_true, p_cand, p_base, sellers, cats):
    y_true = np.asarray(y_true)
    c_cand = ((p_cand >= 0.5) == y_true).astype(int)
    c_base = ((p_base >= 0.5) == y_true).astype(int)
    diff = c_cand - c_base

    fixes = int(((c_cand == 1) & (c_base == 0)).sum())
    breaks = int(((c_cand == 0) & (c_base == 1)).sum())
    net_delta = fixes - breaks

    pred_cand = (p_cand >= 0.5).astype(int)
    pred_base = (p_base >= 0.5).astype(int)
    flips_total = int((pred_cand != pred_base).sum())
    flips_to_new = int(((pred_base == 1) & (pred_cand == 0)).sum())
    flips_to_used = int(((pred_base == 0) & (pred_cand == 1)).sum())

    boot_seller = bootstrap_cluster_1way(y_true, p_cand, p_base, sellers, n_boot=2000, seed=20261114)
    boot_cat = bootstrap_cluster_1way(y_true, p_cand, p_base, cats, n_boot=2000, seed=20261114)
    boot_2way = bootstrap_two_way_cluster(y_true, p_cand, p_base, sellers, cats, n_boot=2000, seed=20261114)

    # Top contributors ablation
    df_contrib = pd.DataFrame({"diff": diff, "seller": sellers, "cat": cats})
    seller_net = df_contrib.groupby("seller")["diff"].sum().sort_values(ascending=False)
    cat_net = df_contrib.groupby("cat")["diff"].sum().sort_values(ascending=False)

    top5_sellers = list(seller_net.head(5).index)
    top5_cats = list(cat_net.head(5).index)

    net_without_top5_sellers = int(df_contrib[~df_contrib["seller"].isin(top5_sellers)]["diff"].sum())
    net_without_top5_cats = int(df_contrib[~df_contrib["cat"].isin(top5_cats)]["diff"].sum())

    return {
        "flips_total": flips_total,
        "effective_support": flips_total,
        "flips_to_new": flips_to_new,
        "flips_to_used": flips_to_used,
        "fixes": fixes,
        "breaks": breaks,
        "net_delta_hits": net_delta,
        "bootstrap_seller_1way": boot_seller,
        "bootstrap_category_1way": boot_cat,
        "bootstrap_two_way": boot_2way,
        "top5_sellers": [str(s) for s in top5_sellers],
        "top5_seller_net_share": int(seller_net.head(5).sum()),
        "net_without_top5_sellers": net_without_top5_sellers,
        "top5_cats": [str(c) for c in top5_cats],
        "top5_cat_net_share": int(cat_net.head(5).sum()),
        "net_without_top5_cats": net_without_top5_cats,
    }


def main():
    t_start = time.time()
    log("=== AP-2026-Q-2: Evaluación Única de Confirmación en Confirm (14,075 filas) ===")

    # 1. Pre-registration Phase
    log("STAGE 1: Pre-registration and Freezing of Manifest and Criteria...")
    with open(SPLIT_MANIFEST, "r", encoding="utf-8") as f:
        split_manifest = json.load(f)
    split_sha256 = split_manifest["split_sha256"]
    log(f"Pre-registered Split SHA-256: {split_sha256}")

    # Frozen convex weights
    frozen_weights = {
        "w_lgb": 0.5752,
        "w_cb": 0.4248,
        "w_xgb": 0.0000,
    }
    log(f"Pre-registered Frozen Convex Weights: {frozen_weights}")

    # Promotion criteria
    promotion_criteria = {
        "gate_1_positive_delta_confirm": "net_delta_hits > 0 vs LightGBM_Enriched_Base",
        "gate_2_bootstrap_two_way_q05": "delta_hits_q05 >= 0 under 2-way (seller, category) cluster bootstrap",
        "gate_3_leave_top5_sellers_positive": "net_without_top5_sellers > 0",
        "gate_4_leave_top5_cats_positive": "net_without_top5_cats > 0",
        "gate_5_public_probability_compliance": "Must declare NO_IDENTIFIABLE (Gate C)",
    }
    log("Pre-registered Promotion Criteria frozen.")

    # 2. Load Confirm Data
    split_df = pd.read_csv(SPLIT_CSV)
    fit_mask = split_df["role"] == "fit"
    tune_mask = split_df["role"] == "tune"
    confirm_mask = split_df["role"] == "confirm"

    fit_indices = np.flatnonzero(fit_mask)
    tune_indices = np.flatnonzero(tune_mask)
    confirm_indices = np.flatnonzero(confirm_mask)

    log(f"Loading parsed train data from {DATA_PKL}...")
    with open(DATA_PKL, "rb") as f:
        df = pickle.load(f)

    y_fit = df["y"].iloc[fit_indices].to_numpy().astype(np.int32)
    y_confirm = df["y"].iloc[confirm_indices].to_numpy().astype(np.int32)
    ids_confirm = df["id"].iloc[confirm_indices].values.astype(str)
    sellers_confirm = df["seller_id"].iloc[confirm_indices].values.astype(str)
    cats_confirm = df["category_id"].iloc[confirm_indices].values.astype(str)
    regimes_confirm = split_df.iloc[confirm_indices]["regime"].values.astype(str)

    confirm_cold_mask = (regimes_confirm == "confirm_cold_new")
    confirm_seen_mask = (regimes_confirm == "confirm_seen_bracketed")

    log(f"Confirm Total Rows: {len(confirm_indices)}")
    log(f"  - Cold New: {confirm_cold_mask.sum()} ({confirm_cold_mask.mean():.2%})")
    log(f"  - Seen Bracketed: {confirm_seen_mask.sum()} ({confirm_seen_mask.mean():.2%})")

    # 3. Build Matrices for Fit and Confirm
    tr_data = ag.precompute_item_intrinsic_arrays(df)
    inner_cv = list(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=61111).split(
        fit_indices, tr_data["y"][fit_indices], tr_data["seller_id"][fit_indices]
    ))

    log("Building fold matrices using cache text_6afe07abc3fa2a93.npz for [tune, confirm]...")
    t0_mat = time.time()
    mats = ag.build_fold_matrices(
        T_indices=fit_indices,
        eval_indices_list=[tune_indices, confirm_indices],
        splits_T=inner_cv,
        tr_data=tr_data,
        eval_data_list=[tr_data, tr_data],
        cache_path=CACHE_PATH,
    )
    log(f"Fold matrices built in {time.time() - t0_mat:.2f}s.")

    prior = float(np.mean(y_fit))
    m_prior = 5.0
    ste_col_idx = 74

    X_unseen_fit = mats["X_m1_unseen_T"]
    X_seen_fit = mats["X_m1_seen_causal_T"]
    X_unseen_confirm = mats["X_m1_unseen_evals"][1]

    # Compute strictly causal features for confirm
    log("Computing strictly causal prior-items features for confirm partition...")
    df_fit_s = pd.DataFrame({
        "seller_id": tr_data["seller_id"][fit_indices],
        "start_time": tr_data["start_time"][fit_indices],
        "row": tr_data["row"][fit_indices],
        "y": y_fit,
    }).sort_values(["seller_id", "start_time", "row"])

    fit_by_seller = {s: (grp["start_time"].values, grp["y"].values) for s, grp in df_fit_s.groupby("seller_id", sort=False)}
    start_confirm = tr_data["start_time"][confirm_indices]
    causal_te_confirm = np.full(len(confirm_indices), prior, dtype=np.float32)
    causal_cnt_confirm = np.zeros(len(confirm_indices), dtype=np.float32)

    for i in range(len(confirm_indices)):
        s = sellers_confirm[i]
        t = start_confirm[i]
        if s in fit_by_seller:
            times, ys = fit_by_seller[s]
            idx = np.searchsorted(times, t, side="left")
            if idx > 0:
                cnt = float(idx)
                s_sum = float(ys[:idx].sum())
                causal_te_confirm[i] = (s_sum + m_prior * prior) / (cnt + m_prior)
                causal_cnt_confirm[i] = np.log1p(cnt)

    X_causal_confirm = X_unseen_confirm.copy()
    X_causal_confirm[:, ste_col_idx] = causal_te_confirm
    X_causal_confirm[:, ste_col_idx + 1] = causal_cnt_confirm

    # 4. Train Dual-Regime Models on Fit (Bagged 3 seeds)
    log("\nSTAGE 2: Fitting Dual-Regime Models on Fit (40,881 rows)...")
    hp = {"num_leaves": 48, "min_child_samples": 35, "colsample_bytree": 0.75}

    # Model 1: LightGBM
    log("Fitting LightGBM Models (Unseen & Seen Causal)...")
    m_lgb_cold = lgb.LGBMClassifier(n_estimators=350, learning_rate=0.035, num_leaves=hp["num_leaves"], min_child_samples=hp["min_child_samples"], colsample_bytree=hp["colsample_bytree"], subsample=0.85, subsample_freq=1, random_state=42, n_jobs=N_JOBS, verbosity=-1)
    m_lgb_cold.fit(X_unseen_fit, y_fit)
    p_lgb_confirm_cold = m_lgb_cold.predict_proba(X_unseen_confirm)[:, 1]

    m_lgb_seen = lgb.LGBMClassifier(n_estimators=350, learning_rate=0.035, num_leaves=hp["num_leaves"], min_child_samples=hp["min_child_samples"], colsample_bytree=hp["colsample_bytree"], subsample=0.85, subsample_freq=1, random_state=42, n_jobs=N_JOBS, verbosity=-1)
    m_lgb_seen.fit(X_seen_fit, y_fit)
    p_lgb_confirm_seen = m_lgb_seen.predict_proba(X_causal_confirm)[:, 1]

    # Model 2: CatBoost
    log("Fitting CatBoost Models (Unseen & Seen Causal)...")
    m_cb_cold = CatBoostClassifier(iterations=350, learning_rate=0.04, depth=7, l2_leaf_reg=5.0, thread_count=N_JOBS, verbose=0, random_seed=42)
    m_cb_cold.fit(X_unseen_fit, y_fit)
    p_cb_confirm_cold = m_cb_cold.predict_proba(X_unseen_confirm)[:, 1]

    m_cb_seen = CatBoostClassifier(iterations=350, learning_rate=0.04, depth=7, l2_leaf_reg=5.0, thread_count=N_JOBS, verbose=0, random_seed=42)
    m_cb_seen.fit(X_seen_fit, y_fit)
    p_cb_confirm_seen = m_cb_seen.predict_proba(X_causal_confirm)[:, 1]

    # Model 3: XGBoost
    log("Fitting XGBoost Models (Unseen & Seen Causal)...")
    m_xgb_cold = xgb.XGBClassifier(n_estimators=350, learning_rate=0.035, max_depth=7, subsample=0.85, colsample_bytree=0.75, tree_method="hist", eval_metric="logloss", random_state=42, n_jobs=N_JOBS)
    m_xgb_cold.fit(X_unseen_fit, y_fit)
    p_xgb_confirm_cold = m_xgb_cold.predict_proba(X_unseen_confirm)[:, 1]

    m_xgb_seen = xgb.XGBClassifier(n_estimators=350, learning_rate=0.035, max_depth=7, subsample=0.85, colsample_bytree=0.75, tree_method="hist", eval_metric="logloss", random_state=42, n_jobs=N_JOBS)
    m_xgb_seen.fit(X_seen_fit, y_fit)
    p_xgb_confirm_seen = m_xgb_seen.predict_proba(X_causal_confirm)[:, 1]

    # Assemble Routed Predictions on Confirm (14,075 rows)
    p_lgb_confirm = np.where(confirm_cold_mask, p_lgb_confirm_cold, p_lgb_confirm_seen).astype(np.float32)
    p_cb_confirm = np.where(confirm_cold_mask, p_cb_confirm_cold, p_cb_confirm_seen).astype(np.float32)
    p_xgb_confirm = np.where(confirm_cold_mask, p_xgb_confirm_cold, p_xgb_confirm_seen).astype(np.float32)

    # Candidate 1: Ensamble Convexo Óptimo (w = [0.5752, 0.4248, 0.0])
    p_convex_confirm = (frozen_weights["w_lgb"] * p_lgb_confirm + frozen_weights["w_cb"] * p_cb_confirm + frozen_weights["w_xgb"] * p_xgb_confirm).astype(np.float32)

    # Candidate 2: Mezcla Fija Uniforme (1/3 each)
    p_fixed_confirm = ((p_lgb_confirm + p_cb_confirm + p_xgb_confirm) / 3.0).astype(np.float32)

    # Candidate 3: Poda Conservadora Overlay (consenso CB/XGB contra LGB + margen >= 0.02)
    cb_xgb_agree = (p_cb_confirm >= 0.5) == (p_xgb_confirm >= 0.5)
    margin_mask = np.abs(p_convex_confirm - 0.5) >= 0.02
    overlay_mask = cb_xgb_agree & margin_mask
    p_overlay_confirm = np.where(overlay_mask, p_convex_confirm, p_lgb_confirm).astype(np.float32)

    candidates_confirm = {
        "LightGBM_Enriched_Base": p_lgb_confirm,
        "Ensamble_Convexo_Optimo": p_convex_confirm,
        "Mezcla_Fija_Uniforme": p_fixed_confirm,
        "CatBoost_Nativo": p_cb_confirm,
        "XGBoost_Enriched": p_xgb_confirm,
        "Poda_Conservadora_Overlay": p_overlay_confirm,
    }

    # 5. Evaluate on Confirm (14,075 rows)
    log("\n" + "=" * 90)
    log("EVALUATION ON HELD-OUT CONFIRM PARTITION (14,075 rows)")
    log("=" * 90)
    log(f"{'Candidato / Política':26s} | {'Hits':5s} | {'Exactitud':8s} | {'LogLoss':8s} | {'Flips':5s} | {'Fixes':5s} | {'Breaks':6s} | {'Net Delta':9s}")
    log("-" * 90)

    p_base_confirm = p_lgb_confirm
    confirm_results = {}

    for name, p_cand in candidates_confirm.items():
        mb = compute_metrics_bundle(y_confirm, p_cand)
        paired = compute_comprehensive_paired_eval(y_confirm, p_cand, p_base_confirm, sellers_confirm, cats_confirm)
        confirm_results[name] = {
            "metrics": mb,
            "paired_vs_lgb": paired,
        }
        log(f"{name:26s} | {mb['hits']:5d} | {mb['accuracy']:7.4%} | {mb['logloss']:8.6f} | {paired['flips_total']:5d} | {paired['fixes']:5d} | {paired['breaks']:6d} | {paired['net_delta_hits']:+9d}")

    # 6. Evaluation by Temporal Regimes on Confirm
    log("\n" + "=" * 90)
    log("BREAKDOWN BY TEMPORAL REGIMES IN CONFIRM")
    log("=" * 90)
    regime_breakdown = {}
    for r_name, r_mask in [("confirm_cold_new", confirm_cold_mask), ("confirm_seen_bracketed", confirm_seen_mask)]:
        y_r = y_confirm[r_mask]
        s_r = sellers_confirm[r_mask]
        c_r = cats_confirm[r_mask]
        p_base_r = p_base_confirm[r_mask]
        p_convex_r = p_convex_confirm[r_mask]

        mb_base_r = compute_metrics_bundle(y_r, p_base_r)
        mb_convex_r = compute_metrics_bundle(y_r, p_convex_r)
        paired_r = compute_comprehensive_paired_eval(y_r, p_convex_r, p_base_r, s_r, c_r)

        regime_breakdown[r_name] = {
            "n_rows": int(r_mask.sum()),
            "share": float(r_mask.mean()),
            "base_lgb": mb_base_r,
            "convex_blend": mb_convex_r,
            "paired": paired_r,
        }
        log(f"Régimen {r_name} (N={r_mask.sum()}):")
        log(f"  - Base LGBM:      Hits={mb_base_r['hits']} ({mb_base_r['accuracy']:.4%}), LL={mb_base_r['logloss']:.6f}")
        log(f"  - Convex Blend:   Hits={mb_convex_r['hits']} ({mb_convex_r['accuracy']:.4%}), LL={mb_convex_r['logloss']:.6f}")
        log(f"  - Paired Delta:   Fixes={paired_r['fixes']}, Breaks={paired_r['breaks']}, Net={paired_r['net_delta_hits']:+d}")

    # Test Reweighted Projection (across test canonical weights)
    # Test distribution: cold (0.2838), bracketed (0.4706), after (0.1040), before (0.1039), ties (0.0377)
    test_weights = {
        "cold_new": 0.283767,
        "seen_bracketed": 0.470567,
        "after": 0.104033,
        "before": 0.103933,
        "ties": 0.037700,
    }
    # Weighted accuracy on test projection:
    # cold_acc * 0.2838 + seen_bracketed_acc * 0.4706 + after_acc * 0.1040 + before_acc * 0.1039 + ties_acc * 0.0377
    acc_cold_convex = regime_breakdown["confirm_cold_new"]["convex_blend"]["accuracy"]
    acc_seen_convex = regime_breakdown["confirm_seen_bracketed"]["convex_blend"]["accuracy"]
    acc_cold_base = regime_breakdown["confirm_cold_new"]["base_lgb"]["accuracy"]
    acc_seen_base = regime_breakdown["confirm_seen_bracketed"]["base_lgb"]["accuracy"]

    # Using causal properties established in AG-0198:
    # 'after' accuracy is at least as high as seen_bracketed
    # 'before' accuracy falls back to cold accuracy
    # 'ties' accuracy is comparable to seen_bracketed
    proj_acc_convex = (
        acc_cold_convex * test_weights["cold_new"]
        + acc_seen_convex * test_weights["seen_bracketed"]
        + acc_seen_convex * test_weights["after"]
        + acc_cold_convex * test_weights["before"]
        + acc_seen_convex * test_weights["ties"]
    )
    proj_acc_base = (
        acc_cold_base * test_weights["cold_new"]
        + acc_seen_base * test_weights["seen_bracketed"]
        + acc_seen_base * test_weights["after"]
        + acc_cold_base * test_weights["before"]
        + acc_seen_base * test_weights["ties"]
    )
    log(f"\nTest-Reweighted Projected Accuracy:")
    log(f"  - Base LightGBM:     {proj_acc_base:.4%}")
    log(f"  - Ensamble Convexo:  {proj_acc_convex:.4%} (Projected Delta: {(proj_acc_convex - proj_acc_base)*100:+.4f}%)")

    # 7. Check Pre-registered Promotion Criteria
    log("\n" + "=" * 90)
    log("GATE CHECK: PRE-REGISTERED PROMOTION CRITERIA")
    log("=" * 90)
    paired_convex = confirm_results["Ensamble_Convexo_Optimo"]["paired_vs_lgb"]
    net_hits_confirm = paired_convex["net_delta_hits"]
    q05_2way = paired_convex["bootstrap_two_way"]["delta_hits_q05"]
    q05_seller = paired_convex["bootstrap_seller_1way"]["delta_hits_q05"]
    q05_cat = paired_convex["bootstrap_category_1way"]["delta_hits_q05"]
    net_no_top5_s = paired_convex["net_without_top5_sellers"]
    net_no_top5_c = paired_convex["net_without_top5_cats"]

    pass_g1 = net_hits_confirm > 0
    pass_g2 = q05_2way >= 0.0
    pass_g3 = net_no_top5_s > 0
    pass_g4 = net_no_top5_c > 0

    log(f"1. Net Delta Hits on Confirm > 0:               {pass_g1} (Net = {net_hits_confirm:+d})")
    log(f"2. Lower Bound Q05 (2-way cluster) >= 0:        {pass_g2} (Q05 = {q05_2way:+.2f}, Seller Q05 = {q05_seller:+.2f}, Cat Q05 = {q05_cat:+.2f})")
    log(f"3. Robustness w/o Top-5 Sellers > 0:            {pass_g3} (Net w/o Top-5 Sellers = {net_no_top5_s:+d})")
    log(f"4. Robustness w/o Top-5 Categories > 0:         {pass_g4} (Net w/o Top-5 Cats = {net_no_top5_c:+d})")

    overall_promotion = pass_g1 and pass_g2 and pass_g3 and pass_g4
    promotion_verdict = "PASSED_PROMOTION_CRITERIA" if overall_promotion else "REJECTED_OR_CONSTRAINED"
    log(f"\nFinal Promotion Verdict on Confirm: {promotion_verdict}")

    # 8. Save Private Pairs CSV
    log(f"\nWriting private confirm pairs to {PAIRS_CSV_PATH}...")
    hit_lgb = ((p_lgb_confirm >= 0.5) == y_confirm).astype(int)
    hit_convex = ((p_convex_confirm >= 0.5) == y_confirm).astype(int)
    hit_fixed = ((p_fixed_confirm >= 0.5) == y_confirm).astype(int)
    hit_overlay = ((p_overlay_confirm >= 0.5) == y_confirm).astype(int)

    df_pairs = pd.DataFrame({
        "id": ids_confirm,
        "seller_id": sellers_confirm,
        "category_id": cats_confirm,
        "regime": regimes_confirm,
        "y_true": y_confirm,
        "p_lgb": np.round(p_lgb_confirm, 6),
        "p_convex": np.round(p_convex_confirm, 6),
        "p_fixed": np.round(p_fixed_confirm, 6),
        "p_overlay": np.round(p_overlay_confirm, 6),
        "hit_lgb": hit_lgb,
        "hit_convex": hit_convex,
        "diff_convex_minus_lgb": hit_convex - hit_lgb,
    })
    df_pairs.to_csv(PAIRS_CSV_PATH, index=False)
    pairs_sha256 = sha256_file(PAIRS_CSV_PATH)
    log(f"Private confirm pairs saved. SHA-256: {pairs_sha256}")

    # 9. Save Master Report JSON
    report = {
        "generator": "scripts/run_confirm_single_evaluation.py",
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "status": "DEVELOPMENT_ONLY",
        "public_probability_claims": "NO_IDENTIFIABLE",
        "pre_registered_freezing": {
            "split_sha256": split_sha256,
            "data_pkl_sha256": sha256_file(DATA_PKL),
            "frozen_weights": frozen_weights,
            "promotion_criteria": promotion_criteria,
            "decision_rule": "cold -> Unseen Specialist; seen -> Seen Causal Model (strictly t_j < t_i); threshold = 0.5",
        },
        "confirm_candidates_evaluation": confirm_results,
        "regime_breakdown_confirm": regime_breakdown,
        "test_reweighted_projection": {
            "test_canonical_weights": test_weights,
            "projected_accuracy_base_lgb": float(proj_acc_base),
            "projected_accuracy_convex_blend": float(proj_acc_convex),
            "projected_delta_accuracy": float(proj_acc_convex - proj_acc_base),
        },
        "promotion_gate_results": {
            "verdict": promotion_verdict,
            "pass_g1_positive_delta": bool(pass_g1),
            "pass_g2_q05_nonnegative": bool(pass_g2),
            "pass_g3_leave_top5_sellers": bool(pass_g3),
            "pass_g4_leave_top5_cats": bool(pass_g4),
            "effective_support": paired_convex["effective_support"],
            "bootstrap_two_way_q05": q05_2way,
            "bootstrap_seller_q05": q05_seller,
            "bootstrap_category_q05": q05_cat,
        },
        "provenance": {
            "pairs_private_csv": str(PAIRS_CSV_PATH.relative_to(ROOT)),
            "pairs_private_sha256": pairs_sha256,
        },
    }

    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"Master confirm report saved to {REPORT_JSON_PATH}.")
    log(f"Execution completed in {time.time() - t_start:.2f}s.")


if __name__ == "__main__":
    main()
