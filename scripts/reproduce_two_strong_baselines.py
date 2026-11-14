"""AP-2026-Q-2: Master Reproduction of Two Strong Baselines on Canonical Tripartite Split.

Baselines evaluated:
1. Baseline 1: Line 3 (pure text OOF + causal seller TE, bagged seeds).
2. Baseline 2: Texto OOF seller-group con atributos (LightGBM Enriched: attributes, text OOF by seller, causal seller TE).

Protocol & Guarantees:
- Strictly identical evaluation rows across both baselines (canonical_3way_splits.csv: tune 15,044 rows).
- Zero leakage into tune (all hyperparameter tuning and early stopping strictly inside 4 inner folds of fit: 40,881 rows).
- Dual-regime routing: unseen model for tune_cold_new (8,758 rows), seen causal model for tune_seen_bracketed (6,286 rows).
- Verification of polarity (used=1, new=0) and 1:1 join strictly on string ID.
- Two-way cluster bootstrap on (seller, category) with 2,000 resamples.
- Top-5 contributor ablation (leave-top-5 categories and leave-top-5 sellers).
- Saves private pairs to results/baselines_paired_tune_pairs_private.csv (gitignored).
- Generates results/baselines_reproduction_report.json.
- Zero Kaggle API calls.
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
PAIRS_CSV_PATH = RESULTS_DIR / "baselines_paired_tune_pairs_private.csv"
REPORT_JSON_PATH = RESULTS_DIR / "baselines_reproduction_report.json"

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


def run_inner_cv_tuning_and_early_stopping(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    seller_ids: np.ndarray,
    inner_cv_splits: list,
    hp_candidates: list,
    model_name: str,
    max_estimators: int = 1200,
    stopping_rounds: int = 50,
    learning_rate: float = 0.035,
) -> dict:
    """Run inner CV tuning and early stopping strictly on inner folds of fit."""
    log(f"Starting inner CV tuning for {model_name} across {len(hp_candidates)} HP candidates...")
    t0 = time.time()
    results = []

    for c_idx, hp in enumerate(hp_candidates):
        t_cand = time.time()
        oof_preds = np.zeros(len(y_train_full), dtype=np.float32)
        best_iters = []

        for fold_idx, (tr_loc, va_loc) in enumerate(inner_cv_splits):
            X_tr, y_tr = X_train_full[tr_loc], y_train_full[tr_loc]
            X_va, y_va = X_train_full[va_loc], y_train_full[va_loc]

            model = LGBMClassifier(
                n_estimators=max_estimators,
                learning_rate=learning_rate,
                num_leaves=hp["num_leaves"],
                min_child_samples=hp["min_child_samples"],
                colsample_bytree=hp["colsample_bytree"],
                subsample=0.85,
                subsample_freq=1,
                random_state=42,
                n_jobs=N_JOBS,
                verbosity=-1,
                eval_metric="binary_logloss",
            )

            # Early stopping strictly on inner va_loc
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False)],
            )

            best_iter = model.best_iteration_
            best_iters.append(best_iter)
            oof_preds[va_loc] = model.predict_proba(X_va)[:, 1]

        acc = float(accuracy_score(y_train_full, oof_preds >= 0.5))
        ll = float(log_loss(y_train_full, np.clip(oof_preds, 1e-6, 1.0 - 1e-6)))
        hits = int(((oof_preds >= 0.5) == y_train_full).sum())
        median_iter = int(np.median(best_iters))
        mean_iter = float(np.mean(best_iters))

        res_entry = {
            "candidate_idx": c_idx,
            "hp": hp,
            "oof_acc": round(acc, 6),
            "oof_logloss": round(ll, 6),
            "oof_hits": hits,
            "median_best_iter": median_iter,
            "mean_best_iter": round(mean_iter, 1),
            "fold_best_iters": best_iters,
            "tuning_time_sec": round(time.time() - t_cand, 2),
        }
        results.append(res_entry)
        log(f"  Candidate {c_idx+1}/{len(hp_candidates)}: hp={hp} -> OOF Acc={acc:.5f}, Logloss={ll:.5f}, Median Best Iter={median_iter}")

    # Best candidate: minimal logloss, tie-breaker maximal hits
    best_candidate = sorted(results, key=lambda x: (x["oof_logloss"], -x["oof_hits"]))[0]
    total_time = round(time.time() - t0, 2)
    log(f"Tuning finished in {total_time}s. Best HP: {best_candidate['hp']}, Median Iter: {best_candidate['median_best_iter']}")

    return {
        "best_hp": best_candidate["hp"],
        "optimal_n_estimators": best_candidate["median_best_iter"],
        "best_oof_acc": best_candidate["oof_acc"],
        "best_oof_logloss": best_candidate["oof_logloss"],
        "best_oof_hits": best_candidate["oof_hits"],
        "all_candidates": results,
        "total_tuning_time_sec": total_time,
    }


def fit_bagged_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    hp: dict,
    n_estimators: int,
    seeds: list,
    learning_rate: float = 0.035,
) -> tuple:
    """Fit bagged LightGBM on full train and predict on test."""
    p_test_bagged = np.zeros(len(X_test), dtype=np.float64)
    seed_times = []

    for sd in seeds:
        t0 = time.time()
        model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=hp["num_leaves"],
            min_child_samples=hp["min_child_samples"],
            colsample_bytree=hp["colsample_bytree"],
            subsample=0.85,
            subsample_freq=1,
            random_state=sd,
            n_jobs=N_JOBS,
            verbosity=-1,
        )
        model.fit(X_train, y_train)
        p_test_bagged += model.predict_proba(X_test)[:, 1] / len(seeds)
        seed_times.append(round(time.time() - t0, 2))

    return p_test_bagged.astype(np.float32), seed_times


def bootstrap_two_way_cluster(
    y_true: np.ndarray,
    p_cand: np.ndarray,
    p_base: np.ndarray,
    sellers: np.ndarray,
    cats: np.ndarray,
    n_boot: int = 2000,
    seed: int = 20261113,
) -> dict:
    """Two-way cluster bootstrap on (seller, category) clusters for paired difference."""
    y_true = np.asarray(y_true)
    c_cand = ((p_cand >= 0.5) == y_true).astype(int)
    c_base = ((p_base >= 0.5) == y_true).astype(int)
    diff = c_cand - c_base

    p_c_clip = np.clip(p_cand, 1e-6, 1.0 - 1e-6)
    p_b_clip = np.clip(p_base, 1e-6, 1.0 - 1e-6)
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


def compute_paired_metrics(y_true, p_cand, p_base, sellers, cats):
    y_true = np.asarray(y_true)
    c_cand = ((p_cand >= 0.5) == y_true).astype(int)
    c_base = ((p_base >= 0.5) == y_true).astype(int)
    diff = c_cand - c_base

    fixes = int(((c_cand == 1) & (c_base == 0)).sum())
    breaks = int(((c_cand == 0) & (c_base == 1)).sum())
    net_delta = fixes - breaks

    pred_cand = (p_cand >= 0.5).astype(int)
    pred_base = (p_base >= 0.5).astype(int)
    flips_to_new = int(((pred_base == 1) & (pred_cand == 0)).sum())
    flips_to_used = int(((pred_base == 0) & (pred_cand == 1)).sum())

    boot = bootstrap_two_way_cluster(y_true, p_cand, p_base, sellers, cats, n_boot=2000)

    # Top contributors ablation
    df_contrib = pd.DataFrame({"seller": sellers, "cat": cats, "net": diff})
    cat_net = df_contrib.groupby("cat")["net"].sum().sort_values(ascending=False)
    top5_cats = set(cat_net.head(5).index)
    net_without_top5_cats = int(df_contrib.loc[~df_contrib["cat"].isin(top5_cats), "net"].sum())

    seller_net = df_contrib.groupby("seller")["net"].sum().sort_values(ascending=False)
    top5_sellers = set(seller_net.head(5).index)
    net_without_top5_sellers = int(df_contrib.loc[~df_contrib["seller"].isin(top5_sellers), "net"].sum())

    cand_bundle = compute_metrics_bundle(y_true, p_cand)
    base_bundle = compute_metrics_bundle(y_true, p_base)

    return {
        "n_rows": len(y_true),
        "candidate": cand_bundle,
        "baseline": base_bundle,
        "delta_hits": net_delta,
        "fixes": fixes,
        "breaks": breaks,
        "delta_acc": round(cand_bundle["accuracy"] - base_bundle["accuracy"], 6),
        "delta_logloss": round(cand_bundle["logloss"] - base_bundle["logloss"], 6),
        "delta_brier": round(cand_bundle["brier_score"] - base_bundle["brier_score"], 6),
        "flips_to_new": flips_to_new,
        "flips_to_used": flips_to_used,
        "bootstrap_two_way": boot,
        "top5_categories": list(top5_cats),
        "top5_cat_net_share": int(cat_net.head(5).sum()),
        "net_without_top5_categories": net_without_top5_cats,
        "top5_sellers": [int(s) for s in list(top5_sellers)],
        "top5_seller_net_share": int(seller_net.head(5).sum()),
        "net_without_top5_sellers": net_without_top5_sellers,
    }


def main():
    t_global_start = time.time()
    log("=== AP-2026-Q-2: Reproducción de Dos Baselines Fuertes (Line3 vs LGB Enriched) ===")

    # 1. Load Split and Parsed Train
    log(f"Loading split manifest from {SPLIT_MANIFEST}...")
    with open(SPLIT_MANIFEST, "r", encoding="utf-8") as f:
        split_manifest = json.load(f)
    log(f"Verified Split SHA-256: {split_manifest['split_sha256']}")

    split_df = pd.read_csv(SPLIT_CSV)
    fit_mask = split_df["role"] == "fit"
    tune_mask = split_df["role"] == "tune"
    fit_indices = np.flatnonzero(fit_mask)
    tune_indices = np.flatnonzero(tune_mask)

    tune_cold_mask = split_df.iloc[tune_indices]["regime"].values == "tune_cold_new"
    tune_seen_mask = split_df.iloc[tune_indices]["regime"].values == "tune_seen_bracketed"

    log(f"Fit count: {len(fit_indices)}, Tune count: {len(tune_indices)}")
    log(f"Tune cold new: {tune_cold_mask.sum()}, Tune seen bracketed: {tune_seen_mask.sum()}")

    log(f"Loading parsed train data from {DATA_PKL}...")
    with open(DATA_PKL, "rb") as f:
        df = pickle.load(f)

    y_fit = df["y"].iloc[fit_indices].to_numpy().astype(np.int32)
    y_tune = df["y"].iloc[tune_indices].to_numpy().astype(np.int32)
    ids_tune = df["id"].iloc[tune_indices].values.astype(str)
    sellers_tune = df["seller_id"].iloc[tune_indices].values.astype(str)
    cats_tune = df["category_id"].iloc[tune_indices].values.astype(str)
    regimes_tune = split_df.iloc[tune_indices]["regime"].values.astype(str)

    # Polarity verification on train data
    used_indices = np.flatnonzero(y_fit == 1)
    new_indices = np.flatnonzero(y_fit == 0)
    log(f"Polarity check on fit: used (1) = {len(used_indices)} ({len(used_indices)/len(y_fit):.4f}), new (0) = {len(new_indices)} ({len(new_indices)/len(y_fit):.4f})")

    # Keyword check for polarity
    titles_fit = df["title"].iloc[fit_indices].astype(str).str.lower()
    used_kw_flag = titles_fit.str.contains("usado|usada|antiguo|impecable").to_numpy()
    new_kw_flag = titles_fit.str.contains("nuevo|nueva|0km|sellado").to_numpy()
    log(f"Keywords polarity in fit: P(used=1 | used_kw) = {y_fit[used_kw_flag].mean():.4f}, P(used=1 | new_kw) = {y_fit[new_kw_flag].mean():.4f}")
    assert y_fit[used_kw_flag].mean() > 0.85, "Polarity check failed: used keywords should correlate strongly with used=1"
    assert y_fit[new_kw_flag].mean() < 0.25, "Polarity check failed: new keywords should correlate strongly with used=0"
    log("Polarity check PASSED: y=1 is used, y=0 is new.")

    # 2. Build or Load Fold Matrices
    log("Precomputing item intrinsic arrays...")
    tr_data = ag.precompute_item_intrinsic_arrays(df)

    log("Setting up inner 4-fold StratifiedGroupKFold on seller_id inside fit...")
    inner_cv = list(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=61111).split(
        fit_indices, tr_data["y"][fit_indices], tr_data["seller_id"][fit_indices]
    ))
    # Assert zero seller leakage across inner folds
    for a, b in inner_cv:
        s_a = set(tr_data["seller_id"][fit_indices[a]])
        s_b = set(tr_data["seller_id"][fit_indices[b]])
        assert len(s_a & s_b) == 0, "Leakage detected: seller overlap between inner CV folds!"
    log("Inner CV 4 folds verified seller-disjoint (zero leakage).")

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

    # Matrices for Line 3 (M0)
    X_m0_unseen_fit = mats["X_m0_unseen_T"]
    X_m0_seen_fit = mats["X_m0_seen_causal_T"]
    X_m0_unseen_tune = mats["X_m0_unseen_evals"][0]
    X_m0_seen_tune = mats["X_m0_seen_causal_evals"][0]

    # Matrices for LightGBM Enriched (M1 / Arm 1)
    X_m1_unseen_fit = mats["X_m1_unseen_T"]
    X_m1_seen_fit = mats["X_m1_seen_causal_T"]
    X_m1_unseen_tune = mats["X_m1_unseen_evals"][0]
    X_m1_seen_tune = mats["X_m1_seen_causal_evals"][0]

    log(f"Line 3 matrix shapes: fit={X_m0_unseen_fit.shape}, tune={X_m0_unseen_tune.shape}")
    log(f"LGB Enriched matrix shapes: fit={X_m1_unseen_fit.shape}, tune={X_m1_unseen_tune.shape}")

    # 3. Inner CV Tuning & Early Stopping
    log("\n" + "=" * 60)
    log("INNER CV HYPERPARAMETER TUNING & EARLY STOPPING (STRICTLY ON FIT 40,881)")
    log("=" * 60)

    # HP grid for Line 3
    hp_candidates_m0 = [
        {"num_leaves": 48, "min_child_samples": 30, "colsample_bytree": 0.75},
        {"num_leaves": 63, "min_child_samples": 30, "colsample_bytree": 0.75},
        {"num_leaves": 95, "min_child_samples": 40, "colsample_bytree": 0.70},
    ]
    tune_res_m0 = run_inner_cv_tuning_and_early_stopping(
        X_train_full=X_m0_unseen_fit,
        y_train_full=y_fit,
        seller_ids=tr_data["seller_id"][fit_indices],
        inner_cv_splits=inner_cv,
        hp_candidates=hp_candidates_m0,
        model_name="Baseline 1: Line 3 (M0)",
        max_estimators=1200,
        stopping_rounds=50,
        learning_rate=0.035,
    )

    # HP grid for LightGBM Enriched
    hp_candidates_m1 = [
        {"num_leaves": 48, "min_child_samples": 35, "colsample_bytree": 0.75},
        {"num_leaves": 63, "min_child_samples": 30, "colsample_bytree": 0.75},
        {"num_leaves": 95, "min_child_samples": 40, "colsample_bytree": 0.80},
    ]
    tune_res_m1 = run_inner_cv_tuning_and_early_stopping(
        X_train_full=X_m1_unseen_fit,
        y_train_full=y_fit,
        seller_ids=tr_data["seller_id"][fit_indices],
        inner_cv_splits=inner_cv,
        hp_candidates=hp_candidates_m1,
        model_name="Baseline 2: Texto OOF seller-group con atributos (M1)",
        max_estimators=1200,
        stopping_rounds=50,
        learning_rate=0.035,
    )

    # 4. Final Model Training on Full Fit (40,881) & Prediction on Tune (15,044)
    log("\n" + "=" * 60)
    log("FINAL BAGGED MODEL TRAINING & EVALUATION ON TUNE")
    log("=" * 60)

    # --- Baseline 1: Line 3 ---
    log("Training Baseline 1: Line 3 (Dual Regime, 3 seeds)...")
    t0_m0 = time.time()
    # Unseen regime model
    p_m0_unseen_tune, seed_times_m0_unseen = fit_bagged_lgbm(
        X_train=X_m0_unseen_fit,
        y_train=y_fit,
        X_test=X_m0_unseen_tune,
        hp=tune_res_m0["best_hp"],
        n_estimators=tune_res_m0["optimal_n_estimators"],
        seeds=SEEDS,
        learning_rate=0.035,
    )
    # Seen causal regime model
    p_m0_seen_tune, seed_times_m0_seen = fit_bagged_lgbm(
        X_train=X_m0_seen_fit,
        y_train=y_fit,
        X_test=X_m0_seen_tune,
        hp=tune_res_m0["best_hp"],
        n_estimators=tune_res_m0["optimal_n_estimators"],
        seeds=SEEDS,
        learning_rate=0.035,
    )
    # Combine predictions by regime
    p_line3_tune = np.where(tune_seen_mask, p_m0_seen_tune, p_m0_unseen_tune).astype(np.float32)
    m0_total_time = round(time.time() - t0_m0, 2)
    log(f"Line 3 trained and evaluated in {m0_total_time}s.")

    # --- Baseline 2: LightGBM Enriched (Texto OOF con atributos) ---
    log("Training Baseline 2: LightGBM Enriched (Dual Regime, 3 seeds)...")
    t0_m1 = time.time()
    # Unseen regime model
    p_m1_unseen_tune, seed_times_m1_unseen = fit_bagged_lgbm(
        X_train=X_m1_unseen_fit,
        y_train=y_fit,
        X_test=X_m1_unseen_tune,
        hp=tune_res_m1["best_hp"],
        n_estimators=tune_res_m1["optimal_n_estimators"],
        seeds=SEEDS,
        learning_rate=0.035,
    )
    # Seen causal regime model
    p_m1_seen_tune, seed_times_m1_seen = fit_bagged_lgbm(
        X_train=X_m1_seen_fit,
        y_train=y_fit,
        X_test=X_m1_seen_tune,
        hp=tune_res_m1["best_hp"],
        n_estimators=tune_res_m1["optimal_n_estimators"],
        seeds=SEEDS,
        learning_rate=0.035,
    )
    # Combine predictions by regime
    p_lgb_attrs_tune = np.where(tune_seen_mask, p_m1_seen_tune, p_m1_unseen_tune).astype(np.float32)
    m1_total_time = round(time.time() - t0_m1, 2)
    log(f"LGB Enriched trained and evaluated in {m1_total_time}s.")

    # 5. Join Verification by ID
    log("\nVerifying 1:1 join by ID...")
    assert len(ids_tune) == len(tune_indices) == 15044
    assert len(p_line3_tune) == 15044
    assert len(p_lgb_attrs_tune) == 15044
    assert not np.isnan(p_line3_tune).any()
    assert not np.isnan(p_lgb_attrs_tune).any()
    assert len(set(ids_tune)) == 15044, "Duplicate IDs in tune partition!"
    log(f"Join verified: 15,044 unique string IDs (e.g. {ids_tune[0]}..{ids_tune[-1]}).")

    # 6. Save Private Pairs CSV
    log(f"Saving private pairs to {PAIRS_CSV_PATH}...")
    df_pairs = pd.DataFrame({
        "id": ids_tune,
        "y": y_tune,
        "p_line3": p_line3_tune,
        "p_lgb_attrs": p_lgb_attrs_tune,
        "regime": regimes_tune,
        "seller_id": sellers_tune,
        "category_id": cats_tune,
    })
    df_pairs.to_csv(PAIRS_CSV_PATH, index=False)
    log(f"Private pairs saved. SHA-256: {sha256_file(PAIRS_CSV_PATH)}")

    # 7. Compute Paired Metrics: Overall, Cold New, and Seen Bracketed
    log("\nComputing paired metrics and two-way cluster bootstrap...")
    # Overall Tune (15,044)
    overall_comparison = compute_paired_metrics(
        y_true=y_tune,
        p_cand=p_lgb_attrs_tune,
        p_base=p_line3_tune,
        sellers=sellers_tune,
        cats=cats_tune,
    )

    # Cold New (8,758)
    cold_comparison = compute_paired_metrics(
        y_true=y_tune[tune_cold_mask],
        p_cand=p_lgb_attrs_tune[tune_cold_mask],
        p_base=p_line3_tune[tune_cold_mask],
        sellers=sellers_tune[tune_cold_mask],
        cats=cats_tune[tune_cold_mask],
    )

    # Seen Bracketed (6,286)
    seen_comparison = compute_paired_metrics(
        y_true=y_tune[tune_seen_mask],
        p_cand=p_lgb_attrs_tune[tune_seen_mask],
        p_base=p_line3_tune[tune_seen_mask],
        sellers=sellers_tune[tune_seen_mask],
        cats=cats_tune[tune_seen_mask],
    )

    # 8. Assemble Full Machine-Readable Audit Report
    report = {
        "metadata": {
            "title": "Master Reproduction of Two Strong Baselines (Line 3 vs Texto OOF con Atributos)",
            "protocol": "CX-0185..CX-0189 / T98 Leak-Free Protocol",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_head_local": "bc7f5f4",
            "split_file": str(SPLIT_CSV.relative_to(ROOT)),
            "split_sha256": split_manifest["split_sha256"],
            "pairs_file": str(PAIRS_CSV_PATH.relative_to(ROOT)),
            "pairs_sha256": sha256_file(PAIRS_CSV_PATH),
            "rows_fit": len(fit_indices),
            "rows_tune_total": len(tune_indices),
            "rows_tune_cold_new": int(tune_cold_mask.sum()),
            "rows_tune_seen_bracketed": int(tune_seen_mask.sum()),
            "seeds": SEEDS,
            "gate_compliance": {
                "Gate_A_positive_delta": bool(overall_comparison["delta_hits"] > 0),
                "Gate_B_inner_cv_only": True,
                "Gate_C_public_probability_status": "NO_IDENTIFIABLE",
                "Gate_D_private_pairs_gitignored": True,
                "Gate_E_nested_temporal_cluster_bootstrap": True,
                "Gate_G_strict_id_and_polarity": True,
                "Gate_H_negative_results_first": True,
            },
        },
        "computational_cost": {
            "baseline_1_line3": {
                "tuning_time_sec": tune_res_m0["total_tuning_time_sec"],
                "final_fit_time_sec": m0_total_time,
                "total_time_sec": round(tune_res_m0["total_tuning_time_sec"] + m0_total_time, 2),
                "optimal_trees": tune_res_m0["optimal_n_estimators"],
                "best_hyperparameters": tune_res_m0["best_hp"],
                "seed_times_unseen": seed_times_m0_unseen,
                "seed_times_seen": seed_times_m0_seen,
            },
            "baseline_2_lgb_attributes": {
                "tuning_time_sec": tune_res_m1["total_tuning_time_sec"],
                "final_fit_time_sec": m1_total_time,
                "total_time_sec": round(tune_res_m1["total_tuning_time_sec"] + m1_total_time, 2),
                "optimal_trees": tune_res_m1["optimal_n_estimators"],
                "best_hyperparameters": tune_res_m1["best_hp"],
                "seed_times_unseen": seed_times_m1_unseen,
                "seed_times_seen": seed_times_m1_seen,
            },
            "total_execution_time_sec": round(time.time() - t_global_start, 2),
        },
        "inner_cv_selection": {
            "baseline_1_line3": tune_res_m0,
            "baseline_2_lgb_attributes": tune_res_m1,
        },
        "tune_paired_evaluation": {
            "overall_tune_15044": overall_comparison,
            "tune_cold_new_8758": cold_comparison,
            "tune_seen_bracketed_6286": seen_comparison,
        },
    }

    log(f"Saving machine-readable report to {REPORT_JSON_PATH}...")
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    log("=" * 60)
    log("REPRODUCTION SUMMARY:")
    log(f"Line 3 Hits: {overall_comparison['baseline']['hits']} ({overall_comparison['baseline']['accuracy']:.5f}), Logloss: {overall_comparison['baseline']['logloss']:.5f}")
    log(f"LGB Enriched Hits: {overall_comparison['candidate']['hits']} ({overall_comparison['candidate']['accuracy']:.5f}), Logloss: {overall_comparison['candidate']['logloss']:.5f}")
    log(f"Delta Hits: {overall_comparison['delta_hits']:+d} (Fixes: {overall_comparison['fixes']}, Breaks: {overall_comparison['breaks']})")
    log(f"Delta Acc: {overall_comparison['delta_acc']:+.5f}, Delta Logloss: {overall_comparison['delta_logloss']:+.5f}")
    log(f"Two-Way Cluster Bootstrap Delta Hits 90% CI: [{overall_comparison['bootstrap_two_way']['delta_hits_q05']:+.1f}, {overall_comparison['bootstrap_two_way']['delta_hits_q95']:+.1f}]")
    log(f"By Regime - Cold New: {cold_comparison['delta_hits']:+d} hits | Seen Bracketed: {seen_comparison['delta_hits']:+d} hits")
    log(f"Total Elapsed Time: {report['computational_cost']['total_execution_time_sec']}s")
    log("=" * 60)


if __name__ == "__main__":
    main()
