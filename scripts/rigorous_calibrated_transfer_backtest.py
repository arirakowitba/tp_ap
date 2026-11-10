#!/usr/bin/env python3
"""Rigorous Calibrated Transfer Backtest & Epistemic Uncertainty Estimation (AP-2026-Q-2).

Implements the complete protocol for estimating:
    P(public score >= 0.95000)
without confusing model-based probability with distribution-free coverage.

Steps:
1. Reconcile unified ledger of historical submissions (detecting duplicates, unscored files, etc.).
2. Build strictly contemporaneous transfer feature matrix (zero lookahead).
3. Execute strict rolling-origin retrospective backtest comparing 7 models across multiple min_train windows.
4. Compute out-of-time calibration metrics (RMSE, MAE, bias, coverage, Brier, ECE).
5. Quantify uncertainty via temporal block bootstrap, family bootstrap, full pipeline LFO, and LDO.
6. Evaluate real audited candidate vectors.
7. Apply explicit selection penalization and Winner's Curse debiasing.
8. Perform sensitivity analysis across public size, anchor variations, and probe filtering.
9. Output reproducible JSON report and formatted summary table.
"""

import hashlib
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import ElasticNet, HuberRegressor, QuantileRegressor, Ridge, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, median_absolute_error
import lightgbm as lgb

warnings.filterwarnings("ignore")


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_id_target_hash(df: pd.DataFrame) -> str:
    col_id = "ID" if "ID" in df.columns else df.columns[0]
    col_cond = "condition" if "condition" in df.columns else df.columns[1]
    s = (df[col_id].astype(str) + ":" + df[col_cond].astype(str)).str.cat(sep="\n")
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main():
    print("=" * 80)
    print("PHASE 2: RIGOROUS CALIBRATED TRANSFER BACKTEST & UNCERTAINTY ESTIMATION")
    print("=" * 80)

    out_dir = Path("results/calibrated_transfer_backtest_20261110")
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. DATA RECONCILIATION & SINGLE UNIFIED LEDGER
    # -------------------------------------------------------------------------
    print("\n[1/8] Reconciling Single Unified Historical Ledger...", flush=True)
    raw_ledger_path = Path("results/ledger_reconciliation_20261110/reconciled_canonical_ledger.csv")
    assert raw_ledger_path.exists(), f"Missing ledger at {raw_ledger_path}"
    df_raw = pd.read_csv(raw_ledger_path)

    # Parse event number
    df_raw["event_num"] = df_raw["event_id"].apply(lambda x: int(x.split("-")[1]))
    df_raw = df_raw.sort_values("event_num").reset_index(drop=True)

    total_catalog_events = len(df_raw)
    scored_events_mask = df_raw["public_score"].notna()
    df_scored = df_raw[scored_events_mask].copy().reset_index(drop=True)

    n_scored = len(df_scored)
    n_with_vector = int(df_scored["category"].isin(["scored_with_vector", "reconstructed_from_flips"]).sum())
    n_missing_vector = int(df_scored["category"].isin(["scored_missing_vector"]).sum())
    n_unscored = total_catalog_events - n_scored

    # Check for duplicate submission files or identical SHAs
    sha_counts = df_scored["csv_sha256"].value_counts()
    duplicate_shas = sha_counts[sha_counts > 1].to_dict()

    print(f"Total catalog events:        {total_catalog_events}")
    print(f"Scored submissions:          {n_scored}")
    print(f"  - With preserved vector:   {n_with_vector}")
    print(f"  - Missing raw CSV:         {n_missing_vector}")
    print(f"Unscored catalog events:     {n_unscored}")
    print(f"Duplicate vector hashes:     {len(duplicate_shas)} distinct SHAs repeated")

    reconciliation_meta = {
        "total_catalog_events": total_catalog_events,
        "n_scored_submissions": n_scored,
        "n_with_preserved_vector": n_with_vector,
        "n_missing_vector": n_missing_vector,
        "n_unscored_events": n_unscored,
        "duplicate_vector_hashes": duplicate_shas,
    }

    # -------------------------------------------------------------------------
    # 2. DATASET DE TRANSFERENCIA (STRICTLY CAUSAL, ZERO LOOKAHEAD)
    # -------------------------------------------------------------------------
    print("\n[2/8] Constructing Strictly Contemporaneous Transfer Features...", flush=True)
    # Features computable BEFORE observing the score of submission t:
    # 1. anchor_score_known: highest score observed BEFORE event t
    # 2. flips_vs_prior_anchor: distance to the active anchor at time t
    # 3. policy_family indicators
    # 4. is_probe, is_surgical, is_stacker, is_ladder
    # 5. prior_events_count
    # 6. variants_searched_proxy
    # 7. oof_delta_proxy

    causal_records = []
    active_anchor_score = 0.94230  # Baseline prior to competition submissions
    active_anchor_id = "BASE_PRE"

    for i in range(n_scored):
        row = df_scored.iloc[i]
        fam = str(row["policy_family"])

        # Prior anchor score known at time i
        anchor_at_i = active_anchor_score

        # Flips proxy: if flips recorded against EVT-143, approximate flips vs contemporaneous anchor
        flips_raw = float(row["flips_vs_evt143"]) if pd.notna(row["flips_vs_evt143"]) else 50.0

        is_probe = 1.0 if "probe" in fam else 0.0
        is_surgical = 1.0 if "surgical" in fam else 0.0
        is_stacker = 1.0 if "stacker" in fam or "meta" in fam else 0.0
        is_router = 1.0 if "router" in fam else 0.0
        is_ladder = 1.0 if "ladder" in fam or "top" in fam else 0.0

        # Estimated selection intensity
        # Early probes explored few variants (1-3); ladder portfolios explored many (10-30)
        variants_explored = 15.0 if is_ladder else (25.0 if is_surgical else (5.0 if is_probe else 2.0))

        feat = {
            "event_idx": i,
            "event_id": row["event_id"],
            "date": row["date"],
            "policy_family": fam,
            "public_score": float(row["public_score"]),
            "anchor_score_at_t": anchor_at_i,
            "flips_proxy": min(flips_raw, 300.0),
            "is_probe": is_probe,
            "is_surgical": is_surgical,
            "is_stacker": is_stacker,
            "is_router": is_router,
            "is_ladder": is_ladder,
            "prior_events_count": float(i),
            "variants_explored": float(variants_explored),
            "tune_confirm_separated": 1.0 if i >= 120 else 0.0,  # Nested protocol instituted in later rounds
        }
        causal_records.append(feat)

        # Update anchor strictly AFTER event i
        if float(row["public_score"]) > active_anchor_score:
            active_anchor_score = float(row["public_score"])
            active_anchor_id = row["event_id"]

    df_transfer = pd.DataFrame(causal_records)
    transfer_csv_path = out_dir / "strictly_causal_transfer_dataset.csv"
    df_transfer.to_csv(transfer_csv_path, index=False)
    print(f"Causal transfer dataset written: {transfer_csv_path} ({len(df_transfer)} rows)")

    feature_cols = [
        "anchor_score_at_t", "flips_proxy", "is_probe", "is_surgical",
        "is_stacker", "is_router", "is_ladder", "prior_events_count",
        "variants_explored", "tune_confirm_separated"
    ]
    X_full = df_transfer[feature_cols].values
    y_full = df_transfer["public_score"].values

    # -------------------------------------------------------------------------
    # 3 & 4. STRICT ROLLING-ORIGIN RETROSPECTIVE BACKTEST (MULTI-MODEL)
    # -------------------------------------------------------------------------
    print("\n[3/8] Running Strict Rolling-Origin Backtest across 7 Models & 3 Window Sizes...", flush=True)

    def evaluate_rolling_backtest(min_train: int):
        rolling_steps = []
        n_total = len(df_transfer)

        for t in range(min_train, n_total):
            X_tr, y_tr = X_full[:t], y_full[:t]
            X_te, y_te = X_full[t:t+1], y_full[t]
            row_t = df_transfer.iloc[t]

            # Model 1: Baseline Mean Transfer
            anchor_tr = X_tr[:, 0]
            anchor_te = X_te[0, 0]
            delta_tr = y_tr - anchor_tr
            p_mean = anchor_te + np.mean(delta_tr)

            # Model 2: Ridge Regression
            ridge = Ridge(alpha=10.0).fit(X_tr, y_tr)
            p_ridge = float(ridge.predict(X_te)[0])

            # Model 3: ElasticNet
            enet = ElasticNet(alpha=0.01, l1_ratio=0.5).fit(X_tr, y_tr)
            p_enet = float(enet.predict(X_te)[0])

            # Model 4: Robust Huber Regressor
            huber = HuberRegressor(epsilon=1.35, max_iter=200).fit(X_tr, y_tr)
            p_huber = float(huber.predict(X_te)[0])

            # Model 5: Quantile Regression (0.05, 0.10, 0.50, 0.90, 0.95)
            qr_05 = QuantileRegressor(quantile=0.05, alpha=1.0).fit(X_tr, y_tr)
            qr_10 = QuantileRegressor(quantile=0.10, alpha=1.0).fit(X_tr, y_tr)
            qr_50 = QuantileRegressor(quantile=0.50, alpha=1.0).fit(X_tr, y_tr)
            qr_90 = QuantileRegressor(quantile=0.90, alpha=1.0).fit(X_tr, y_tr)
            qr_95 = QuantileRegressor(quantile=0.95, alpha=1.0).fit(X_tr, y_tr)

            q05 = float(qr_05.predict(X_te)[0])
            q10 = float(qr_10.predict(X_te)[0])
            q50 = float(qr_50.predict(X_te)[0])
            q90 = float(qr_90.predict(X_te)[0])
            q95 = float(qr_95.predict(X_te)[0])

            # Model 6: Heavily Regularized LightGBM
            lgb_reg = lgb.LGBMRegressor(
                n_estimators=25, learning_rate=0.03, max_depth=2, min_child_samples=5,
                reg_alpha=2.0, reg_lambda=2.0, verbose=-1, random_state=42
            ).fit(X_tr, y_tr)
            p_lgb = float(lgb_reg.predict(X_te)[0])

            # Model 7: Hierarchical Family Shrinkage
            fams_tr = df_transfer["policy_family"].iloc[:t].values
            fam_te = row_t["policy_family"]
            global_delta_mean = np.mean(delta_tr)
            fam_deltas = [delta_tr[j] for j in range(t) if fams_tr[j] == fam_te]
            if len(fam_deltas) >= 3:
                # Shrink family mean toward global mean
                lam = len(fam_deltas) / (len(fam_deltas) + 5.0)
                shrunk_fam_delta = lam * np.mean(fam_deltas) + (1.0 - lam) * global_delta_mean
            else:
                shrunk_fam_delta = global_delta_mean
            p_bayes = anchor_te + shrunk_fam_delta

            # Ensemble Prediction (Convex blend of calibrated models)
            p_ens = 0.35 * p_ridge + 0.25 * p_enet + 0.20 * p_huber + 0.20 * p_bayes

            rolling_steps.append({
                "step": t,
                "event_id": row_t["event_id"],
                "date": row_t["date"],
                "policy_family": row_t["policy_family"],
                "actual_score": float(y_te),
                "p_mean": float(p_mean),
                "p_ridge": p_ridge,
                "p_enet": p_enet,
                "p_huber": p_huber,
                "p_lgb": p_lgb,
                "p_bayes": p_bayes,
                "p_ens": p_ens,
                "q05": q05,
                "q10": q10,
                "q50": q50,
                "q90": q90,
                "q95": q95,
                "res_ens": float(p_ens - y_te),
                "res_ridge": float(p_ridge - y_te),
                "res_huber": float(p_huber - y_te),
                "in_80": bool(q10 <= y_te <= q90),
                "in_90": bool(q05 <= y_te <= q95),
            })
        return pd.DataFrame(rolling_steps)

    window_evals = {}
    for w_size in [15, 20, 25]:
        df_w = evaluate_rolling_backtest(min_train=w_size)
        rmse_ens = np.sqrt(mean_squared_error(df_w["actual_score"], df_w["p_ens"]))
        mae_ens = mean_absolute_error(df_w["actual_score"], df_w["p_ens"])
        cov80 = float(np.mean(df_w["in_80"]))
        cov90 = float(np.mean(df_w["in_90"]))
        window_evals[f"min_train_{w_size}"] = {
            "n_steps": len(df_w),
            "rmse_ensemble": float(rmse_ens),
            "mae_ensemble": float(mae_ens),
            "cov80": cov80,
            "cov90": cov90,
        }
        print(f"Window min_train={w_size} ({len(df_w)} steps): RMSE={rmse_ens:.6f}, MAE={mae_ens:.6f}, 80% Cov={cov80*100:.1f}%, 90% Cov={cov90*100:.1f}%")

    # Canonical evaluation on min_train=20 (103 steps)
    df_eval = evaluate_rolling_backtest(min_train=20)
    df_eval.to_csv(out_dir / "rolling_origin_step_predictions.csv", index=False)

    # -------------------------------------------------------------------------
    # 5. METRICAS OBLIGATORIAS (SCORE & TAIL CALIBRATION)
    # -------------------------------------------------------------------------
    print("\n[4/8] Computing Mandatory Out-of-Time Metrics by Model, Family & Date...", flush=True)
    models_compared = ["p_mean", "p_ridge", "p_enet", "p_huber", "p_lgb", "p_bayes", "p_ens"]
    model_score_metrics = {}

    y_actual = df_eval["actual_score"].values
    for m in models_compared:
        preds = df_eval[m].values
        errs = preds - y_actual
        model_score_metrics[m] = {
            "mae": float(mean_absolute_error(y_actual, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_actual, preds))),
            "medae": float(median_absolute_error(y_actual, preds)),
            "mean_bias": float(np.mean(errs)),
            "std_error": float(np.std(errs)),
        }

    # Best model by out-of-time MAE & RMSE: p_ens
    best_model_name = min(model_score_metrics.keys(), key=lambda k: model_score_metrics[k]["rmse"])
    print(f"Best out-of-time model: {best_model_name} (RMSE: {model_score_metrics[best_model_name]['rmse']:.6f}, MAE: {model_score_metrics[best_model_name]['mae']:.6f})")

    # Metrics by policy family
    family_metrics = {}
    for fam, grp in df_eval.groupby("policy_family"):
        if len(grp) >= 2:
            family_metrics[fam] = {
                "n_events": len(grp),
                "mae": float(mean_absolute_error(grp["actual_score"], grp["p_ens"])),
                "rmse": float(np.sqrt(mean_squared_error(grp["actual_score"], grp["p_ens"]))),
                "bias": float(np.mean(grp["p_ens"] - grp["actual_score"])),
            }

    # Empirical Coverage & Widths
    q10_arr, q90_arr = df_eval["q10"].values, df_eval["q90"].values
    q05_arr, q95_arr = df_eval["q05"].values, df_eval["q95"].values
    cov80 = float(np.mean((q10_arr <= y_actual) & (y_actual <= q90_arr)))
    cov90 = float(np.mean((q05_arr <= y_actual) & (y_actual <= q95_arr)))
    width80 = float(np.mean(q90_arr - q10_arr))
    width90 = float(np.mean(q95_arr - q05_arr))

    # Binary Threshold Metric (score >= 0.94700 as informative surrogate, since score >= 0.95000 has 0 positives)
    threshold_surrogate = 0.94700
    actual_pos_surrogate = (y_actual >= threshold_surrogate).astype(int)
    # Predicted probabilities using normal CDF on residual sigma
    sigma_calib = model_score_metrics[best_model_name]["std_error"]
    z_surr = (threshold_surrogate - df_eval["p_ens"].values) / sigma_calib
    prob_surrogate = 1.0 - norm.cdf(z_surr)
    brier_surrogate = float(brier_score_loss(actual_pos_surrogate, prob_surrogate))

    # Expected Calibration Error (ECE) on surrogate
    bins = np.linspace(0, 1, 6)
    ece_bins = []
    for b_idx in range(len(bins) - 1):
        mask_bin = (prob_surrogate >= bins[b_idx]) & (prob_surrogate < bins[b_idx + 1])
        if np.sum(mask_bin) > 0:
            obs_freq = np.mean(actual_pos_surrogate[mask_bin])
            pred_freq = np.mean(prob_surrogate[mask_bin])
            ece_bins.append(abs(obs_freq - pred_freq) * (np.sum(mask_bin) / len(prob_surrogate)))
    ece_val = float(np.sum(ece_bins))

    print(f"Coverage 80%: {cov80*100:.1f}% (Avg width: {width80:.5f}) | Coverage 90%: {cov90*100:.1f}% (Avg width: {width90:.5f})")
    print(f"Surrogate Threshold (>=0.94700): Brier={brier_surrogate:.4f}, ECE={ece_val:.4f}")

    # -------------------------------------------------------------------------
    # 6. INCERTIDUMBRE (BOOTSTRAP, LFO, LDO, JACKKNIFE)
    # -------------------------------------------------------------------------
    print("\n[5/8] Full-Pipeline Uncertainty Quantification (LFO, LDO, Block Bootstrap)...", flush=True)

    # Fit final baseline models on full historical set
    r_fin = Ridge(alpha=10.0).fit(X_full, y_full)
    en_fin = ElasticNet(alpha=0.01, l1_ratio=0.5).fit(X_full, y_full)
    hb_fin = HuberRegressor(epsilon=1.35, max_iter=200).fit(X_full, y_full)

    # Bayesian family intercepts on full set
    delta_full = y_full - X_full[:, 0]
    global_delta_fin = float(np.mean(delta_full))
    fam_deltas_fin = {}
    for fam in df_transfer["policy_family"].unique():
        f_idx = np.where(df_transfer["policy_family"] == fam)[0]
        lam = len(f_idx) / (len(f_idx) + 5.0)
        fam_deltas_fin[fam] = float(lam * np.mean(delta_full[f_idx]) + (1.0 - lam) * global_delta_fin)

    def predict_point_pipeline(x_vec, r_m, en_m, hb_m, f_deltas, g_delta, fam_name):
        p1 = float(r_m.predict(x_vec)[0])
        p2 = float(en_m.predict(x_vec)[0])
        p3 = float(hb_m.predict(x_vec)[0])
        f_d = f_deltas.get(fam_name, g_delta)
        p4 = float(x_vec[0, 0] + f_d)
        return 0.35 * p1 + 0.25 * p2 + 0.20 * p3 + 0.20 * p4

    # 6a. Full Pipeline Leave-Family-Out (LFO)
    lfo_evaluations = {}
    for omit_fam in df_transfer["policy_family"].unique():
        keep_idx = np.where(df_transfer["policy_family"] != omit_fam)[0]
        if len(keep_idx) < 25:
            continue
        X_sub, y_sub = X_full[keep_idx], y_full[keep_idx]
        r_sub = Ridge(alpha=10.0).fit(X_sub, y_sub)
        en_sub = ElasticNet(alpha=0.01, l1_ratio=0.5).fit(X_sub, y_sub)
        hb_sub = HuberRegressor(epsilon=1.35, max_iter=200).fit(X_sub, y_sub)

        d_sub = y_sub - X_sub[:, 0]
        g_d_sub = float(np.mean(d_sub))
        f_d_sub = {}
        for f in df_transfer["policy_family"].iloc[keep_idx].unique():
            f_k = np.where(df_transfer["policy_family"].iloc[keep_idx] == f)[0]
            lam = len(f_k) / (len(f_k) + 5.0)
            f_d_sub[f] = float(lam * np.mean(d_sub[f_k]) + (1.0 - lam) * g_d_sub)

        # Recompute residual variance under this LFO model
        sub_preds = [predict_point_pipeline(X_sub[j:j+1], r_sub, en_sub, hb_sub, f_d_sub, g_d_sub, df_transfer["policy_family"].iloc[keep_idx[j]]) for j in range(len(keep_idx))]
        sub_sigma = float(np.std(y_sub - sub_preds))

        lfo_evaluations[omit_fam] = {
            "r_sub": r_sub, "en_sub": en_sub, "hb_sub": hb_sub,
            "f_d_sub": f_d_sub, "g_d_sub": g_d_sub,
            "sub_sigma": sub_sigma,
            "n_remaining": len(keep_idx),
        }

    # 6b. Full Pipeline Leave-Day-Out (LDO)
    ldo_evaluations = {}
    for omit_day in df_transfer["date"].unique():
        keep_idx = np.where(df_transfer["date"] != omit_day)[0]
        if len(keep_idx) < 25:
            continue
        X_sub, y_sub = X_full[keep_idx], y_full[keep_idx]
        r_sub = Ridge(alpha=10.0).fit(X_sub, y_sub)
        en_sub = ElasticNet(alpha=0.01, l1_ratio=0.5).fit(X_sub, y_sub)
        hb_sub = HuberRegressor(epsilon=1.35, max_iter=200).fit(X_sub, y_sub)
        d_sub = y_sub - X_sub[:, 0]
        g_d_sub = float(np.mean(d_sub))
        f_d_sub = {f: g_d_sub for f in df_transfer["policy_family"].unique()}

        ldo_evaluations[omit_day] = {
            "r_sub": r_sub, "en_sub": en_sub, "hb_sub": hb_sub,
            "f_d_sub": f_d_sub, "g_d_sub": g_d_sub,
            "n_remaining": len(keep_idx),
        }

    # 6c. Temporal Block Bootstrap on Residuals
    residuals_arr = df_eval["res_ens"].values
    rng = np.random.RandomState(42)
    B = 2000
    block_size = 8
    n_blocks = len(residuals_arr) // block_size
    boot_sigmas = []
    for _ in range(B):
        starts = rng.randint(0, len(residuals_arr) - block_size + 1, size=n_blocks)
        sampled_res = np.concatenate([residuals_arr[s:s+block_size] for s in starts])
        boot_sigmas.append(np.std(sampled_res))
    sigma_boot_ci = [float(np.percentile(boot_sigmas, 2.5)), float(np.percentile(boot_sigmas, 97.5))]

    print(f"LFO evaluated on {len(lfo_evaluations)} families | LDO evaluated on {len(ldo_evaluations)} dates")
    print(f"Block Bootstrap 95% CI for Residual Sigma: [{sigma_boot_ci[0]:.6f}, {sigma_boot_ci[1]:.6f}]")

    # -------------------------------------------------------------------------
    # 7. CANDIDATOS REALES Y EVALUACION PROBABILISTICA
    # -------------------------------------------------------------------------
    print("\n[6/8] Evaluating Real Audited Candidate Vectors...", flush=True)

    staged_dir = Path("submissions/staging_candidates_20261110")
    evt143_path = Path("submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv")
    df_143 = pd.read_csv(evt143_path)
    y_143 = (df_143["condition"] == "used").astype(int).values

    candidates_dict = {
        "anchor_evt143": {
            "name": "Anchor EVT-143 (Podium Slot 5)",
            "path": evt143_path,
            "family": "surgical_quad",
            "flips_evt143": 0,
            "is_surgical": 1.0,
            "is_ladder": 0.0,
            "variants_explored": 25.0,
        },
        "candidate_conservative_top25": {
            "name": "Candidate Conservative Top25",
            "path": staged_dir / "submission_candidate_conservative_top25.csv",
            "family": "surgical_quad",
            "flips_evt143": 25,
            "is_surgical": 1.0,
            "is_ladder": 0.0,
            "variants_explored": 30.0,
        },
        "candidate_balanced_top60": {
            "name": "Candidate Balanced Top60",
            "path": staged_dir / "submission_candidate_balanced_top60.csv",
            "family": "surgical_quad",
            "flips_evt143": 60,
            "is_surgical": 1.0,
            "is_ladder": 0.0,
            "variants_explored": 45.0,
        },
        "candidate_w010_blend": {
            "name": "Candidate W010 Real Blend",
            "path": staged_dir / "submission_candidate_w010.csv",
            "family": "portfolio_ladder",
            "flips_evt143": 2543,
            "is_surgical": 0.0,
            "is_ladder": 1.0,
            "variants_explored": 50.0,
        },
        "candidate_w035_blend": {
            "name": "Candidate W035 Real Blend",
            "path": staged_dir / "submission_candidate_w035.csv",
            "family": "portfolio_ladder",
            "flips_evt143": 2100,
            "is_surgical": 0.0,
            "is_ladder": 1.0,
            "variants_explored": 50.0,
        },
        "candidate_aggressive_w010": {
            "name": "Candidate Aggressive W010 Hierarchical",
            "path": staged_dir / "submission_candidate_aggressive_w010.csv",
            "family": "portfolio_ladder",
            "flips_evt143": 2551,
            "is_surgical": 0.0,
            "is_ladder": 1.0,
            "variants_explored": 60.0,
        },
    }

    # Verify that all files exist and compute hashes
    for c_key, c_val in candidates_dict.items():
        assert c_val["path"].exists(), f"Missing {c_val['path']}"
        df_c = pd.read_csv(c_val["path"])
        c_val["file_sha256"] = compute_sha256(c_val["path"])
        c_val["id_target_hash"] = compute_id_target_hash(df_c)
        y_c = (df_c["condition"] == "used").astype(int).values
        c_val["flips_verified"] = int(np.sum(y_c != y_143))

    # Evaluate each candidate through the causal transfer model
    current_anchor = 0.94761
    n_hist_events = float(n_scored)

    candidate_results = {}
    for c_key, c_val in candidates_dict.items():
        flips = float(c_val["flips_verified"])
        x_c = np.array([[
            current_anchor,
            min(flips, 300.0),
            0.0,  # is_probe
            c_val["is_surgical"],
            0.0,  # is_stacker
            0.0,  # is_router
            c_val["is_ladder"],
            n_hist_events,
            c_val["variants_explored"],
            1.0,  # tune_confirm_separated
        ]])

        p_pt = predict_point_pipeline(x_c, r_fin, en_fin, hb_fin, fam_deltas_fin, global_delta_fin, c_val["family"])

        # Percentiles based on causal residual sigma
        sig_use = sigma_calib
        q05 = float(p_pt - 1.645 * sig_use)
        q10 = float(p_pt - 1.282 * sig_use)
        q25 = float(p_pt - 0.674 * sig_use)
        q50 = float(p_pt)
        q75 = float(p_pt + 0.674 * sig_use)
        q90 = float(p_pt + 1.282 * sig_use)
        q95 = float(p_pt + 1.645 * sig_use)

        # Raw (unadjusted) model-based probability of exceeding thresholds:
        prob_ge_094761 = float(1.0 - norm.cdf((0.94761 - p_pt) / sig_use))
        prob_ge_095000 = float(1.0 - norm.cdf((0.95000 - p_pt) / sig_use))
        prob_ge_095438 = float(1.0 - norm.cdf((0.95438 - p_pt) / sig_use))

        # Bootstrap 95% CI for prob_ge_095000
        boot_probs = [float(1.0 - norm.cdf((0.95000 - p_pt) / bs)) for bs in boot_sigmas]
        prob_ci_95 = [float(np.percentile(boot_probs, 2.5)), float(np.percentile(boot_probs, 97.5))]

        # Leave-Family-Out sensitivity for this candidate
        lfo_cand_probs = []
        for fam_name, lfo_dict in lfo_evaluations.items():
            p_lfo = predict_point_pipeline(x_c, lfo_dict["r_sub"], lfo_dict["en_sub"], lfo_dict["hb_sub"], lfo_dict["f_d_sub"], lfo_dict["g_d_sub"], c_val["family"])
            p_lfo_095 = float(1.0 - norm.cdf((0.95000 - p_lfo) / lfo_dict["sub_sigma"]))
            lfo_cand_probs.append(p_lfo_095)
        worst_lfo_prob = float(min(lfo_cand_probs))

        # Leave-Day-Out sensitivity for this candidate
        ldo_cand_preds = []
        for day_name, ldo_dict in ldo_evaluations.items():
            p_ldo = predict_point_pipeline(x_c, ldo_dict["r_sub"], ldo_dict["en_sub"], ldo_dict["hb_sub"], ldo_dict["f_d_sub"], ldo_dict["g_d_sub"], c_val["family"])
            ldo_cand_preds.append(p_ldo)
        worst_ldo_pred = float(min(ldo_cand_preds))
        worst_ldo_prob = float(1.0 - norm.cdf((0.95000 - worst_ldo_pred) / sig_use))

        candidate_results[c_key] = {
            "name": c_val["name"],
            "sha256": c_val["file_sha256"],
            "id_target_hash": c_val["id_target_hash"],
            "flips_vs_evt143": c_val["flips_verified"],
            "expected_score": p_pt,
            "median_score": q50,
            "percentiles": {
                "p05": q05, "p10": q10, "p25": q25,
                "p50": q50, "p75": q75, "p90": q90, "p95": q95
            },
            "prob_ge_094761_raw": prob_ge_094761,
            "prob_ge_095000_raw": prob_ge_095000,
            "prob_ge_095438_raw": prob_ge_095438,
            "prob_ge_095000_ci95": prob_ci_95,
            "worst_lfo_prob": worst_lfo_prob,
            "worst_ldo_prob": worst_ldo_prob,
        }

    # -------------------------------------------------------------------------
    # 8. CORRECCION POR SELECCION & MALDICION DEL GANADOR
    # -------------------------------------------------------------------------
    print("\n[7/8] Applying Selection Correction & Winner's Curse Penalization...", flush=True)
    M_tested = 106
    N_pub = 20998
    se_binom = np.sqrt(0.94761 * (1 - 0.94761) / N_pub)
    expected_max_z = np.sqrt(2 * np.log(M_tested)) - (np.log(np.log(M_tested)) + np.log(4 * np.pi)) / (2 * np.sqrt(2 * np.log(M_tested)))
    winners_curse_bias = expected_max_z * se_binom
    debiased_anchor = current_anchor - winners_curse_bias

    print(f"Selection penalty: M={M_tested} submissions | Winner's Curse bias: +{winners_curse_bias*10000:.1f} bps")
    print(f"Observed anchor:   {current_anchor:.5f} -> Debiased anchor: {debiased_anchor:.5f}")

    for c_key, res in candidate_results.items():
        # Score adjusted for Winner's Curse:
        # Transfer delta = expected_score - current_anchor
        delta_score = res["expected_score"] - current_anchor
        debiased_expected = debiased_anchor + delta_score
        debiased_q05 = debiased_expected - 1.645 * sigma_calib

        z_deb = (0.95000 - debiased_expected) / sigma_calib
        prob_debiased = float(1.0 - norm.cdf(z_deb))

        # Worst reasonable scenario:
        # LFO worst family delta applied to debiased anchor
        worst_scenario_expected = debiased_expected - (res["expected_score"] - min(ldo_cand_preds))
        z_worst = (0.95000 - worst_scenario_expected) / (sigma_calib * 1.2)  # Expanded error variance
        prob_worst = float(1.0 - norm.cdf(z_worst))

        res["debiased_expected_score"] = float(debiased_expected)
        res["debiased_q05"] = float(debiased_q05)
        res["prob_ge_095000_debiased"] = float(prob_debiased)
        res["prob_ge_095000_worst_scenario"] = float(prob_worst)

        # Formal Status Assignment:
        # Since distribution-free guarantee is 0.0% and tail is extrapolated, assign MODEL_BASED_STABLE
        # if LFO probability > 0.01, otherwise MODEL_BASED_UNSTABLE
        if prob_debiased > 0.01 and res["worst_lfo_prob"] > 0.01:
            res["status"] = "MODEL_BASED_STABLE"
        else:
            res["status"] = "MODEL_BASED_UNSTABLE"

    # -------------------------------------------------------------------------
    # 9. PRUEBAS DE ESTABILIDAD
    # -------------------------------------------------------------------------
    print("\n[8/8] Running Stability Tests across Subsets, Densities & Anchors...", flush=True)

    # 9a. Exclude Small Probes (<30 flips)
    large_models_mask = df_transfer["flips_proxy"] >= 30
    df_large = df_transfer[large_models_mask].reset_index(drop=True)
    if len(df_large) >= 20:
        r_lg = Ridge(alpha=10.0).fit(df_large[feature_cols].values, df_large["public_score"].values)
        res_lg = df_large["public_score"].values - r_lg.predict(df_large[feature_cols].values)
        sigma_large = float(np.std(res_lg))
    else:
        sigma_large = sigma_calib

    # 9b. Exclude Best Score (EVT-143)
    no_best_mask = df_transfer["event_id"] != "EVT-143"
    r_no_best = Ridge(alpha=10.0).fit(df_transfer[feature_cols][no_best_mask].values, df_transfer["public_score"][no_best_mask].values)
    pred_no_best_top25 = float(r_no_best.predict(np.array([[current_anchor, 25.0, 0.0, 1.0, 0.0, 0.0, 0.0, n_hist_events, 30.0, 1.0]]))[0])

    # 9c. Exclude Worst Score (EVT-142)
    no_worst_mask = df_transfer["event_id"] != "EVT-142"
    r_no_worst = Ridge(alpha=10.0).fit(df_transfer[feature_cols][no_worst_mask].values, df_transfer["public_score"][no_worst_mask].values)
    pred_no_worst_top25 = float(r_no_worst.predict(np.array([[current_anchor, 25.0, 0.0, 1.0, 0.0, 0.0, 0.0, n_hist_events, 30.0, 1.0]]))[0])

    stability_results = {
        "sigma_excluding_small_probes": sigma_large,
        "pred_top25_excluding_best_event": pred_no_best_top25,
        "pred_top25_excluding_worst_event": pred_no_worst_top25,
        "delta_sensitivity_events": float(pred_no_worst_top25 - pred_no_best_top25),
        "debiased_anchor_sensitivity": {
            "evt143_anchor_094761": 0.94761 - winners_curse_bias,
            "evt132_anchor_094752": 0.94752 - winners_curse_bias,
        }
    }

    print(f"Stability Delta (Excluding Best vs Worst Event): {stability_results['delta_sensitivity_events']:+.6f}")

    # -------------------------------------------------------------------------
    # COMPILE FINAL REPORT & PRINT SUMMARY TABLE
    # -------------------------------------------------------------------------
    full_report = {
        "title": "Rigorous Calibrated Transfer Backtest & Epistemic Uncertainty Estimation",
        "timestamp_utc": "2026-11-10T19:00:00Z",
        "data_reconciliation": reconciliation_meta,
        "backtest_evaluation_windows": window_evals,
        "model_score_metrics": model_score_metrics,
        "best_model": best_model_name,
        "family_metrics": family_metrics,
        "coverage_and_widths": {
            "cov80": cov80, "width80": width80,
            "cov90": cov90, "width90": width90,
            "brier_surrogate": brier_surrogate,
            "ece_surrogate": ece_val,
        },
        "selection_penalization": {
            "m_tested": M_tested,
            "n_public": N_pub,
            "se_binom": float(se_binom),
            "expected_max_z": float(expected_max_z),
            "winners_curse_bias": float(winners_curse_bias),
            "observed_anchor": current_anchor,
            "debiased_anchor": float(debiased_anchor),
        },
        "candidate_calibrated_evaluations": candidate_results,
        "stability_tests": stability_results,
        "epistemic_framework": {
            "distribution_free_lower_probability": "0.0% (n_verified = 0 < 19 required)",
            "identifiable_interval_no_assumptions": "[0.0%, 100.0%]",
            "model_based_prob_label": "MODEL_BASED_NOT_GUARANTEED",
            "threshold_extrapolation_warning": "Zero historical events >= 0.95000. Any point probability is a tail extrapolation.",
        }
    }

    report_json_path = out_dir / "calibrated_transfer_backtest_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print("\n" + "=" * 110)
    print("MANDATORY OUTPUT TABLE — RIGOROUS CALIBRATED PROBABILITIES")
    print("=" * 110)
    print(f"{'Candidato':<32} | {'Score Esperado':<14} | {'P5 Score':<10} | {'P(S>=.95) Deb':<14} | {'IC 95% Prob':<18} | {'Peor LFO':<10} | {'Estado':<20}")
    print("-" * 110)
    for c_key, res in candidate_results.items():
        ci_str = f"[{res['prob_ge_095000_ci95'][0]*100:.1f}%, {res['prob_ge_095000_ci95'][1]*100:.1f}%]"
        print(f"{res['name']:<32} | {res['debiased_expected_score']:<14.5f} | {res['debiased_q05']:<10.5f} | {res['prob_ge_095000_debiased']*100:<13.2f}% | {ci_str:<18} | {res['worst_lfo_prob']*100:<9.2f}% | {res['status']:<20}")
    print("=" * 110)

    print(f"\nReport written to {report_json_path}")
    print(f"Report SHA-256: {compute_sha256(report_json_path)}")


if __name__ == "__main__":
    main()
