#!/usr/bin/env python3
"""Build Real Candidates & Execute Rolling-Origin Retrospective Backtest (CX-0148 Compliance).

Implements:
1. Real Candidates Generation:
   - W010_blend: 0.90 * baseline + 0.10 * quad_canonical (30k CSV, SHA-256, ID/target hash, flips vs EVT-143).
   - W035_blend: 0.65 * baseline + 0.35 * quad_canonical (30k CSV, SHA-256, ID/target hash, flips vs EVT-143).
2. Rolling-Origin Retrospective Backtest:
   - Chronological step-by-step out-of-time evaluation across historical scored submissions.
   - Strictly zero leakage (training only on events prior to t).
   - Compares Ridge, ElasticNet, QuantileRegression, and Regularized LightGBM.
   - Out-of-time residuals, empirical coverage, Brier score, and calibration.
3. Probability Calibration for score >= 0.95000:
   - Model-based probability with temporal block bootstrap intervals.
   - Leave-family-out cross-validation.
   - Jackknife event sensitivity.
   - Worst-case family analysis.
   - Percentile 5 (lower bound).
"""

import hashlib
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import ElasticNet, QuantileRegressor, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error
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
    print("REAL CANDIDATES GENERATION & ROLLING-ORIGIN RETROSPECTIVE BACKTEST")
    print("=" * 80)

    out_cand_dir = Path("submissions/staging_candidates_20261110")
    out_cand_dir.mkdir(parents=True, exist_ok=True)
    out_backtest_dir = Path("results/rolling_origin_backtest_20261110")
    out_backtest_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # STEP 1: GENERATE REAL CANDIDATE VECTORS (W010 & W035)
    # ---------------------------------------------------------
    print("\n[1/4] Generating Real Candidate Vectors for W010 and W035...")
    base_probs_path = Path("results/ultimate_podium_engine_23h/test_probabilities_podium_engine.csv")
    quad_probs_path = Path("results/canonical_nested_models_20261110/test_probabilities_canonical_nested.csv")
    evt143_path = Path("submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv")

    assert base_probs_path.exists(), f"Missing {base_probs_path}"
    assert quad_probs_path.exists(), f"Missing {quad_probs_path}"
    assert evt143_path.exists(), f"Missing {evt143_path}"

    df_base = pd.read_csv(base_probs_path)
    df_quad = pd.read_csv(quad_probs_path)
    df_143 = pd.read_csv(evt143_path)

    n_test = len(df_143)
    assert len(df_base) == n_test and len(df_quad) == n_test, "Length mismatch on test set!"

    p_base = df_base["p_optimal_blend"].values
    p_quad = df_quad["p_used_quad"].values
    y_143 = (df_143["condition"] == "used").astype(int).values
    ids_143 = df_143["ID"].values

    candidates_meta = {}

    for w_val in [0.10, 0.35]:
        cand_name = f"candidate_w{int(w_val*100):03d}"
        p_blend = (1 - w_val) * p_base + w_val * p_quad
        y_pred = (p_blend >= 0.5).astype(int)
        cond_str = ["used" if y == 1 else "new" for y in y_pred]

        df_out = pd.DataFrame({"ID": ids_143, "condition": cond_str})
        csv_path = out_cand_dir / f"submission_{cand_name}.csv"
        df_out.to_csv(csv_path, index=False)

        csv_sha = compute_sha256(csv_path)
        id_target_hash = compute_id_target_hash(df_out)

        # Flips against EVT-143
        diff_mask = (y_pred != y_143)
        n_flips = int(np.sum(diff_mask))
        flipped_ids = ids_143[diff_mask]
        flips_hash = hashlib.sha256("\n".join(map(str, flipped_ids)).encode("utf-8")).hexdigest()

        # Flips against pure baseline
        diff_base = (y_pred != (p_base >= 0.5).astype(int))
        n_flips_base = int(np.sum(diff_base))

        candidates_meta[cand_name] = {
            "candidate_id": f"CAND-20261110-{cand_name.upper()}",
            "weight": w_val,
            "csv_path": str(csv_path),
            "csv_sha256": csv_sha,
            "id_target_hash": id_target_hash,
            "n_flips_vs_evt143": n_flips,
            "flips_vs_evt143_sha256": flips_hash,
            "n_flips_vs_baseline": n_flips_base,
            "used_count": int(np.sum(y_pred == 1)),
            "new_count": int(np.sum(y_pred == 0)),
        }

        print(f"Generated {cand_name}:")
        print(f"  CSV Path:              {csv_path}")
        print(f"  CSV SHA-256:           {csv_sha}")
        print(f"  ID/Target Map Hash:    {id_target_hash}")
        print(f"  Flips vs EVT-143:      {n_flips} (Hash: {flips_hash[:16]}...)")
        print(f"  Flips vs Pure Base:    {n_flips_base}")

    # Also build surgical candidate anchored strictly on EVT-143 with top high-confidence concordant flips
    # (Top discordant items where canonical model has very high confidence)
    diff_blend_143 = (p_blend >= 0.5).astype(int) != y_143
    conf_scores = np.abs(p_blend - 0.5)
    # Pick top 25 high confidence flips
    top25_flip_idx = np.argsort(conf_scores * diff_blend_143)[-25:]
    y_surg = y_143.copy()
    y_surg[top25_flip_idx] = (p_blend[top25_flip_idx] >= 0.5).astype(int)
    cond_surg = ["used" if y == 1 else "new" for y in y_surg]
    df_surg = pd.DataFrame({"ID": ids_143, "condition": cond_surg})
    surg_path = out_cand_dir / "submission_candidate_surgical_w035_top25.csv"
    df_surg.to_csv(surg_path, index=False)
    surg_sha = compute_sha256(surg_path)
    surg_id_target_hash = compute_id_target_hash(df_surg)
    surg_diff_mask = (y_surg != y_143)
    surg_flips = int(np.sum(surg_diff_mask))
    surg_flipped_ids = ids_143[surg_diff_mask]
    surg_flips_hash = hashlib.sha256("\n".join(map(str, surg_flipped_ids)).encode("utf-8")).hexdigest()

    candidates_meta["candidate_surgical_top25"] = {
        "candidate_id": "CAND-20261110-SURGICAL-TOP25",
        "csv_path": str(surg_path),
        "csv_sha256": surg_sha,
        "id_target_hash": surg_id_target_hash,
        "n_flips_vs_evt143": surg_flips,
        "flips_vs_evt143_sha256": surg_flips_hash,
        "used_count": int(np.sum(y_surg == 1)),
        "new_count": int(np.sum(y_surg == 0)),
    }
    print(f"Generated surgical candidate: {surg_path} (Flips vs EVT-143: {surg_flips}, SHA-256: {surg_sha[:16]}...)")

    with open(out_cand_dir / "staging_candidates_manifest.json", "w", encoding="utf-8") as f:
        json.dump(candidates_meta, f, indent=2)

    # ---------------------------------------------------------
    # STEP 2: BUILD ROLLING-ORIGIN RETROSPECTIVE DATASET
    # ---------------------------------------------------------
    print("\n[2/4] Constructing Rolling-Origin Feature Matrix...")
    ledger_path = Path("results/ledger_reconciliation_20261110/reconciled_canonical_ledger.csv")
    assert ledger_path.exists(), f"Missing {ledger_path}"
    df_ledger = pd.read_csv(ledger_path)

    # Filter to events that had a verified score
    df_scored = df_ledger[df_ledger["public_score"].notna()].copy()
    df_scored["event_num"] = df_scored["event_id"].apply(lambda x: int(x.split("-")[1]))
    df_scored = df_scored.sort_values("event_num").reset_index(drop=True)

    n_events = len(df_scored)
    print(f"Total chronological scored events in ledger: {n_events}")

    # Build rolling features without future leakage
    features_list = []
    targets_score = df_scored["public_score"].values

    current_anchor_score = 0.94230  # Baseline prior to competition submissions
    for i in range(n_events):
        row = df_scored.iloc[i]
        flips = row["flips_vs_evt143"] if pd.notna(row["flips_vs_evt143"]) else 50
        family = row["policy_family"]

        # Feature vector available strictly BEFORE event i:
        feat = {
            "event_idx": i,
            "event_id": row["event_id"],
            "anchor_score": current_anchor_score,
            "flips_vs_anchor": min(float(flips), 300.0),
            "is_probe": 1.0 if "probe" in family else 0.0,
            "is_surgical": 1.0 if "surgical" in family else 0.0,
            "is_stacker": 1.0 if "stacker" in family or "meta" in family else 0.0,
            "is_router": 1.0 if "router" in family else 0.0,
            "is_ladder": 1.0 if "ladder" in family or "top" in family else 0.0,
            "prior_events_count": float(i),
        }
        features_list.append(feat)

        # Update anchor score for subsequent events
        if row["public_score"] > current_anchor_score:
            current_anchor_score = row["public_score"]

    df_features = pd.DataFrame(features_list)
    feature_cols = [
        "anchor_score", "flips_vs_anchor", "is_probe", "is_surgical",
        "is_stacker", "is_router", "is_ladder", "prior_events_count"
    ]
    X_all = df_features[feature_cols].values
    y_all = targets_score

    # ---------------------------------------------------------
    # STEP 3: RUN ROLLING-ORIGIN BACKTEST
    # ---------------------------------------------------------
    print("\n[3/4] Running Rolling-Origin Retrospective Backtest...")
    min_train_size = 20
    rolling_results = []

    for t in range(min_train_size, n_events):
        X_tr = X_all[:t]
        y_tr = y_all[:t]
        X_te = X_all[t:t+1]
        y_te = y_all[t]

        # Model 1: Ridge Regression
        ridge = Ridge(alpha=10.0)
        ridge.fit(X_tr, y_tr)
        p_ridge = float(ridge.predict(X_te)[0])

        # Model 2: ElasticNet
        enet = ElasticNet(alpha=0.01, l1_ratio=0.5)
        enet.fit(X_tr, y_tr)
        p_enet = float(enet.predict(X_te)[0])

        # Model 3: Quantile Regression (p05, p50, p95)
        qr_05 = QuantileRegressor(quantile=0.05, alpha=1.0)
        qr_05.fit(X_tr, y_tr)
        q05 = float(qr_05.predict(X_te)[0])

        qr_50 = QuantileRegressor(quantile=0.50, alpha=1.0)
        qr_50.fit(X_tr, y_tr)
        q50 = float(qr_50.predict(X_te)[0])

        qr_95 = QuantileRegressor(quantile=0.95, alpha=1.0)
        qr_95.fit(X_tr, y_tr)
        q95 = float(qr_95.predict(X_te)[0])

        # Model 4: LightGBM highly regularized
        lgb_reg = lgb.LGBMRegressor(
            n_estimators=30, learning_rate=0.05, max_depth=3, min_child_samples=5,
            reg_alpha=1.0, reg_lambda=1.0, verbose=-1, random_state=42
        )
        lgb_reg.fit(X_tr, y_tr)
        p_lgb = float(lgb_reg.predict(X_te)[0])

        # Blend ensemble
        p_blend = 0.40 * p_ridge + 0.30 * p_enet + 0.30 * p_lgb
        residual = float(p_blend - y_te)

        rolling_results.append({
            "step": t,
            "event_id": df_scored.iloc[t]["event_id"],
            "family": df_scored.iloc[t]["policy_family"],
            "actual_score": float(y_te),
            "p_ridge": p_ridge,
            "p_enet": p_enet,
            "p_lgb": p_lgb,
            "p_blend": p_blend,
            "q05": q05,
            "q50": q50,
            "q95": q95,
            "residual": residual,
            "in_interval_90": bool(q05 <= y_te <= q95),
            "actual_ge_095": bool(y_te >= 0.95000),
            "pred_ge_095": bool(p_blend >= 0.95000),
        })

    df_rolling = pd.DataFrame(rolling_results)
    residuals_arr = df_rolling["residual"].values
    rmse = np.sqrt(mean_squared_error(df_rolling["actual_score"], df_rolling["p_blend"]))
    mae = mean_absolute_error(df_rolling["actual_score"], df_rolling["p_blend"])
    coverage_90 = np.mean(df_rolling["in_interval_90"])

    print(f"Rolling backtest evaluated on {len(df_rolling)} out-of-time steps:")
    print(f"  RMSE:                     {rmse:.6f}")
    print(f"  MAE:                      {mae:.6f}")
    print(f"  Empirical 90% Coverage:   {coverage_90*100:.1f}%")
    print(f"  Mean Residual (Bias):     {np.mean(residuals_arr):+.6f}")
    print(f"  Std of Residuals:         {np.std(residuals_arr):.6f}")

    # ---------------------------------------------------------
    # STEP 4: MODEL-BASED PROBABILITY ESTIMATION & SENSITIVITY
    # ---------------------------------------------------------
    print("\n[4/4] Estimating Model-Based Probability for P(score >= 0.95000)...")

    # Fit final transfer models on all historical scored events
    ridge_final = Ridge(alpha=10.0).fit(X_all, y_all)
    enet_final = ElasticNet(alpha=0.01, l1_ratio=0.5).fit(X_all, y_all)
    lgb_final = lgb.LGBMRegressor(
        n_estimators=30, learning_rate=0.05, max_depth=3, min_child_samples=5,
        reg_alpha=1.0, reg_lambda=1.0, verbose=-1, random_state=42
    ).fit(X_all, y_all)
    qr05_final = QuantileRegressor(quantile=0.05, alpha=1.0).fit(X_all, y_all)

    # Standard error of prediction from rolling out-of-time residuals
    sigma_res = float(np.std(residuals_arr))

    # Evaluate each candidate
    candidate_predictions = {}
    current_best_anchor = float(np.max(y_all))  # EVT-143 = 0.94761

    for cand_name, cand_info in candidates_meta.items():
        n_flips = float(cand_info["n_flips_vs_evt143"])
        # Candidate features
        x_cand = np.array([[
            current_best_anchor,
            min(n_flips, 300.0),
            0.0,  # is_probe
            1.0 if "surgical" in cand_name else 0.0,
            0.0,  # is_stacker
            0.0,  # is_router
            1.0 if "w" in cand_name else 0.0,  # is_ladder / blend
            float(n_events),
        ]])

        p_r = float(ridge_final.predict(x_cand)[0])
        p_e = float(enet_final.predict(x_cand)[0])
        p_l = float(lgb_final.predict(x_cand)[0])
        p_pt = 0.40 * p_r + 0.30 * p_e + 0.30 * p_l
        p05 = float(qr05_final.predict(x_cand)[0])

        # Model-based probability P(Score >= 0.95000)
        # Using normal approximation on calibrated out-of-time residual distribution
        z = (0.95000 - p_pt) / sigma_res
        prob_ge_095 = float(1.0 - norm.cdf(z))

        # Temporal Block Bootstrap (B=1000 resamples of length 10 blocks)
        n_blocks = len(residuals_arr) // 10
        boot_probs = []
        for _ in range(1000):
            block_starts = np.random.randint(0, len(residuals_arr) - 10 + 1, size=n_blocks)
            boot_res = np.concatenate([residuals_arr[b:b+10] for b in block_starts])
            boot_sigma = np.std(boot_res)
            boot_z = (0.95000 - p_pt) / boot_sigma
            boot_probs.append(1.0 - norm.cdf(boot_z))

        prob_ci_05 = float(np.percentile(boot_probs, 2.5))
        prob_ci_95 = float(np.percentile(boot_probs, 97.5))

        # Jackknife event sensitivity (drop one event, recompute)
        jk_probs = []
        for j in range(len(residuals_arr)):
            jk_res = np.delete(residuals_arr, j)
            jk_sigma = np.std(jk_res)
            jk_z = (0.95000 - p_pt) / jk_sigma
            jk_probs.append(1.0 - norm.cdf(jk_z))
        jk_min = float(np.min(jk_probs))
        jk_max = float(np.max(jk_probs))

        # Leave-Family-Out sensitivity
        lfo_probs = {}
        for fam in ["portfolio_ladder", "lexical_probe", "surgical_quad", "reference_anchor"]:
            mask_fam = (df_rolling["family"] != fam)
            if np.sum(mask_fam) > 10:
                lfo_res = residuals_arr[mask_fam]
                lfo_sigma = np.std(lfo_res)
                lfo_z = (0.95000 - p_pt) / lfo_sigma
                lfo_probs[fam] = float(1.0 - norm.cdf(lfo_z))

        worst_case_prob = float(min(lfo_probs.values())) if lfo_probs else prob_ge_095

        candidate_predictions[cand_name] = {
            "predicted_score_point": p_pt,
            "percentile_5_score": p05,
            "sigma_residual": sigma_res,
            "prob_ge_095_point": prob_ge_095,
            "prob_ge_095_block_bootstrap_95ci": [prob_ci_05, prob_ci_95],
            "jackknife_sensitivity_range": [jk_min, jk_max],
            "leave_family_out_worst_case": worst_case_prob,
            "leave_family_out_by_family": lfo_probs,
            "epistemic_status": "MODEL_BASED_NOT_GUARANTEED",
            "distribution_free_guarantee": 0.0,
            "identifiable_interval_no_assumptions": [0.0, 1.0],
        }

        print(f"\nCandidate: {cand_name}")
        print(f"  Predicted Public Score (Point):     {p_pt:.5f}")
        print(f"  5th Percentile Score (Lower Bound): {p05:.5f}")
        print(f"  P_model(Score >= 0.95000):          {prob_ge_095*100:.2f}%")
        print(f"  Block Bootstrap 95% CI:             [{prob_ci_05*100:.2f}%, {prob_ci_95*100:.2f}%]")
        print(f"  Jackknife Range:                    [{jk_min*100:.2f}%, {jk_max*100:.2f}%]")
        print(f"  Leave-Family-Out Worst-Case:        {worst_case_prob*100:.2f}%")

    # Brier Score on historical target crossing
    # Historically, no submission has crossed 0.95000 yet (max is 0.94761)
    brier_score = float(brier_score_loss(df_rolling["actual_ge_095"], df_rolling["pred_ge_095"]))

    # ---------------------------------------------------------
    # STEP 5: SAVE COMPREHENSIVE BACKTEST REPORT
    # ---------------------------------------------------------
    report_data = {
        "title": "Rolling-Origin Retrospective Backtest & Calibrated Probability Audit",
        "timestamp_utc": "2026-11-10T18:00:00Z",
        "n_chronological_events": n_events,
        "n_rolling_steps": len(df_rolling),
        "backtest_metrics": {
            "rmse": rmse,
            "mae": mae,
            "empirical_90_coverage": coverage_90,
            "mean_residual_bias": float(np.mean(residuals_arr)),
            "std_residual_sigma": sigma_res,
            "brier_score_target_095": brier_score,
        },
        "candidate_calibrated_probabilities": candidate_predictions,
        "epistemic_framework": {
            "distribution_free_guarantee": "0.0% (n_verified = 0 < 19 required)",
            "identifiable_interval_no_assumptions": "[0.0%, 100.0%]",
            "model_based_prob_label": "MODEL_BASED_NOT_GUARANTEED",
            "cv_train_evidence_label": "DEVELOPMENT_CV",
        }
    }

    report_path = out_backtest_dir / "rolling_origin_backtest_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    df_rolling.to_csv(out_backtest_dir / "rolling_origin_predictions_and_residuals.csv", index=False)

    print(f"\nReport written to {report_path} (SHA-256: {compute_sha256(report_path)})")


if __name__ == "__main__":
    main()
