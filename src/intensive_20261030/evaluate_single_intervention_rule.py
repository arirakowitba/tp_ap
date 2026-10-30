"""Evaluation of Single Frozen Algorithmic Intervention Rule (CX-0006 & CX-0007).

Constructs, freezes, and evaluates UNA SINGLE algorithmic intervention rule:
- Target: Seen sellers only (`is_seen == 1`).
- Consensus condition: Both CatBoost Bag3 and LightGBM unanimously agree against baseline
  (`pred_cb != pred_base` AND `pred_lgb != pred_base` AND `pred_cb == pred_lgb`).
- Confidence margin: `|p_blend - 0.50| >= 0.06`.

Evaluated with:
1. Exact paired comparisons on all 3 confirm splits (Seeds 2026, 314159, 20261030).
2. Fixes, breaks, net gain, flip precision, and 95% cluster-bootstrap by seller (2,000 resamples).
3. Exact application to 30,000 test cases against submission_48_reference.csv.
4. Generates candidate CSV in submissions/20261030/ and verifies with submission_guard.
5. Explicit declaration of exploratory transfer status vs 48.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_20261030.paired_eval import seller_cluster_bootstrap_ci
from src.eval_20261030.submission_guard import check_submission_file, REGISTRY_JSON, S48_JSON
from src.intensive_20261030.shared_cache import compute_file_sha256

OUT_DIR = ROOT / "results" / "intensive_20261030"
CAND_DIR = ROOT / "submissions" / "20261030"
MARGIN_M = 0.06


def evaluate_rule_on_confirm():
    """Evaluates the single frozen intervention rule on the 3 confirm splits."""
    confirm_results = {}
    for seed in [2026, 314159, 20261030]:
        pair_p = OUT_DIR / f"paired_all_models_seed_{seed}_confirm.csv.gz"
        assert pair_p.exists(), f"Missing confirm pair file: {pair_p}"
        df = pd.read_csv(pair_p)
        y = df["y_true"].values
        is_seen = df["is_seen"].values
        sellers = df["seller_id"].astype(str).values
        pred_base = df["pred_A_hist_tuned"].values
        p_base = df["p_A_hist"].values
        p_cb = df["p_CB_Bag3_hist"].values
        p_lgb = df["p_LGB_hist"].values
        p_blend = df["p_Blend075_Bag3_hist"].values

        pred_cb = (p_cb >= 0.50).astype(int)
        pred_lgb = (p_lgb >= 0.50).astype(int)

        # Frozen intervention condition
        mask_intervene = (
            (is_seen == 1)
            & (pred_cb != pred_base)
            & (pred_lgb != pred_base)
            & (pred_cb == pred_lgb)
            & (np.abs(p_blend - 0.50) >= MARGIN_M)
        )

        pred_rule = pred_base.copy()
        pred_rule[mask_intervene] = pred_cb[mask_intervene]

        # Metric computation
        n_flips = int(mask_intervene.sum())
        fixes = int(((pred_rule == y) & (pred_base != y)).sum())
        breaks = int(((pred_rule != y) & (pred_base == y)).sum())
        net = fixes - breaks
        precision = round(fixes / max(fixes + breaks, 1), 4)

        # Baseline and rule accuracy
        acc_base = float((pred_base == y).mean())
        acc_rule = float((pred_rule == y).mean())

        # LogLoss baseline vs rule (using blended probabilities on flipped items)
        eps = 1e-15
        p_eval_rule = p_base.copy()
        p_eval_rule[mask_intervene] = p_blend[mask_intervene]
        p_clip_base = np.clip(p_base, eps, 1 - eps)
        p_clip_rule = np.clip(p_eval_rule, eps, 1 - eps)
        ll_base = float(-np.mean(y * np.log(p_clip_base) + (1 - y) * np.log(1 - p_clip_base)))
        ll_rule = float(-np.mean(y * np.log(p_clip_rule) + (1 - y) * np.log(1 - p_clip_rule)))

        ci_low, ci_high, _ = seller_cluster_bootstrap_ci(y, pred_base, pred_rule, sellers, n_boot=2000, seed=42)

        confirm_results[str(seed)] = {
            "n_confirm": int(len(df)),
            "n_seen": int((is_seen == 1).sum()),
            "n_flips": n_flips,
            "fixes": fixes,
            "breaks": breaks,
            "net": net,
            "flip_precision": precision,
            "acc_base": round(acc_base, 5),
            "acc_rule": round(acc_rule, 5),
            "delta_acc": round(acc_rule - acc_base, 5),
            "logloss_base": round(ll_base, 5),
            "logloss_rule": round(ll_rule, 5),
            "delta_logloss": round(ll_rule - ll_base, 5),
            "bootstrap_seller_ci95": [round(ci_low, 2), round(ci_high, 2)],
        }

    return confirm_results


def apply_rule_to_test():
    """Applies the identical frozen intervention rule to 30,000 test cases against submission 48."""
    test_path = OUT_DIR / "test_30k_mainline_predictions.csv.gz"
    assert test_path.exists(), f"Missing test mainline predictions file: {test_path}"
    df_test = pd.read_csv(test_path)

    ids = df_test["ID"].values
    s48_used = df_test["s48_used"].values
    s48_cond = df_test["s48_condition"].values
    is_seen = df_test["is_seen"].values
    p_cb = df_test["p_cb_bag3"].values
    p_lgb = df_test["p_lgb_bag3"].values
    p_blend = df_test["p_blend_bag3"].values

    pred_cb = (p_cb >= 0.50).astype(int)
    pred_lgb = (p_lgb >= 0.50).astype(int)

    # Identical intervention condition on test
    mask_intervene = (
        (is_seen == 1)
        & (pred_cb != s48_used)
        & (pred_lgb != s48_used)
        & (pred_cb == pred_lgb)
        & (np.abs(p_blend - 0.50) >= MARGIN_M)
    )

    flipped_ids = ids[mask_intervene].tolist()
    pred_test_used = s48_used.copy()
    pred_test_used[mask_intervene] = pred_cb[mask_intervene]

    # Directions
    to_used = int(((s48_used == 0) & mask_intervene).sum())
    to_new = int(((s48_used == 1) & mask_intervene).sum())

    # Build output CSV
    out_csv = CAND_DIR / "candidate_single_intervention_rule_on_48.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["ID", "condition"])
        for i, u in zip(ids, pred_test_used):
            w.writerow([int(i), "used" if u == 1 else "new"])

    sha256 = compute_file_sha256(out_csv)

    # Validate with submission guard
    with open(S48_JSON, "r", encoding="utf-8") as f:
        s48_dict = {int(k): v for k, v in json.load(f).items()}
    with open(REGISTRY_JSON, "r", encoding="utf-8") as f:
        reg_data = json.load(f)

    guard_ok, guard_msg = check_submission_file(out_csv, reg_data, s48_dict, verbose=False)

    return {
        "candidate_file": out_csv.name,
        "sha256": sha256,
        "total_flips_vs_48": len(flipped_ids),
        "to_used": to_used,
        "to_new": to_new,
        "seen_sellers_count": len(flipped_ids),
        "new_sellers_count": 0,
        "guard_passed": guard_ok,
        "guard_message": guard_msg,
        "flipped_ids": flipped_ids,
        "sample_flipped_ids": flipped_ids[:15],
    }


def main():
    print("=" * 80)
    print(f"EVALUATING SINGLE FROZEN INTERVENTION RULE (MARGIN M >= {MARGIN_M})")
    print("=" * 80)

    confirm_res = evaluate_rule_on_confirm()
    print("\n--- CONFIRM VALIDATION (AGAINST TRAINABLE PROXY A_hist) ---")
    total_net = 0
    total_flips = 0
    total_fixes = 0
    total_breaks = 0
    for seed, r in confirm_res.items():
        print(
            f"Seed {seed} (N={r['n_confirm']}, Seen={r['n_seen']}): "
            f"Flips={r['n_flips']}, Fixes/Breaks={r['fixes']}/{r['breaks']}, "
            f"Net={r['net']:+d}, Prec={r['flip_precision']:.1%}, "
            f"Delta LL={r['delta_logloss']:+.5f}, CI95={r['bootstrap_seller_ci95']}"
        )
        total_net += r["net"]
        total_flips += r["n_flips"]
        total_fixes += r["fixes"]
        total_breaks += r["breaks"]

    print(
        f"\nAggregate Confirm: Flips={total_flips}, Fixes/Breaks={total_fixes}/{total_breaks}, "
        f"Net={total_net:+d}, Prec={total_fixes / max(total_fixes + total_breaks, 1):.1%}"
    )

    test_res = apply_rule_to_test()
    print("\n--- TEST APPLICATION AGAINST SUBMISSION 48 REFERENCE ---")
    print(f"Candidate File: {test_res['candidate_file']}")
    print(f"SHA256: {test_res['sha256']}")
    print(f"Total Flips vs 48: {test_res['total_flips_vs_48']} (to_used={test_res['to_used']}, to_new={test_res['to_new']})")
    print(f"Guard Validation: {'PASS' if test_res['guard_passed'] else 'FAIL'}")

    report = {
        "rule_specification": {
            "name": "Single_Frozen_Consensus_Seen_Margin006",
            "condition": "is_seen == 1 AND pred_cb_bag3 != base AND pred_lgb_bag3 != base AND pred_cb_bag3 == pred_lgb_bag3 AND |p_blend - 0.50| >= 0.06",
            "margin_threshold": MARGIN_M,
            "train_baseline_proxy": "A_hist (Deep CatBoost Base with tune threshold)",
            "test_reference": "submission_48_reference.csv (.94633)",
            "epistemological_status": "EXPLORATORY_TRANSFER_PROBE_AGAINST_48",
            "epistemological_disclaimer": "Esta prueba explora transferencia contra submission_48 y NO está validada vs 48, debido a que 48 incluye microflips manuales históricos no mapeables a train y fue evaluada contra el proxy A_hist."
        },
        "confirm_evaluation": confirm_res,
        "confirm_aggregate": {
            "total_flips": total_flips,
            "total_fixes": total_fixes,
            "total_breaks": total_breaks,
            "total_net": total_net,
            "aggregate_precision": round(total_fixes / max(total_fixes + total_breaks, 1), 4),
        },
        "test_candidate": test_res
    }

    out_json = OUT_DIR / "single_intervention_rule_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report to: {out_json}")


if __name__ == "__main__":
    main()
