"""Evaluation of Single Frozen Algorithmic Intervention Rule B4 (CX-0006..CX-0012).

Fixes CX-0011 & CX-0008 findings:
1. Exact model parity between `confirm` and `test`:
   - Both `confirm` and `test` use `CB_Bag3`, `LGB_Bag3`, and `Blend075_Bag3 = 0.75 * CB_Bag3 + 0.25 * LGB_Bag3`
     with explicit column and linear-combination assertions.
   - Side-by-side comparison against single-seed `LGB_single` (`p_LGB_hist` / `p_lgb_single`) is reported to
     document the exact impact of fixing the `CX-0011` discrepancy.
2. No summation across overlapping confirm splits (`2026`, `314159`, `20261030` overlap by ~18.1%-18.6% in confirm).
3. Quantitative decomposition of intervention rates per 1,000 `seen_seller` rows:
   - Within-family on `confirm` vs `A_hist_tuned` and `A_hist@0.50` (`0.52` to `3.84` per 1k seen rows), plus
     3 external cuts (`disjoint_3seed_20261101`, `nested_outer_777001`, `nested_outer_777002`).
   - Within-family on `test` (`21,487` seen rows) vs `A_hist_inductive_70k@0.50` (`26` flips = `1.21` per 1k seen rows),
     proving that 70k training scale does NOT inflate the intervention rate.
   - Cross-family transfer on `test` (`21,487` seen rows) vs `submission_48` (`453` flips = `21.08` per 1k seen rows),
     and on `confirm` vs historical `R156_Ensemble` (`22.29` and `21.37` per 1k seen rows), proving the jump to 453 flips
     comes from changing the reference baseline from `A_hist` to a different historical model family (`S48`).
4. Atomic candidate generation: writes to `.csv.tmp`, validates with `submission_guard`, unlinks and raises/exits
   non-zero on failure, and preserves the verified artifact (`51ea764c8b69...`) if bytes match.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_20261030.paired_eval import seller_cluster_bootstrap_ci
from src.eval_20261030.submission_guard import REGISTRY_JSON, S48_JSON, check_submission_file
from src.intensive_20261030.shared_cache import compute_file_sha256

OUT_DIR = ROOT / "results" / "intensive_20261030"
CAND_DIR = ROOT / "submissions" / "20261030"
MARGIN_M = 0.06
DECISION_TH = 0.50

RULE_CONFIG = {
    "rule_id": "B4_Single_Frozen_Consensus_Seen_Margin006",
    "target_segment": "is_seen == 1",
    "models": ["CB_Bag3", "LGB_Bag3", "Blend075_Bag3"],
    "blend_weights": {"CB_Bag3": 0.75, "LGB_Bag3": 0.25},
    "polarity": "used=1, new=0",
    "decision_threshold": DECISION_TH,
    "margin_threshold": MARGIN_M,
}
RULE_CONFIG_SHA256 = hashlib.sha256(
    json.dumps(RULE_CONFIG, sort_keys=True).encode("utf-8")
).hexdigest()


def _margin_band_counts(p_blend: np.ndarray, mask: np.ndarray) -> Dict[str, int]:
    margin = np.abs(p_blend - DECISION_TH)
    return {
        "0.06_to_0.10": int((mask & (margin >= 0.06) & (margin < 0.10)).sum()),
        "0.10_to_0.20": int((mask & (margin >= 0.10) & (margin < 0.20)).sum()),
        "0.20_plus": int((mask & (margin >= 0.20)).sum()),
    }


def _compute_intervention_mask(
    is_seen: np.ndarray,
    pred_base: np.ndarray,
    p_cb: np.ndarray,
    p_lgb: np.ndarray,
    p_blend: np.ndarray,
    margin_m: float = MARGIN_M,
) -> np.ndarray:
    pred_cb = (p_cb >= DECISION_TH).astype(int)
    pred_lgb = (p_lgb >= DECISION_TH).astype(int)
    pred_blend = (p_blend >= DECISION_TH).astype(int)
    return (
        (is_seen == 1)
        & (pred_cb != pred_base)
        & (pred_lgb != pred_base)
        & (pred_blend != pred_base)
        & (pred_cb == pred_lgb)
        & (pred_cb == pred_blend)
        & (np.abs(p_blend - DECISION_TH) >= margin_m)
    )


def _eval_confirm_slice(
    y: np.ndarray,
    is_seen: np.ndarray,
    sellers: np.ndarray,
    pred_base: np.ndarray,
    p_base: np.ndarray,
    p_cb: np.ndarray,
    p_lgb: np.ndarray,
    p_blend: np.ndarray,
) -> Dict[str, Any]:
    mask_intervene = _compute_intervention_mask(is_seen, pred_base, p_cb, p_lgb, p_blend)
    pred_cb = (p_cb >= DECISION_TH).astype(int)
    pred_rule = pred_base.copy()
    pred_rule[mask_intervene] = pred_cb[mask_intervene]

    n_seen = int((is_seen == 1).sum())
    n_flips = int(mask_intervene.sum())
    to_used = int(((pred_base == 0) & mask_intervene).sum())
    to_new = int(((pred_base == 1) & mask_intervene).sum())
    fixes = int(((pred_rule == y) & (pred_base != y)).sum())
    breaks = int(((pred_rule != y) & (pred_base == y)).sum())
    net = fixes - breaks
    precision = round(fixes / max(fixes + breaks, 1), 4)

    acc_base = float((pred_base == y).mean())
    acc_rule = float((pred_rule == y).mean())

    eps = 1e-15
    p_eval_rule = p_base.copy()
    p_eval_rule[mask_intervene] = p_blend[mask_intervene]
    p_clip_base = np.clip(p_base, eps, 1 - eps)
    p_clip_rule = np.clip(p_eval_rule, eps, 1 - eps)
    ll_base = float(-np.mean(y * np.log(p_clip_base) + (1 - y) * np.log(1 - p_clip_base)))
    ll_rule = float(-np.mean(y * np.log(p_clip_rule) + (1 - y) * np.log(1 - p_clip_rule)))

    ci_low, ci_high, _ = seller_cluster_bootstrap_ci(
        y, pred_base, pred_rule, sellers, n_boot=2000, seed=42
    )

    return {
        "n_confirm": int(len(y)),
        "n_seen": n_seen,
        "n_flips": n_flips,
        "rate_per_1k_seen": round(1000.0 * n_flips / max(n_seen, 1), 2),
        "to_used": to_used,
        "to_new": to_new,
        "margin_bands": _margin_band_counts(p_blend, mask_intervene),
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


def evaluate_rule_on_confirm() -> Dict[str, Any]:
    """Evaluates the single frozen intervention rule on confirm splits with exact Bag3 parity."""
    per_seed_bag3_vs_tuned = {}
    per_seed_bag3_vs_050 = {}
    per_seed_single_lgb_vs_tuned = {}
    per_seed_bag3_vs_r156 = {}

    for seed in [2026, 314159, 20261030]:
        pair_p = OUT_DIR / f"paired_all_models_seed_{seed}_confirm.csv.gz"
        assert pair_p.exists(), f"Missing confirm pair file: {pair_p}"
        df = pd.read_csv(pair_p)

        required_cols = {
            "y_true",
            "is_seen",
            "seller_id",
            "p_A_hist",
            "pred_A_hist_tuned",
            "p_CB_Bag3_hist",
            "p_LGB_Bag3_hist",
            "p_LGB_hist",
            "p_Blend075_Bag3_hist",
        }
        missing = required_cols - set(df.columns)
        assert not missing, f"Missing required columns in {pair_p}: {missing}"

        y = df["y_true"].values
        is_seen = df["is_seen"].values
        sellers = df["seller_id"].astype(str).values
        pred_base_tuned = df["pred_A_hist_tuned"].values
        p_base = df["p_A_hist"].values
        pred_base_050 = (p_base >= DECISION_TH).astype(int)

        p_cb_bag3 = df["p_CB_Bag3_hist"].values
        p_lgb_bag3 = df["p_LGB_Bag3_hist"].values
        p_lgb_single = df["p_LGB_hist"].values
        p_blend_bag3 = df["p_Blend075_Bag3_hist"].values

        blend_err = float(np.max(np.abs(0.75 * p_cb_bag3 + 0.25 * p_lgb_bag3 - p_blend_bag3)))
        assert blend_err < 1e-5, f"Blend parity violation on seed {seed}: max_err={blend_err}"

        per_seed_bag3_vs_tuned[str(seed)] = _eval_confirm_slice(
            y, is_seen, sellers, pred_base_tuned, p_base, p_cb_bag3, p_lgb_bag3, p_blend_bag3
        )
        per_seed_bag3_vs_050[str(seed)] = _eval_confirm_slice(
            y, is_seen, sellers, pred_base_050, p_base, p_cb_bag3, p_lgb_bag3, p_blend_bag3
        )
        per_seed_single_lgb_vs_tuned[str(seed)] = _eval_confirm_slice(
            y, is_seen, sellers, pred_base_tuned, p_base, p_cb_bag3, p_lgb_single, p_blend_bag3
        )

        r156_p = ROOT / "results" / "revision_submissions" / f"paired_ensemble_seed_{seed}.csv"
        if r156_p.exists():
            df156 = pd.read_csv(r156_p)
            p_156 = df156["p_ensemble"].values
            pred_156 = (
                df156["pred_ensemble"].values
                if "pred_ensemble" in df156.columns
                else (p_156 >= DECISION_TH).astype(int)
            )
            per_seed_bag3_vs_r156[str(seed)] = _eval_confirm_slice(
                y, is_seen, sellers, pred_156, p_156, p_cb_bag3, p_lgb_bag3, p_blend_bag3
            )

    external_cuts = {}
    for cut_name in ["disjoint_3seed_20261101", "nested_outer_777001", "nested_outer_777002"]:
        ext_p = OUT_DIR / f"paired_external_{cut_name}.csv.gz"
        if ext_p.exists():
            dfe = pd.read_csv(ext_p)
            external_cuts[cut_name] = {
                "vs_cb_single_tuned": _eval_confirm_slice(
                    dfe["y_true"].values,
                    dfe["is_seen"].values,
                    dfe["seller_id"].astype(str).values,
                    dfe["pred_cb_single_tuned"].values,
                    dfe["p_cb_single"].values,
                    dfe["p_cb_bag3"].values,
                    dfe["p_lgb_bag3"].values,
                    dfe["p_blend_bag3"].values,
                ),
                "vs_cb_single_050": _eval_confirm_slice(
                    dfe["y_true"].values,
                    dfe["is_seen"].values,
                    dfe["seller_id"].astype(str).values,
                    dfe["pred_cb_single_050"].values,
                    dfe["p_cb_single"].values,
                    dfe["p_cb_bag3"].values,
                    dfe["p_lgb_bag3"].values,
                    dfe["p_blend_bag3"].values,
                ),
            }

    return {
        "canonical_bag3_vs_A_hist_tuned": per_seed_bag3_vs_tuned,
        "canonical_bag3_vs_A_hist_050": per_seed_bag3_vs_050,
        "legacy_single_lgb_vs_A_hist_tuned_cx0011_comparison": per_seed_single_lgb_vs_tuned,
        "cross_family_bag3_vs_R156_Ensemble": per_seed_bag3_vs_r156,
        "external_cuts_bag3": external_cuts,
    }


def write_and_validate_candidate_csv(
    ids: np.ndarray,
    pred_test_used: np.ndarray,
    out_csv: Path,
    reg_data: Dict[str, Any],
    s48_dict: Dict[int, str],
) -> Dict[str, Any]:
    """Atomically writes candidate CSV via a temporary file, runs submission_guard, and fails fast if invalid."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = out_csv.parent / f".{out_csv.name}.{os.getpid()}.tmp"
    try:
        with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["ID", "condition"])
            for i, u in zip(ids, pred_test_used):
                w.writerow([int(i), "used" if int(u) == 1 else "new"])

        guard_ok, guard_msg = check_submission_file(tmp_csv, reg_data, s48_dict, verbose=False)
        if not guard_ok:
            tmp_csv.unlink(missing_ok=True)
            raise RuntimeError(f"Submission guard rejected candidate {out_csv.name}: {guard_msg}")

        tmp_sha = compute_file_sha256(tmp_csv)
        if out_csv.exists() and compute_file_sha256(out_csv) == tmp_sha:
            # Preserve existing verified file untouched (CX-0012)
            tmp_csv.unlink(missing_ok=True)
        else:
            os.replace(tmp_csv, out_csv)

        return {
            "sha256": compute_file_sha256(out_csv),
            "guard_passed": True,
            "guard_message": guard_msg,
        }
    except Exception:
        tmp_csv.unlink(missing_ok=True)
        raise


def apply_rule_to_test(out_csv: Optional[Path] = None) -> Dict[str, Any]:
    """Applies the identical frozen Bag3 intervention rule to 30,000 test cases against both A_hist_70k and S48."""
    test_path = OUT_DIR / "test_30k_mainline_predictions.csv.gz"
    assert test_path.exists(), f"Missing test mainline predictions file: {test_path}"
    df_test = pd.read_csv(test_path)

    required_cols = {
        "ID",
        "is_seen",
        "s48_used",
        "p_cb_single",
        "p_lgb_single",
        "p_cb_bag3",
        "p_lgb_bag3",
        "p_blend_bag3",
    }
    missing = required_cols - set(df_test.columns)
    assert not missing, f"Missing required columns in {test_path}: {missing}"

    ids = df_test["ID"].values
    assert int(ids[0]) == 1 and int(ids[-1]) == 30000 and len(ids) == 30000, "Test IDs must be 1..30000"

    s48_used = df_test["s48_used"].values
    a_hist_70k_used = (df_test["p_cb_single"].values >= DECISION_TH).astype(int)
    is_seen = df_test["is_seen"].values
    n_seen_test = int((is_seen == 1).sum())

    p_cb_bag3 = df_test["p_cb_bag3"].values
    p_lgb_bag3 = df_test["p_lgb_bag3"].values
    p_lgb_single = df_test["p_lgb_single"].values
    p_blend_bag3 = df_test["p_blend_bag3"].values

    blend_err = float(np.max(np.abs(0.75 * p_cb_bag3 + 0.25 * p_lgb_bag3 - p_blend_bag3)))
    assert blend_err < 1e-5, f"Test blend parity violation: max_err={blend_err}"

    # 1. Within-family on test: vs A_hist_inductive_70k@0.50
    mask_vs_ahist_bag3 = _compute_intervention_mask(
        is_seen, a_hist_70k_used, p_cb_bag3, p_lgb_bag3, p_blend_bag3
    )
    mask_vs_ahist_single_lgb = _compute_intervention_mask(
        is_seen, a_hist_70k_used, p_cb_bag3, p_lgb_single, p_blend_bag3
    )

    # 2. Cross-family transfer on test: vs submission_48
    mask_vs_s48_bag3 = _compute_intervention_mask(
        is_seen, s48_used, p_cb_bag3, p_lgb_bag3, p_blend_bag3
    )
    mask_vs_s48_single_lgb = _compute_intervention_mask(
        is_seen, s48_used, p_cb_bag3, p_lgb_single, p_blend_bag3
    )

    pred_cb = (p_cb_bag3 >= DECISION_TH).astype(int)
    flipped_ids = ids[mask_vs_s48_bag3].astype(int).tolist()
    pred_test_used = s48_used.copy()
    pred_test_used[mask_vs_s48_bag3] = pred_cb[mask_vs_s48_bag3]

    to_used = int(((s48_used == 0) & mask_vs_s48_bag3).sum())
    to_new = int(((s48_used == 1) & mask_vs_s48_bag3).sum())

    target_csv = out_csv if out_csv is not None else (CAND_DIR / "candidate_single_intervention_rule_on_48.csv")
    with open(S48_JSON, "r", encoding="utf-8") as f:
        s48_dict = {int(k): v for k, v in json.load(f).items()}
    with open(REGISTRY_JSON, "r", encoding="utf-8") as f:
        reg_data = json.load(f)

    guard_info = write_and_validate_candidate_csv(ids, pred_test_used, target_csv, reg_data, s48_dict)

    disagree_all = int((a_hist_70k_used != s48_used).sum())
    disagree_seen = int(((a_hist_70k_used != s48_used) & (is_seen == 1)).sum())

    return {
        "candidate_file": target_csv.name,
        "id_range_verified": "1..30000",
        "sha256": guard_info["sha256"],
        "rule_config_sha256": RULE_CONFIG_SHA256,
        "n_test_total": 30000,
        "n_test_seen": n_seen_test,
        "n_test_new": int((is_seen == 0).sum()),
        "baseline_disagreement_A_hist_70k_vs_S48": {
            "total_disagreements_30k": disagree_all,
            "seen_seller_disagreements": disagree_seen,
            "seen_disagreement_rate_per_1k": round(1000.0 * disagree_seen / n_seen_test, 2),
        },
        "within_family_vs_A_hist_inductive_70k_050": {
            "bag3_flips": int(mask_vs_ahist_bag3.sum()),
            "bag3_rate_per_1k_seen": round(1000.0 * int(mask_vs_ahist_bag3.sum()) / n_seen_test, 2),
            "bag3_to_used": int(((a_hist_70k_used == 0) & mask_vs_ahist_bag3).sum()),
            "bag3_to_new": int(((a_hist_70k_used == 1) & mask_vs_ahist_bag3).sum()),
            "bag3_margin_bands": _margin_band_counts(p_blend_bag3, mask_vs_ahist_bag3),
            "single_lgb_flips": int(mask_vs_ahist_single_lgb.sum()),
            "single_lgb_rate_per_1k_seen": round(1000.0 * int(mask_vs_ahist_single_lgb.sum()) / n_seen_test, 2),
        },
        "cross_family_transfer_vs_submission_48": {
            "bag3_flips": len(flipped_ids),
            "bag3_rate_per_1k_seen": round(1000.0 * len(flipped_ids) / n_seen_test, 2),
            "bag3_to_used": to_used,
            "bag3_to_new": to_new,
            "bag3_margin_bands": _margin_band_counts(p_blend_bag3, mask_vs_s48_bag3),
            "single_lgb_flips": int(mask_vs_s48_single_lgb.sum()),
            "single_lgb_rate_per_1k_seen": round(1000.0 * int(mask_vs_s48_single_lgb.sum()) / n_seen_test, 2),
        },
        "total_flips_vs_48": len(flipped_ids),
        "to_used": to_used,
        "to_new": to_new,
        "seen_sellers_count": len(flipped_ids),
        "new_sellers_count": 0,
        "guard_passed": guard_info["guard_passed"],
        "guard_message": guard_info["guard_message"],
        "flipped_ids": flipped_ids,
        "sample_flipped_ids": flipped_ids[:15],
    }


def main() -> int:
    print("=" * 80)
    print(f"EVALUATING SINGLE FROZEN INTERVENTION RULE B4 (MARGIN M >= {MARGIN_M})")
    print("=" * 80)

    confirm_res = evaluate_rule_on_confirm()
    print("\n--- 1. CONFIRM VALIDATION (CANONICAL LGB_Bag3 vs A_hist_tuned & A_hist@0.50) ---")
    for seed in ["2026", "314159", "20261030"]:
        rt = confirm_res["canonical_bag3_vs_A_hist_tuned"][seed]
        r0 = confirm_res["canonical_bag3_vs_A_hist_050"][seed]
        rs = confirm_res["legacy_single_lgb_vs_A_hist_tuned_cx0011_comparison"][seed]
        print(
            f"Seed {seed} (Seen={rt['n_seen']}): "
            f"[vs A_hist_tuned LGB_Bag3] Flips={rt['n_flips']} ({rt['rate_per_1k_seen']}/1k), "
            f"Fixes/Breaks={rt['fixes']}/{rt['breaks']} (Net={rt['net']:+d}, CI95={rt['bootstrap_seller_ci95']}) | "
            f"[LGB_single was {rs['n_flips']} flips, Net={rs['net']:+d}] | "
            f"[vs A_hist@0.50 LGB_Bag3] Flips={r0['n_flips']} ({r0['rate_per_1k_seen']}/1k, Net={r0['net']:+d})"
        )

    print("\n--- 2. CONFIRM CROSS-FAMILY TRANSFER (CANONICAL LGB_Bag3 vs R156_Ensemble) ---")
    for seed, rr in confirm_res["cross_family_bag3_vs_R156_Ensemble"].items():
        print(
            f"Seed {seed} vs R156_Ensemble (Seen={rr['n_seen']}): "
            f"Flips={rr['n_flips']} ({rr['rate_per_1k_seen']}/1k), "
            f"Fixes/Breaks={rr['fixes']}/{rr['breaks']} (Net={rr['net']:+d}, CI95={rr['bootstrap_seller_ci95']})"
        )

    test_res = apply_rule_to_test()
    wf = test_res["within_family_vs_A_hist_inductive_70k_050"]
    cf = test_res["cross_family_transfer_vs_submission_48"]
    print("\n--- 3. TEST 30K APPLICATION (SEEN=21,487): WITHIN-FAMILY vs CROSS-FAMILY ---")
    print(
        f"Test vs A_hist_inductive_70k@0.50 (LGB_Bag3): Flips={wf['bag3_flips']} ({wf['bag3_rate_per_1k_seen']}/1k seen), "
        f"to_used={wf['bag3_to_used']}, to_new={wf['bag3_to_new']}"
    )
    print(
        f"Test vs submission_48             (LGB_Bag3): Flips={cf['bag3_flips']} ({cf['bag3_rate_per_1k_seen']}/1k seen), "
        f"to_used={cf['bag3_to_used']}, to_new={cf['bag3_to_new']}"
    )
    print(f"Candidate File: {test_res['candidate_file']} (IDs {test_res['id_range_verified']})")
    print(f"SHA256: {test_res['sha256']}")
    print(f"Guard Validation: {'PASS' if test_res['guard_passed'] else 'FAIL'}")

    report = {
        "rule_specification": {
            **RULE_CONFIG,
            "rule_config_sha256": RULE_CONFIG_SHA256,
            "train_baseline_within_family": "A_hist (Deep CatBoost R151)",
            "train_baseline_cross_family_proxy": "R156_Ensemble (Historical Ensemble on Seeds 2026 & 314159)",
            "test_within_family_reference": "A_hist_inductive_70k@0.50",
            "test_submission_reference": "submission_48_reference.csv (.94633)",
            "epistemological_status": "UNVALIDATED_EXPLORATORY_TRANSFER_PROBE_AGAINST_48_DO_NOT_PROMOTE",
            "non_independence_notice": (
                "Seeds 2026, 314159, and 20261030 overlap in confirm (~18.1%-18.6% pairwise). "
                "Counts across seeds MUST NOT be summed or treated as independent observations."
            ),
            "why_453_flips_on_test_vs_confirm_explanation": (
                "When applied on the 30k test set (21,487 seen rows) against its own baseline family "
                "(A_hist_inductive_70k@0.50), the frozen Bag3 rule produces only 26 flips (1.21 per 1,000 seen rows), "
                "matching the within-family confirm rate (0.52 to 3.84 per 1,000 seen rows). "
                "The jump to 453 flips (21.08 per 1,000 seen rows) when applied against submission_48 occurs because "
                "submission_48 comes from a different historical model family that disagrees with A_hist_inductive_70k "
                "on 1,081 test rows (712 seen rows = 33.14 per 1,000 seen rows). On confirm, applying the exact same "
                "Bag3 rule against the historical R156_Ensemble produces 22.29 and 21.37 flips per 1,000 seen rows "
                "(87 and 82 flips). Because neither A_hist nor R156_Ensemble is submission_48, B4 remains unvalidated "
                "against submission_48 and is NOT approved as a demonstrated improvement."
            ),
        },
        "confirm_evaluation": confirm_res,
        "test_candidate": test_res,
    }

    out_json = OUT_DIR / "single_intervention_rule_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report to: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
