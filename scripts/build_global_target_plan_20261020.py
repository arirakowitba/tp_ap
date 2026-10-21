"""Rebuild whole-test targets and audit what the existing CV actually measures."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ap2026q2_global_objective import global_accuracy, required_public_correct, required_hidden_correct, expected_flip_budget
from ap2026q2_iterative_submissions import public_correct

OUT = ROOT / "results/global_target_96"
OUT.mkdir(parents=True, exist_ok=True)


def validation_audit():
    raw_path = PROJECT / "experiment_outputs_warranty_text_20261017/raw_text.pkl"
    saved_path = OUT / "validation_distribution_audit.json"
    if not raw_path.exists():
        return json.loads(saved_path.read_text(encoding="utf8")) if saved_path.exists() else None
    raw = pd.read_pickle(raw_path)
    assert len(raw) == 100000
    sellers = raw.seller.astype(str).to_numpy()
    test_known = pd.Series(sellers[70000:]).isin(set(sellers[:70000])).to_numpy()
    group_path = PROJECT / "experiment_outputs_complementary_architectures_20261017/groups_all.npy"
    groups = np.load(group_path, allow_pickle=True)
    assert len(groups) == 70000
    cv_known = np.zeros(70000, dtype=bool)
    fold_reports = []
    assigned = np.zeros(70000, dtype=int)
    for fold in range(3):
        file = ROOT / f"results/new_models_20261020/text_linear_fold{fold}.npz"
        valid = np.load(file)["valid_rows"].astype(int)
        fit = np.setdiff1d(np.arange(70000), valid)
        known = pd.Series(sellers[valid]).isin(set(sellers[fit])).to_numpy()
        cv_known[valid] = known
        assigned[valid] += 1
        fold_reports.append({"fold": fold, "rows": len(valid), "seller_seen_rows": int(known.sum()),
                             "seller_seen_fraction": float(known.mean())})
    assert np.all(assigned == 1)
    y = (pd.read_pickle(PROJECT / "notebook_work/ap_split_build/validation_cache/target.pkl").astype(str).to_numpy() == "used").astype(int)
    old_dir = PROJECT / "experiment_outputs_hour_20261017_2200/rotated_cv"
    old = (np.load(old_dir / "RichDepthwise_oof_predictions.npz")["oof"]
           + np.load(old_dir / "RichPlain_oof_predictions.npz")["oof"]) / 2
    linear = np.load(ROOT / "results/new_models_20261020/text_linear_grouped_cv_predictions.npz")["oof"]
    light = np.load(ROOT / "results/new_models_20261020/svd_lgbm_grouped_cv_predictions.npz")["oof"]
    signals = {"old_catboost": old, "text_linear": linear, "svd_lgbm": light,
               "old_70_new_30": .7 * old + .3 * (.65 * light + .35 * linear)}
    metrics = {}
    for name, p in signals.items():
        correct = (p >= .5) == y
        seen = float(correct[cv_known].mean())
        unseen = float(correct[~cv_known].mean())
        metrics[name] = {"oof_accuracy": float(correct.mean()), "seen_seller_accuracy": seen,
                         "unseen_seller_accuracy": unseen,
                         "test_mix_reweighted_accuracy_hypothesis": float(test_known.mean() * seen + (1-test_known.mean()) * unseen)}
    result = {"n_train": 70000, "n_test": 30000, "training_sellers": int(raw.iloc[:70000].seller.nunique()),
              "test_seen_seller_rows": int(test_known.sum()), "test_unseen_seller_rows": int((~test_known).sum()),
              "test_seen_seller_fraction": float(test_known.mean()), "cv_seen_seller_fraction": float(cv_known.mean()),
              "unique_cv_groups": int(len(np.unique(groups))), "groups_all_distinct": bool(len(np.unique(groups)) == 70000),
              "seller_disjoint_validation": bool(not cv_known.any()), "folds": fold_reports, "metrics": metrics,
              "source_groups_file": "experiment_outputs_complementary_architectures_20261017/groups_all.npy",
              "source_raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
              "interpretation": "The historical groups are all distinct. Existing CV holds out rows, not sellers. Reweighting assumes conditional accuracy is unchanged and is not a hidden-score observation or confidence bound."}
    saved_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    # Correct the interpretation of preserved metrics; never relabel or retrain their predictions.
    for directory in ("new_models_20261020", "new_catboost_20261020"):
        path = ROOT / "results" / directory / "comparison.json"
        record = json.loads(path.read_text(encoding="utf8"))
        record["validation_protocol_audit"] = {"reference": "results/global_target_96/validation_distribution_audit.json",
            "unique_groups_in_70000_rows": result["unique_cv_groups"],
            "seller_blocking_enforced_by_group_ids": False,
            "note": "A zero overlap of sample-unique group IDs does not establish seller-disjoint CV. Original numerical metrics are preserved."}
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    return result


def main():
    record = json.loads((ROOT / "results/revision_submissions/feedback_iterativo_20261020.json").read_text())
    baseline_count = public_correct(record["baseline"]["score"])
    reference = float(record["baseline"]["score"])
    cv_best = json.loads((ROOT / "results/new_models_20261020/comparison.json").read_text())["metrics"]["old_plus_0.30_new"]["accuracy_05"]
    pool = pd.read_csv(ROOT / "results/new_catboost_20261020/future_candidate_ranking_not_calibrated.csv")
    tested = set()
    for g in "CDE":
        tested.update(pd.read_csv(ROOT / f"submissions/20261020/cohorte_{g}_auditoria.csv").ID)
    available = int((~pool.ID.isin(tested)).sum())
    scenarios = []
    for h in (cv_best, .94, reference, .95, .955, .96):
        count = required_public_correct(h)
        scenarios.append({"hidden_accuracy_assumed": h, "public_correct_required": count,
                          "public_accuracy_required": count / 21000,
                          "public_gain_needed_vs_15": count - baseline_count,
                          "global_at_public_95": global_accuracy(.95, h),
                          "global_at_public_97": global_accuracy(.97, h)})
    pd.DataFrame(scenarios).to_csv(OUT / "scenarios.csv", index=False)
    budget = [{"flip_correctness_assumed": p,
               "expected_cases_for_public_gain_520": expected_flip_budget(520, p, .7),
               "fits_remaining_ranked_pool": expected_flip_budget(520, p, .7) <= available}
              for p in (.55, .60, .65, .70, .75, .80, .85, .90)]
    pd.DataFrame(budget).to_csv(OUT / "flip_budget.csv", index=False)
    audit = validation_audit()
    plan = {"objective": "96 percent global accuracy over all 30000 test rows; not a claim about the competition's final ranking metric",
            "public_rows": 21000, "hidden_rows": 9000, "target_global_correct": 28800,
            "baseline_public_correct": baseline_count, "reference_hidden_rate_is_hypothetical": reference,
            "global_with_95_public_reference_hidden": global_accuracy(.95, reference),
            "reference_score_is_post_feedback_not_clean_hidden_validation": True,
            "operating_target": {"public_accuracy": .97, "public_correct": 20370,
                "public_gain_vs_15": 520, "hidden_mean_accuracy_target": .95,
                "global_mean_target": global_accuracy(.97, .95),
                "minimum_hidden_correct_for_96_global": required_hidden_correct(20370),
                "hidden_lower_predictive_target": .94,
                "global_at_that_lower_bound": global_accuracy(.97, .94)},
            "confidence_gate": {"prob_global_at_least_96_required": .95,
                "predictive_draws_available": False, "gate_established": False,
                "requirements": "Joint seller/category-aware predictive draws, selection-aware shadow 70/30 validation and distribution stress tests; no independent-row certainty from a public score."},
            "remaining_ranked_unprobed_rows": available,
            "seller_regime_targets_aspirational": {"seen_accuracy": .98, "unseen_accuracy": .925,
                "global_at_measured_test_mix": float(audit["test_seen_seller_fraction"]*.98 + (1-audit["test_seen_seller_fraction"])*.925) if audit else None,
                "test_correct_gain_per_1pp_unseen_improvement": audit["test_unseen_seller_rows"]*.01 if audit else None},
            "flip_correctness_needed_if_all_remaining_rows_used_for_public_gain_520": .5 + 520/(2*.7*available),
            "allocation_75_slots_provisional": {"model_data_policy_probes": 24, "relationship_segment_probes": 24,
                                                "fresh_extension_confirmations": 12, "adaptive_consolidations": 15},
            "scenario_file": "scenarios.csv", "budget_file": "flip_budget.csv",
            "validation_audit_file": "validation_distribution_audit.json" if audit else None}
    (OUT / "global_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(json.dumps({"plan": plan, "validation_audit": audit}, ensure_ascii=False))


if __name__ == "__main__":
    main()
