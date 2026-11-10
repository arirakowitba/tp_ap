#!/usr/bin/env python3
"""One-sided prequential conformal gate for public leaderboard score.

This gate does NOT infer individual test labels. It calibrates the complete
candidate-generation procedure using predictions recorded before their public
scores were observed.

Finite-sample marginal coverage is guaranteed only under exchangeability of
the accepted calibration events and the new frozen candidate. When screening
multiple candidates, Bonferroni correction is applied.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_CAL_COLUMNS = {
    "event_id",
    "policy_family",
    "vector_sha256",
    "predicted_public_score",
    "observed_public_score",
    "prediction_time_utc",
    "score_time_utc",
    "is_prequential",
    "is_frozen",
    "is_comparable",
}
REQUIRED_CAND_COLUMNS = {
    "candidate_id",
    "policy_family",
    "vector_sha256",
    "predicted_public_score",
    "frozen_time_utc",
}


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_time(value: str) -> datetime:
    value = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError(f"timestamp lacks timezone: {value}")
    return dt.astimezone(timezone.utc)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_hash(value: str) -> bool:
    value = value.strip().lower()
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def conformal_quantile(errors: list[float], coverage: float) -> tuple[float | None, int]:
    """Return ceil((n+1)*coverage)-th order statistic (1-indexed).

    None means finite-sample coverage cannot be certified with n observations.
    """
    n = len(errors)
    k = math.ceil((n + 1) * coverage)
    if k > n:
        return None, k
    return sorted(errors)[k - 1], k


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--target", type=float, default=0.95000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--rounding-half-unit", type=float, default=0.000005)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not 0.0 < args.confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    cal_rows = read_csv(args.calibration)
    cand_rows = read_csv(args.candidates)
    if not cal_rows or not cand_rows:
        raise ValueError("calibration and candidates must be non-empty")
    if not REQUIRED_CAL_COLUMNS.issubset(cal_rows[0]):
        raise ValueError(f"missing calibration columns: {sorted(REQUIRED_CAL_COLUMNS - set(cal_rows[0]))}")
    if not REQUIRED_CAND_COLUMNS.issubset(cand_rows[0]):
        raise ValueError(f"missing candidate columns: {sorted(REQUIRED_CAND_COLUMNS - set(cand_rows[0]))}")

    # Multiple screening requires family-wise correction.
    m = len(cand_rows)
    alpha_family = 1.0 - args.confidence
    alpha_each = alpha_family / m
    coverage_each = 1.0 - alpha_each

    results = []
    for candidate in cand_rows:
        family = candidate["policy_family"].strip()
        reasons = []
        if not validate_hash(candidate["vector_sha256"]):
            reasons.append("invalid_candidate_sha256")
        try:
            frozen_time = parse_time(candidate["frozen_time_utc"])
        except Exception:
            frozen_time = None
            reasons.append("invalid_candidate_frozen_time")

        valid = []
        rejected = []
        for row in cal_rows:
            row_reasons = []
            if row["policy_family"].strip() != family:
                row_reasons.append("different_policy_family")
            if not parse_bool(row["is_prequential"]):
                row_reasons.append("not_prequential")
            if not parse_bool(row["is_frozen"]):
                row_reasons.append("not_frozen")
            if not parse_bool(row["is_comparable"]):
                row_reasons.append("not_comparable")
            if not validate_hash(row["vector_sha256"]):
                row_reasons.append("invalid_sha256")
            try:
                pred_time = parse_time(row["prediction_time_utc"])
                score_time = parse_time(row["score_time_utc"])
                if pred_time >= score_time:
                    row_reasons.append("prediction_not_before_score")
                if frozen_time is not None and score_time >= frozen_time:
                    # Calibration must precede the new frozen decision.
                    row_reasons.append("calibration_not_prior_to_candidate")
                predicted = float(row["predicted_public_score"])
                observed = float(row["observed_public_score"])
            except Exception:
                row_reasons.append("invalid_numeric_or_time")
                predicted = observed = 0.0
            if row_reasons:
                rejected.append({"event_id": row.get("event_id", ""), "reasons": row_reasons})
            else:
                # Worst-case true observed score under displayed-score rounding.
                observed_lower = observed - args.rounding_half_unit
                valid.append(
                    {
                        "event_id": row["event_id"],
                        "overprediction_error": predicted - observed_lower,
                    }
                )

        errors = [x["overprediction_error"] for x in valid]
        q, order_index = conformal_quantile(errors, coverage_each)
        predicted_new = float(candidate["predicted_public_score"])

        if q is None:
            reasons.append(
                f"insufficient_prequential_events:n={len(errors)},"
                f"required_at_least={math.ceil(coverage_each / (1.0 - coverage_each))}"
            )
            lower = None
            qualified = False
        else:
            lower = predicted_new - q
            # Conservative requirement for a displayed score at least target.
            qualified = lower >= args.target + args.rounding_half_unit
            if not qualified:
                reasons.append("lower_bound_below_target")

        results.append(
            {
                "candidate_id": candidate["candidate_id"],
                "vector_sha256": candidate["vector_sha256"],
                "policy_family": family,
                "predicted_public_score": predicted_new,
                "n_valid_prequential": len(valid),
                "n_rejected": len(rejected),
                "familywise_confidence": args.confidence,
                "per_candidate_coverage": coverage_each,
                "conformal_order_index": order_index,
                "overprediction_quantile": q,
                "lower_public_score_bound": lower,
                "target": args.target,
                "qualified": bool(qualified and not reasons),
                "status": "QUALIFIED" if qualified and not reasons else "NO_QUALIFIED",
                "reasons": reasons,
                "rejected_events": rejected,
            }
        )

    report = {
        "method": "one_sided_split_conformal_prequential_score_gate",
        "guarantee_scope": (
            "Finite-sample marginal coverage under exchangeability of prior frozen "
            "prequential events and the new candidate; Bonferroni family-wise correction."
        ),
        "does_not_guarantee": (
            "Coverage under policy drift, post-hoc predictions, non-comparable probes, "
            "or adversarial dependence. It does not infer individual labels."
        ),
        "candidate_count": m,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all(r["qualified"] for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
