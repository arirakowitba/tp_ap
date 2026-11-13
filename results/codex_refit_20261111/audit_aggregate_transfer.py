"""Replay simple aggregate score predictors on strict historical day blocks."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from audit_public_denominator import hits

N_PUBLIC = 20998


def main():
    folder = Path(sys.argv[1])
    catalog_files = sorted(folder.glob("ledger.*"))
    vectors = sorted(folder.glob("block_vectors_*.json"))
    catalog = {e["event_id"]: e for e in json.loads("".join(p.read_text() for p in catalog_files))}
    rows = []
    for p in vectors:
        for v in json.loads(p.read_text()):
            e = catalog.get(v["id"])
            if e is None or not e["date"] or not e["public_score"] or e["is_alias"]:
                continue
            if e["public_score"] != v["score"]:
                continue
            n = hits(round(v["score"] * 100000), N_PUBLIC)
            if n is not None:
                rows.append((e["date"][:10], v["id"], frozenset(v["flips"]), n))
    rows.sort(key=lambda x: (x[0], x[1]))
    assert len({r[1] for r in rows}) == len(rows)
    groups = {}
    conflicts = set()
    for day, event, mask, value in rows:
        previous = groups.setdefault(mask, (day, event, mask, value))
        if previous[-1] != value:
            conflicts.add(mask)
    rows = [r for mask, r in groups.items() if mask not in conflicts]
    historical = []
    residuals = {k: [] for k in ("zero", "nearest", "shrinkage")}
    results = []
    for day in sorted({r[0] for r in rows}):
        batch = [r for r in rows if r[0] == day]
        if len(historical) >= 10:
            _, _, anchor_mask, anchor = max(historical, key=lambda r: r[-1])
            day_records = []
            for _, _, mask, value in batch:
                distances = np.array([len(mask ^ r[2]) for r in historical])
                deltas = np.array([r[-1] - anchor for r in historical])
                assert np.all(distances > 0)
                weight = 1 / np.maximum(distances, 10)**2
                predictions = {"zero": 0., "nearest": float(deltas[distances.argmin()]),
                               "shrinkage": float(np.dot(weight, deltas)/(weight.sum()+1/100**2))}
                # Use only previous days' errors to form empirical intervals.
                record = {}
                for name, pred in predictions.items():
                    error = value-anchor-pred
                    previous_errors = residuals[name]
                    cover = None
                    if len(previous_errors) >= 10:
                        lo, hi = np.quantile(previous_errors, [.05, .95])
                        cover = bool(lo <= error <= hi)
                    record[name] = (float(error), cover, value > anchor, pred > 0.)
                day_records.append(record)
            for name in residuals:
                residuals[name].extend(r[name][0] for r in day_records)
            results.append((day, day_records))
        historical.extend(batch)
    stats = {}
    for name in residuals:
        data = [r[name] for _, day in results for r in day]
        errs = np.array([x[0] for x in data])
        coverage = [x[1] for x in data if x[1] is not None]
        stats[name] = {"mae_hits": float(abs(errs).mean()), "bias_actual_minus_pred": float(errs.mean()),
                       "n_interval_events": len(coverage), "covered_90": sum(coverage),
                       "predicted_improvements": sum(x[3] for x in data),
                       "real_improvements": sum(x[2] for x in data),
                       "correct_predicted_improvements": sum(x[2] and x[3] for x in data)}
    report = {
        "scope": "Retrospective day-block replay using catalog dates, deduplicated exact aggregate masks. "
                 "No prospective proof, no outcome-derived row labels, no extrapolation to future candidate families.",
        "matched_rows": len(rows), "conflicting_masks_removed": len(conflicts),
        "evaluated_events": sum(len(r) for _, r in results),
        "evaluated_day_counts": {day: len(r) for day,r in results}, "metrics": stats,
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [*catalog_files,*vectors]},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
