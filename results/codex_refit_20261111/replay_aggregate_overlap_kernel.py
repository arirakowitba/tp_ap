"""Evaluate aggregate gain regression using only day-block historical overlap."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from audit_public_denominator import hits


def read(folder):
    catalog = sorted(folder.glob("ledger.*"))
    vectors = sorted(folder.glob("block_vectors_*.json"))
    meta = {r["event_id"]: r for r in json.loads("".join(p.read_text() for p in catalog))}
    grouped = {}
    for path in vectors:
        for v in json.loads(path.read_text()):
            m = meta.get(v["id"])
            if not m or not m["date"] or m["is_alias"] or m["public_score"] != v["score"]:
                continue
            value = hits(round(v["score"] * 100000), 20998)
            if value is not None:
                grouped.setdefault(frozenset(v["flips"]), []).append((m["date"][:10], value))
    rows = sorted((min(r)[0], mask, r[0][1]) for mask, r in grouped.items()
                  if len({value for _, value in r}) == 1)
    return rows, catalog + vectors


def predict(masks, values, anchor_mask, anchor_value, queries):
    # Kernel on signed differences from the current best vector. Only aggregate
    # inner products are retained. Distinct candidate masks can exceed the anchor.
    def dot(a, b):
        return len(a & b)

    centered = [mask ^ anchor_mask for mask in masks]
    gram = np.asarray([[dot(a, b) for b in centered] for a in centered], dtype=float)
    coef = np.linalg.solve(gram + 50 * np.eye(len(gram)), np.asarray(values) - anchor_value)
    return [float(np.dot([dot(q ^ anchor_mask, m) for m in centered], coef)) for q in queries]


def main():
    rows, files = read(Path(sys.argv[1]))
    past = []
    errors = {"zero": [], "overlap_ridge": []}
    outcome = {name: {"positives": 0, "true_positives": 0, "eligible_intervals": 0, "covered": 0}
               for name in errors}
    evaluated_days = []
    observed_improvements = 0
    for date in sorted({r[0] for r in rows}):
        batch = [r for r in rows if r[0] == date]
        if len(past) >= 10:
            _, anchor_mask, anchor = max(past, key=lambda row: row[-1])
            observed = np.asarray([v-anchor for _, _, v in batch])
            estimated = {"zero": np.zeros(len(batch)), "overlap_ridge": np.asarray(predict(
                [r[1] for r in past], [r[-1] for r in past], anchor_mask, anchor, [r[1] for r in batch]))}
            for name, values in estimated.items():
                err = observed - values
                metrics = outcome[name]
                metrics["positives"] += int((values > 0).sum())
                metrics["true_positives"] += int(((values > 0) & (observed > 0)).sum())
                if len(errors[name]) >= 10:
                    lo, hi = np.quantile(errors[name], [.05, .95])
                    metrics["eligible_intervals"] += len(batch)
                    metrics["covered"] += int(((err >= lo) & (err <= hi)).sum())
                errors[name].extend(err.tolist())
            evaluated_days.append((date, len(batch)))
            observed_improvements += int((observed > 0).sum())
        past.extend(batch)
    for name, values in errors.items():
        outcome[name]["mae_hits"] = float(np.abs(values).mean())
    print(json.dumps({"scope": "Retrospective day-block replay; catalogue dates and vectors lack prospective certification.",
                      "model": "Signed difference Gram kernel around the contemporaneous anchor, fixed ridge 50.",
                      "matched_rows": len(rows), "evaluated_days": evaluated_days,
                      "observed_improvements": observed_improvements, "metrics": outcome,
                      "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}, indent=2))


if __name__ == "__main__":
    main()
