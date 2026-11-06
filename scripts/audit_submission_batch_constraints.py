#!/usr/bin/env python3
"""Audit aggregate leaderboard evidence without inferring per-ID labels.

Example:
  python scripts/audit_submission_batch_constraints.py \
    --anchor top60.csv --anchor-name top60 --anchor-score 0.94709 \
    --candidate slot1 slot1.csv 0.94695 \
    --candidate slot2 slot2.csv 0.94690 \
    --public-size 20998 --output audit.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


VALID = {"new", "used"}


def read_vector(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty CSV")
        lower = {name.lower(): name for name in reader.fieldnames}
        id_col = lower.get("id") or lower.get("item_id")
        target_col = lower.get("condition") or lower.get("target")
        if not id_col or not target_col:
            raise ValueError(f"{path}: expected ID/item_id and condition/target")
        out: dict[str, str] = {}
        for row_num, row in enumerate(reader, 2):
            item_id = row[id_col].strip()
            target = row[target_col].strip().lower()
            if not item_id or target not in VALID:
                raise ValueError(f"{path}:{row_num}: invalid row")
            if item_id in out:
                raise ValueError(f"{path}:{row_num}: duplicate ID {item_id}")
            out[item_id] = target
    return out


def vector_hash(vector: dict[str, str]) -> str:
    payload = "".join(f"{item_id},{vector[item_id]}\n" for item_id in sorted(vector))
    return hashlib.sha256(payload.encode()).hexdigest()


def score_to_hits(score: float, n_public: int) -> dict[str, float | int]:
    raw = score * n_public
    nearest = round(raw)
    return {"raw": raw, "nearest_integer": nearest, "rounding_residual": raw - nearest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", required=True, type=Path)
    parser.add_argument("--anchor-name", default="anchor")
    parser.add_argument("--anchor-score", required=True, type=float)
    parser.add_argument("--candidate", nargs=3, action="append", metavar=("NAME", "CSV", "SCORE"), default=[])
    parser.add_argument("--public-size", type=int, required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    entries = [(args.anchor_name, args.anchor, args.anchor_score)]
    entries.extend((name, Path(path), float(score)) for name, path, score in args.candidate)
    vectors = {name: read_vector(path) for name, path, _ in entries}
    ids = set(vectors[args.anchor_name])
    for name, vector in vectors.items():
        if set(vector) != ids:
            raise ValueError(f"{name}: ID universe differs from anchor")

    scores = {name: score for name, _, score in entries}
    hits = {name: score_to_hits(score, args.public_size) for name, score in scores.items()}
    rows = {
        name: {
            "path": str(path),
            "score": score,
            "rows": len(vectors[name]),
            "normalized_vector_sha256": vector_hash(vectors[name]),
            "nearest_public_hits": hits[name]["nearest_integer"],
            "score_rounding_residual_hits": hits[name]["rounding_residual"],
        }
        for name, path, score in entries
    }

    pairwise = []
    names = [name for name, _, _ in entries]
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            changes = []
            directions: dict[str, int] = {}
            for item_id in sorted(ids):
                a, b = vectors[left][item_id], vectors[right][item_id]
                if a != b:
                    changes.append(item_id)
                    key = f"{a}->{b}"
                    directions[key] = directions.get(key, 0) + 1
            delta_hits = int(hits[right]["nearest_integer"]) - int(hits[left]["nearest_integer"])
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "n_changed_total": len(changes),
                    "directions": directions,
                    "changed_ids": changes,
                    "observed_delta_public_hits": delta_hits,
                    "interpretation": (
                        "Aggregate linear constraint only. It does not identify which changed IDs "
                        "were public or individually beneficial."
                    ),
                }
            )

    output = {
        "public_size_assumed": args.public_size,
        "entries": rows,
        "pairwise_constraints": pairwise,
        "warnings": [
            "Rounded public scores are aggregate evidence, never per-ID labels.",
            "Test rows outside the public subset contribute zero to a score difference.",
            "A positive batch delta does not imply every flip in that batch is beneficial.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
