"""Audit aggregate-score vector coverage; never infer hidden labels from scores."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def csv_vector(path):
    raw = Path(path).read_bytes()
    with Path(path).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    pred = {int(row["ID"]): row["condition"] for row in rows}
    if len(rows) != 30000 or len(pred) != 30000 or set(pred) != set(range(1, 30001)):
        raise ValueError(f"Invalid ID coverage: {path}")
    if set(pred.values()) - {"used", "new"}:
        raise ValueError(f"Invalid targets: {path}")
    return pred, hashlib.sha256(raw).hexdigest()


def binary_rank(sets):
    ids = sorted(set().union(*sets))
    index = {value: col for col, value in enumerate(ids)}
    pivots = {}
    for row in sets:
        bits = 0
        for value in row:
            bits ^= 1 << index[value]
        while bits:
            pivot = bits.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = bits
                break
            bits ^= pivots[pivot]
    return len(ids), len(pivots)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    parser.add_argument("d1d2_csv", type=Path)
    parser.add_argument("--recent", nargs=5, action="append",
                        metavar=("KEY", "CSV", "SHA256_LF", "N48", "ND12"), default=[])
    args = parser.parse_args()
    entries = json.loads(args.ledger.read_text(encoding="utf-8"))["entries"]
    base, _ = csv_vector(args.d1d2_csv)
    anchor = next(e for e in entries if e["key"] == "candidate_d1_d2_combined_on48")
    s48 = base.copy()
    for item_id in anchor["flipped_ids"]:
        s48[item_id] = "new" if base[item_id] == "used" else "used"
    observed = [(e["key"], set(e["flipped_ids"])) for e in entries
                if isinstance(e.get("flipped_ids"), list)
                and len(e["flipped_ids"]) == e.get("n_flips_vs_48")]
    missing = [e["key"] for e in entries if e["key"] not in {key for key, _ in observed}]
    recent = {}
    for key, filename, expected_hash, declared_48, declared_d12 in args.recent:
        pred, digest = csv_vector(filename)
        if digest != expected_hash:
            raise ValueError(f"Hash mismatch: {key}")
        changed = {item_id for item_id in pred if pred[item_id] != s48[item_id]}
        distance_d12 = sum(pred[item_id] != base[item_id] for item_id in pred)
        observed.append((key, changed))
        recent[key] = {"declared_vs_48": int(declared_48),
                       "actual_vs_48": len(changed),
                       "declared_vs_d1d2": int(declared_d12),
                       "actual_vs_d1d2": distance_d12}
    n_ids, rank = binary_rank([row for _, row in observed])
    print(json.dumps({"ledger_score_rows": len(entries), "vector_rows": len(observed),
                      "missing_vector_keys": missing, "intervened_ids": n_ids,
                      "gf2_rank": rank, "gf2_nullity": n_ids - rank,
                      "recent_count_checks": recent}, indent=2))


if __name__ == "__main__":
    main()
