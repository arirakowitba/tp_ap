"""Infer public leaderboard denominator from rounded aggregate accuracies.

Assumption: five-decimal scores except ledger values recorded with <=3 decimals,
which are conservatively treated at their recorded precision.
This infers no individual test labels.
"""
import json
import sys

def decimals(value):
    s = str(value)
    return len(s.partition(".")[2])

def candidates(entries, lo=15000, hi=30000):
    by_hash = {}
    for row in entries:
        if row.get("sha256_vector") and isinstance(row.get("score"), (int, float)):
            by_hash[row["sha256_vector"]] = row
    scores = list(by_hash.values())
    out = []
    for n in range(lo, hi + 1):
        bad = []
        for row in scores:
            score = row["score"]
            d = 3 if decimals(score) <= 3 else 5
            tolerance = n * 0.5 * 10 ** (-d)
            residual = abs(score * n - round(score * n))
            if residual > tolerance + 1e-8:
                bad.append(row.get("key"))
        if not bad:
            out.append(n)
    return {"usable_unique_scores": len(scores), "compatible_denominators": out,
            "assumption": "3-digit values have 3-digit recording precision; other values have 5-digit precision"}

if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as f:
        ledger = json.load(f)
    print(json.dumps(candidates(ledger["entries"]), indent=2))
