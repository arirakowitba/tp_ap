"""Audit accuracy denominators from rounded aggregate scores."""
import hashlib
import json
from pathlib import Path
import sys
from decimal import Decimal


ANCHOR = 94766
TARGET = 95404
RECENT = {94214, 94652, 94509, 94538, 94480}
SCALE = 100000


def hits(score, n):
    h = (score * n + SCALE // 2) // SCALE
    # Upper boundary is open under half-up. At n <= 30000 at most one hit count fits.
    return h if -n <= 2 * (h * SCALE - score * n) < n else None


def main():
    files = sorted(Path(sys.argv[1]).glob("block_vectors_*.json"))
    assert files
    events = [e for f in files for e in json.loads(f.read_text(), parse_float=Decimal)]
    # Strings with fewer digits are ambiguous between trailing zeros and coarse logs.
    history = {int(e["score"] * SCALE) for e in events if e["score"].as_tuple().exponent == -5}
    recent = RECENT | {ANCHOR}
    all_scores = history | recent
    candidates = range(1, 30001)
    missed = [(n, sum(hits(s, n) is None for s in all_scores)) for n in candidates]
    n_exact = [n for n, k in missed if k == 0]
    report = {
        "scope": "Aggregate accuracy only; conditional on common denominator and five-decimal round-half-up. "
                 "Ledger scores have not been reconciled to official submission IDs.",
        "historical_rows": len(events),
        "distinct_five_decimal_scores_with_recent": len(all_scores),
        "short_precision_rows_omitted": sum(e["score"].as_tuple().exponent != -5 for e in events),
        "recent_and_anchor_compatible": [n for n in candidates if all(hits(s,n) is not None for s in recent)],
        "all_compatible": n_exact,
        "one_score_discard_compatible": [n for n, k in missed if k <= 1],
        "two_score_discard_compatible": [n for n, k in missed if k <= 2],
        "exact_results": {str(n): {"anchor_hits": hits(ANCHOR,n), "target_hits": hits(TARGET,n),
                                    "gain_to_target": hits(TARGET,n)-hits(ANCHOR,n)}
                          for n in n_exact if hits(TARGET,n) is not None},
        "historical_input_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
