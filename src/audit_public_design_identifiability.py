"""Audit the identifiability of per-ID public-score probes.

Usage: python src/audit_public_design_identifiability.py LEDGER.json
No test labels are read or inferred.
"""
import json
import sys
from collections import Counter, defaultdict

def audit(path):
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)["entries"]
    usable = [e for e in entries if e.get("sha256_vector") and isinstance(e.get("flipped_ids"), list)]
    unique = list({e["sha256_vector"]: e for e in usable}.values())
    sets = [set(map(int, e["flipped_ids"])) for e in unique]
    assert all(all(1 <= i <= 30000 for i in s) for s in sets)
    groups = defaultdict(list)
    for item_id in range(1, 30001):
        signature = tuple(j for j, flips in enumerate(sets) if item_id in flips)
        groups[signature].append(item_id)
    sizes = Counter(map(len, groups.values()))
    return {
        "usable_entries": len(usable),
        "unique_vectors": len(unique),
        "unique_intervention_signatures": len(groups),
        "never_flipped_ids": len(groups.get((), [])),
        "touched_ids": 30000 - len(groups.get((), [])),
        "singleton_signatures": sizes[1],
        "ids_in_non_singleton_signatures": sum(n * count for n, count in sizes.items() if n > 1),
        "largest_signature_sizes": sorted((len(v) for v in groups.values()), reverse=True)[:10],
        "interpretation": "Identical design signatures cannot be separated by aggregate scores alone; public membership is also unknown.",
    }

if __name__ == "__main__":
    print(json.dumps(audit(sys.argv[1]), indent=2, ensure_ascii=False))
