"""Audit closing vectors using aggregate scores and public-set capacity."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


def vector(path, size):
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    lookup = {int(row["ID"]): row["condition"] for row in rows}
    if len(rows) != size or set(lookup) != set(range(1, size + 1)):
        raise ValueError(f"Invalid ID universe: {path}")
    if not set(lookup.values()) <= {"used", "new"}:
        raise ValueError(f"Invalid classes: {path}")
    return lookup


def changes(anchor, candidate):
    return frozenset(i for i, label in anchor.items() if candidate[i] != label)


def run(args):
    anchor = vector(args.anchor, args.size)
    historical = json.loads(args.history.read_text())
    fifth = changes(anchor, vector(args.fifth, args.size))
    negative_names = {
        "ap_20261111_01_consenso_ruteado", "ap_proxima_disjunta_104",
        "ap_quinta_catboost_disjunta", "ap_probe_A_refit_exclusivo",
        "ap_probe_B_ag_exclusivo", "A", "B", "C", "D",
    }
    negative_masks = [set(ids) for name, ids, delta in historical if name in negative_names and delta < 0]
    negative_masks.append(fifth)
    disjoint_loss = set().union(*negative_masks)
    if len(negative_masks) != 10 or len(disjoint_loss) != sum(map(len, negative_masks)):
        raise ValueError("The ten documented negative masks must be mutually disjoint")
    historical.append(["fifth", fifth, args.fifth_delta])
    events = {}
    duplicates = 0
    for name, ids, delta in historical:
        mask = frozenset(ids)
        if not mask <= anchor.keys() or abs(delta) > len(mask):
            raise ValueError(f"Invalid historical constraint: {name}")
        if mask in events:
            if events[mask] != delta:
                raise ValueError(f"Conflicting historical constraint: {name}")
            duplicates += 1
        events[mask] = delta
    historical_masks = list(events)
    candidates = [vector(path, args.size) for path in args.candidate]
    candidate_masks = [changes(anchor, v) for v in candidates]
    signatures = [0] * (args.size + 1)
    for j, mask in enumerate(historical_masks + candidate_masks):
        for i in mask:
            signatures[i] |= 1 << j
    patterns = Counter(signatures[1:])
    counts = np.array(list(patterns.values()))
    n = len(patterns)
    membership = np.array(
        [[(key >> j) & 1 for key in patterns] for j in range(len(events) + len(candidates))],
        dtype=float,
    )
    equalities = np.vstack((
        np.column_stack((membership[:len(events)], -membership[:len(events)])),
        np.r_[np.ones(n), np.zeros(n)],
        np.r_[np.zeros(n), np.ones(n)],
    ))
    rhs = np.r_[list(events.values()), args.public_size - args.anchor_hits, args.anchor_hits]
    capacities = np.column_stack((np.eye(n), np.eye(n)))
    known = set().union(*historical_masks)
    out = {
        "anchor_sha256": hashlib.sha256(args.anchor.read_bytes()).hexdigest(),
        "history_sha256": hashlib.sha256(args.history.read_bytes()).hexdigest(),
        "fifth_sha256": hashlib.sha256(args.fifth.read_bytes()).hexdigest(),
        "event_constraints": len(events),
        "duplicate_constraints": duplicates,
        "covered_rows": len(known),
        "fifth_changes": len(fifth),
        "fifth_delta": args.fifth_delta,
        "public_size_assumption": args.public_size,
        "anchor_hits": args.anchor_hits,
        "candidates": {},
    }
    for j, (path, labels, mask) in enumerate(zip(args.candidate, candidates, candidate_masks)):
        coeff = membership[len(events) + j]
        objective = np.r_[coeff, -coeff]
        low = linprog(objective, A_ub=capacities, b_ub=counts, A_eq=equalities, b_eq=rhs, method="highs")
        high = linprog(-objective, A_ub=capacities, b_ub=counts, A_eq=equalities, b_eq=rhs, method="highs")
        if not (low.success and high.success):
            raise ValueError((low.message, high.message))
        out["candidates"][path.name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "map_sha256": hashlib.sha256("".join(f"{i}:{labels[i]}\n" for i in sorted(labels)).encode()).hexdigest(),
            "flips": len(mask),
            "all_history_overlap": len(mask & known),
            "ten_negative_blocks_overlap": len(mask & disjoint_loss),
            "fifth_overlap": len(mask & fifth),
            "same_vector_in_history": mask in events,
            "public_net_lp_bounds": [round(low.fun, 8), round(-high.fun, 8)],
        }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchor", type=Path, required=True)
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--fifth", type=Path, required=True)
    p.add_argument("--candidate", type=Path, nargs="+", required=True)
    p.add_argument("--size", type=int, default=30000)
    p.add_argument("--public-size", type=int, default=20998)
    p.add_argument("--anchor-hits", type=int, default=19899)
    p.add_argument("--fifth-delta", type=int, default=-85)
    run(p.parse_args())
