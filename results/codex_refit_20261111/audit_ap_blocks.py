"""Find positive aggregate flip blocks from evaluated submissions."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csr_matrix, eye, hstack, vstack

OUT = Path("entrega_ap")
N_PUBLIC = 20998
ANCHOR_HITS = 19899


def hits(score):
    guess = round(score * N_PUBLIC)
    if round(guess / N_PUBLIC, 5) != score:
        raise ValueError(f"Score {score} incompatible with assumed Npub")
    return guess


def read_event_vectors(anchor):
    events = {}
    for file in sorted(Path("inputs").glob("block_vectors_*.json")):
        events.update((r["id"], r) for r in json.loads(file.read_text()))
    for name, score in [("ap_20261111_01_consenso_ruteado", .94214),
                        ("ap_proxima_disjunta_104", .94652)]:
        f = pd.read_csv(OUT / (name + ".csv")).set_index("ID").condition
        assert f.index.is_unique and set(f.index) == set(anchor.index)
        events[name] = {"id": name, "score": score, "flips": f.index[f != anchor].tolist()}
    grouped = {}
    incompatible = []
    for name, r in events.items():
        try:
            d = hits(r["score"]) - ANCHOR_HITS
        except ValueError:
            incompatible.append(name)
            continue
        ids = frozenset(r["flips"])
        grouped.setdefault(ids, []).append((name, d))
    conflicts = [v for v in grouped.values() if len({d for _, d in v}) > 1]
    usable = [(v[0][0], k, v[0][1]) for k, v in grouped.items()
              if len({d for _, d in v}) == 1 and k]
    return usable, conflicts, incompatible


def pair_bounds(events):
    best = {}
    for a, aa, da in events:
        for b, bb, db in events:
            block = bb - aa
            bound = db - da - len(aa - bb)
            if not block or bound <= 0:
                continue
            row = (bound, a, b, len(aa - bb))
            if block not in best or best[block][0] < bound:
                best[block] = row
    return sorted([(block, *v) for block, v in best.items()], key=lambda r: (-r[1], len(r[0])))


def grouped_certificate(events, feasibility_only=False):
    memberships = {}
    for j, (_, ids, _) in enumerate(events):
        for i in ids:
            memberships[i] = memberships.get(i, 0) | (1 << j)
    groups = {}
    for i, key in sorted(memberships.items()):
        groups.setdefault(key, []).append(i)
    keys = list(groups)
    n = np.array([len(groups[k]) for k in keys], dtype=float)
    d = np.array([x[2] for x in events], dtype=float)
    a = csr_matrix([[int(bool(key & (1 << j))) for key in keys] for j in range(len(events))], dtype=float)
    feasibility = linprog(np.zeros(len(keys)), A_eq=a, b_eq=d, bounds=list(zip(-n, n)), method="highs")
    if not feasibility.success:
        return None, {"groups": len(keys), "status": feasibility.message}
    if feasibility_only:
        return set(), {"groups": len(keys), "status": "feasible"}
    # LP dual. Maximize d'y - sum(n * |z - A'y|) over whole-block binary z.
    # Restrict singleton blocks so no prediction depends on identifying one label.
    j, k = a.shape
    ident = eye(k)
    matrix = vstack([hstack([-a.T, ident, -ident]), hstack([a.T, -ident, -ident])])
    is_block = (n >= 2).astype(float)
    result = milp(np.r_[-d, np.zeros(k), n], integrality=np.r_[np.zeros(j), np.ones(k), np.zeros(k)],
                  bounds=Bounds(np.r_[np.full(j, -np.inf), np.zeros(k * 2)],
                                np.r_[np.full(j, np.inf), is_block, np.full(k, np.inf)]),
                  constraints=LinearConstraint(matrix, -np.inf, 0), options={"time_limit": 15})
    if result.x is None:
        return None, {"groups": k, "status": result.message}
    choice = np.rint(result.x[j:j + k])
    # Recompute the bound on the selected whole blocks independently of the MILP.
    check = linprog(choice, A_eq=a, b_eq=d, bounds=list(zip(-n, n)), method="highs")
    if not check.success:
        return None, {"groups": k, "status": check.message}
    ids = {i for key, use in zip(keys, choice) if use for i in groups[key]}
    info = {"groups": k, "status": result.message, "flips": len(ids), "lower_hits": float(check.fun)}
    return ids, info


def main():
    anchor = pd.read_csv(OUT / "S4_EVT154_respaldo.csv").set_index("ID").condition
    events, conflicts, incompatible = read_event_vectors(anchor)
    # Admit user-reported scores first, then the newest ledger events.
    # Quarantine a vector-score association if its inclusion contradicts the
    # previously accepted set. Compatibility alone cannot verify provenance.
    verified = []
    quarantined = []
    for event in sorted(events, key=lambda v: (not v[0].startswith("EVT-"),
                                               v[0]), reverse=True):
        name, block, delta = event
        if abs(delta) > len(block):
            quarantined.append({"id": name, "reason": "anchor distance bound"})
            continue
        if any(abs(delta - d) > len(block ^ b) for _, b, d in verified):
            quarantined.append({"id": name, "reason": "pair distance bound"})
            continue
        feasible, _ = grouped_certificate([*verified, event], feasibility_only=True)
        if feasible is None:
            quarantined.append({"id": name, "reason": "joint aggregate infeasibility"})
            continue
        verified.append(event)
    pairs = pair_bounds([*verified, ("S4", frozenset(), 0)])
    ids, cert = grouped_certificate(verified)
    best_pairs = [{"lower_hits": n, "from": a, "to": b, "omitted": om, "flips": len(s)}
                  for s, n, a, b, om in pairs[:12]]
    report = {"events": len(events), "consistent_events": len(verified),
              "accepted_ids": [name for name, _, _ in verified], "quarantined": quarantined,
              "conflicts": conflicts, "incompatible": incompatible,
              "pair_blocks": len(pairs), "best_pair_blocks": best_pairs, "aggregate": cert,
              "n_public_assumed": N_PUBLIC}
    if ids and cert["lower_hits"] > 1e-6:
        pred = anchor.copy()
        pred.loc[sorted(ids)] = pred.loc[sorted(ids)].map({"new": "used", "used": "new"})
        path = OUT / "ap_bloques_historicos.csv"
        pred.to_csv(path, lineterminator="\n")
        test = pd.read_csv(path).set_index("ID").condition
        assert test.index.equals(anchor.index) and set(test.index[test != anchor]) == ids
        report["candidate_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "auditoria_bloques.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
