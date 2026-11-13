"""Reconcile the Nov 13 scores and bound whole-vector gains."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import hypergeom

old, today = (Path(x).resolve() for x in sys.argv[1:3])
sys.path.insert(0, str(old))
import audit_ap_blocks as audit

SCORES = {"A": .94576, "B": .94642, "C": .94690, "D": .94657}
OLDER = {
    "ap_20261111_01_consenso_ruteado": .94214, "ap_proxima_disjunta_104": .94652,
    "ap_quinta_catboost_disjunta": .94509, "ap_probe_A_refit_exclusivo": .94538,
    "ap_probe_B_ag_exclusivo": .94480,
}
anchor = pd.read_csv(old / "entrega_ap/S4_EVT154_respaldo.csv").set_index("ID").condition


def mask(path):
    candidate = pd.read_csv(path).set_index("ID").condition
    assert candidate.index.equals(anchor.index) and candidate.isin(("used", "new")).all()
    return frozenset(candidate.index[candidate != anchor])


def bound(support, observed):
    masks = [x[1] for x in observed] + [support]
    membership = {}
    for j, ids in enumerate(masks):
        for i in ids:
            membership[i] = membership.get(i, 0) | (1 << j)
    patterns, count = np.unique(np.array(list(membership.values()), dtype=object), return_counts=True)
    matrix = np.array([[bool(k & (1 << j)) for k in patterns] for j in range(len(masks))], dtype=float)
    limits = []
    for sign in (1, -1):
        res = linprog(sign * matrix[-1], A_eq=matrix[:-1], b_eq=[r[-1] for r in observed],
                      bounds=list(zip(-count, count)), method="highs")
        assert res.success, res.message
        limits.append(round(sign * float(res.fun), 7))
    return limits


def main():
    observed = [(name, mask(old / "entrega_ap" / f"{name}.csv"), audit.hits(score)-audit.ANCHOR_HITS)
                for name, score in OLDER.items()]
    paths = {tag: next(today.glob(f"ap_20261113_{tag}_*.csv")) for tag in SCORES}
    for tag, score in SCORES.items():
        observed.append((tag, mask(paths[tag]), audit.hits(score)-audit.ANCHOR_HITS))
    assert len(set().union(*(r[1] for r in observed))) == sum(len(r[1]) for r in observed)
    os.chdir(old)
    history, conflicts, incompatible = audit.read_event_vectors(anchor)
    accepted = observed.copy()
    rejected = []
    for event in sorted(history, key=lambda r: r[0], reverse=True):
        name, ids, delta = event
        same = [gain for _, m, gain in accepted if m == ids]
        if same:
            if delta not in same:
                rejected.append(name)
            continue
        if any(abs(delta-gain) > len(ids ^ m) for _, m, gain in accepted):
            rejected.append(name)
            continue
        if audit.grouped_certificate([*accepted, event], feasibility_only=True)[0] is None:
            rejected.append(name)
        else:
            accepted.append(event)
    models = {}
    for name in ("ap_20261113_R_consenso_fuera_proxy.csv", "ap_20261113_05_consenso_cruzado.csv"):
        path = today / name
        ids = mask(path)
        assert not ids & set().union(*(r[1] for r in observed))
        maximum = len(ids)
        limits = bound(ids, accepted)
        models[name] = {"flips": maximum, "bounds": limits,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "support_only_ceiling_score": (audit.ANCHOR_HITS+maximum)/audit.N_PUBLIC,
                        "compatible_history_ceiling_score": (audit.ANCHOR_HITS+limits[1])/audit.N_PUBLIC,
                        "optimistic_uniform_public_tail": float(hypergeom.sf(
                            49, 30000, maximum, audit.N_PUBLIC))}
    out = {
        "source": "Four Nov 13 scores reported by Ariel. Submission IDs and timestamps not independently reconciled.",
        "anchor_score": .94766, "n_public_assumed": audit.N_PUBLIC, "net_to_exceed_095": 50,
        "today": {tag: {"score": SCORES[tag], "flips": len(ids), "net": gain,
                        "sha256": hashlib.sha256(paths[tag].read_bytes()).hexdigest()}
                  for tag, ids, gain in observed[-4:]},
        "today_union_net": sum(r[-1] for r in observed[-4:]),
        "today_union_score": (audit.ANCHOR_HITS+sum(r[-1] for r in observed[-4:]))/audit.N_PUBLIC,
        "nine_union_net": sum(r[-1] for r in observed),
        "nine_union_support": sum(len(r[1]) for r in observed),
        "historical_aggregate_constraints_including_nine": len(accepted),
        "historical_quarantine": {"incompatible": incompatible, "same_vector_conflicts": conflicts,
                                  "rejected": rejected},
        "candidates": models,
        "method": "Linear relaxation of signed net contributions per membership pattern. "
                  "No individual labels or public membership recovered. Optimistic tail assumes every flip is correct "
                  "and a uniform public subset. It is not a calibrated estimate.",
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
