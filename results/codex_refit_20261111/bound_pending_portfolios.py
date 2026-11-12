"""Bound net public gain of fixed vectors using only aggregate score constraints."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linprog

OLD = Path(sys.argv[1])
sys.path.insert(0, str(OLD))
import audit_ap_blocks as audit

root = OLD / "entrega_ap"
anchor = pd.read_csv(root / "S4_EVT154_respaldo.csv").set_index("ID").condition


def flips(path):
    f = pd.read_csv(path).set_index("ID").condition
    assert f.index.is_unique and set(f.index) == set(anchor.index)
    assert f.isin(("new", "used")).all()
    return frozenset(anchor.index[f.reindex(anchor.index).ne(anchor)])


five = {
    "ap_20261111_01_consenso_ruteado": .94214,
    "ap_proxima_disjunta_104": .94652,
    "ap_quinta_catboost_disjunta": .94509,
    "ap_probe_A_refit_exclusivo": .94538,
    "ap_probe_B_ag_exclusivo": .94480,
}
known = []
for name, score in five.items():
    known.append((name, flips(root / (name + ".csv")), audit.hits(score) - audit.ANCHOR_HITS))
os.chdir(OLD)
events, _, _ = audit.read_event_vectors(anchor)
compatible = list(known)
for event in sorted(events, key=lambda row: (not row[0].startswith("EVT-"), row[0]), reverse=True):
    _, ids, delta = event
    if any(ids == b for _, b, _ in compatible):
        continue
    if abs(delta) > len(ids) or any(abs(delta - d) > len(ids ^ b) for _, b, d in compatible):
        continue
    if audit.grouped_certificate([*compatible, event], feasibility_only=True)[0] is not None:
        compatible.append(event)


def bound(candidate, constraints):
    codes = {}
    for j, ids in enumerate([row[1] for row in constraints] + [candidate]):
        for i in ids:
            codes[i] = codes.get(i, 0) | (1 << j)
    counter = Counter(codes.values())
    patterns = list(counter)
    mat = np.array([[int(bool(k & (1 << j))) for k in patterns] for j in range(len(constraints) + 1)])
    n = np.array([counter[k] for k in patterns])
    d = [row[2] for row in constraints]
    mins = []
    for sign in (1, -1):
        r = linprog(sign * mat[-1], A_eq=mat[:-1], b_eq=d, bounds=list(zip(-n, n)), method="highs")
        assert r.success, r.message
        mins.append(sign * float(r.fun))
    return mins


sources = [root / "ap_20261111_02_refit_ruteado.csv", root / "ap_20261111_03_especialistas_ag.csv"]
if len(sys.argv) > 2:
    sources.append(Path(sys.argv[2]))
report = {"constraints": len(compatible), "public_n_assumed": audit.N_PUBLIC, "vectors": {}}
for source in sources:
    candidate = flips(source)
    report["vectors"][source.stem] = {
        "flips": len(candidate), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "overlap": {n: len(candidate & ids) for n, ids, _ in known},
        "net_bounds_five": bound(candidate, known), "net_bounds_all_compatible": bound(candidate, compatible)}
print(json.dumps(report, indent=2))
