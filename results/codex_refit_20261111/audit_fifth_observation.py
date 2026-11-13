"""Verify the reported fifth score and quantify what it says about whole blocks."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_nov13_outcomes import anchor, audit, bound, old
from audit_public_denominator import hits


def main():
    path = Path("upload/ap_20261113_05_ampliada.csv")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    frozen = json.loads(Path("ap_20261113_05_ampliada_report.json").read_text())
    assert digest == frozen["candidate_sha256"]
    vector = pd.read_csv(path).set_index("ID").condition
    assert vector.index.equals(anchor.index) and vector.isin(["new", "used"]).all()
    mask = frozenset(vector.index[vector.ne(anchor)])
    assert len(mask) == frozen["flips"] == 268
    ref = pd.read_csv(old / "line3_today/test_probs.csv").set_index("ID")
    assert ref.index.equals(anchor.index)
    same = frozenset(ref.index[ref.m1.ge(.5).eq(anchor.eq("used"))])
    history_path = Path("ap_history_nov13/constraints_private.json")
    past = [(name, frozenset(ids), gain) for name, ids, gain in json.loads(history_path.read_text())]
    assert not mask & set().union(*(r[1] for r in past[:9]))
    public_hits = hits(94361, audit.N_PUBLIC)
    delta = public_hits - audit.ANCHOR_HITS
    current = ("nov13_fifth", mask, delta)
    _, feasibility = audit.grouped_certificate([*past, current], feasibility_only=True)
    assert feasibility["status"] == "feasible"
    partitions = {
        "line3_agrees_with_s4": mask & same,
        "line3_differs_from_s4": mask - same,
        "previously_touched": mask & set().union(*(r[1] for r in past)),
    }
    levels = [94790, 94838, 94871, 94876, 95095, 95485, 95552]
    report = {
        "score_source": "Ariel reported the latest score; attachment matches the candidate SHA. Kaggle submission ID pending.",
        "file_sha256": digest, "historical_inputs_sha256": hashlib.sha256(history_path.read_bytes()).hexdigest(),
        "public_n_assumed": audit.N_PUBLIC, "score": .94361, "hits": public_hits,
        "anchor": .94766, "delta_hits": delta, "changes": len(mask),
        "total_disjoint_changes": sum(len(v) for _, v, _ in past[:9]) + len(mask),
        "total_disjoint_net": sum(g for _, _, g in past[:9]) + delta,
        "top_reported_score_net_gaps": {
            f"{s/100000:.5f}": hits(s, audit.N_PUBLIC) - audit.ANCHOR_HITS for s in levels},
        "subgroups": {name: {"changes": len(ids), "before": bound(ids, past), "after": bound(ids, [*past, current])}
                      for name, ids in partitions.items()},
        "scope": "Linear relaxation for whole subsets. A feasible system does not verify provenance or individual labels.",
    }
    Path("fifth_observation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
