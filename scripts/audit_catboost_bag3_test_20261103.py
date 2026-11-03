"""Aggregate-only audit of bag3 full70k probabilities against D1+D2 anchor.

Emits no test labels or item-level predictions to the repository. Test is unlabeled.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROBS = ROOT / "results" / "catboost_bag3_full70k" / "test_probabilities.csv"
OUT = ROOT / "results" / "catboost_bag3_full70k" / "audit_vs_d1d2.json"
ANCHOR = Path(os.environ["ANCHOR_CSV"])
ANCHOR_SHA = "57172f209ff5018d4ceef8c57172cc731828cc78eb9cb02dedccff4a51292502"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sha(ANCHOR) == ANCHOR_SHA
    p = pd.read_csv(PROBS)
    a = pd.read_csv(ANCHOR)
    assert len(p) == len(a) == 30000
    assert p.ID.tolist() == a.ID.tolist() == list(range(1, 30001))
    assert p.item_id.is_unique and a.ID.is_unique
    assert set(a.condition.unique()) == {"new", "used"}
    anchor = (a.condition == "used").to_numpy()
    seen = p.seller_seen.to_numpy().astype(bool)
    report = {"n_test": 30000, "n_seen": int(seen.sum()),
              "anchor_sha256": sha(ANCHOR), "probabilities_sha256": sha(PROBS),
              "used_is_one": True, "thresholds": {}}
    for threshold in (.45, .475, .5):
        bag = (p.p_bag3.to_numpy() >= threshold)
        single = (p.p_seed0.to_numpy() >= threshold)
        diff = bag != anchor
        bag_only = bag != single
        report["thresholds"][str(threshold)] = {
            "bag_vs_anchor": int(diff.sum()),
            "bag_vs_anchor_seen": int((diff & seen).sum()),
            "bag_vs_anchor_new": int((diff & ~seen).sum()),
            "bag_to_used": int((diff & bag).sum()),
            "bag_to_new": int((diff & ~bag).sum()),
            "bag_only_vs_single": int(bag_only.sum()),
            "bag_only_anchor_matched_single": int((bag_only & (anchor == single)).sum()),
            "bag_only_anchor_matched_bag": int((bag_only & (anchor == bag)).sum()),
        }
    report["caveat"] = "Unlabeled test audit only; no accuracy or expected public gain can be inferred from disagreement counts."
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
