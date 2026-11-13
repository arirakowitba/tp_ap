"""Recover hash-verified historical masks and certify whole-block gain bounds."""
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from audit_nov13_outcomes import OLDER, SCORES, anchor, audit, bound, mask, old, today

N = audit.N_PUBLIC
HITS = audit.ANCHOR_HITS


def chunks(root, name):
    return "".join(p.read_text() for p in sorted(root.glob(name + ".*")))


def normalized_hash(v):
    return hashlib.sha256("".join(
        f"{i},{'used' if used else 'new'}\n" for i, used in enumerate(v,1)).encode()).hexdigest()


def load_vector(content):
    v = pd.read_csv(io.StringIO(content)).set_index("ID").condition.sort_index()
    assert v.index.equals(anchor.index) and v.isin(("new","used")).all()
    return v.eq("used").to_numpy()


def reconstruct(root):
    s48 = load_vector(chunks(root,"s48"))
    d1 = load_vector(chunks(root,"d1d2"))
    early = {str(r["canonical_key"]):r for r in csv.DictReader((root/"unique").open())}
    probes = {r["key"]:r for r in json.loads(chunks(root,"probes"))["entries"] if "key" in r}
    catalog = json.loads(chunks(old/"inputs","ledger"))
    mapped = {}
    issues = Counter()
    a = anchor.eq("used").to_numpy()
    for event in catalog:
        path = str(event["vector_path"]).replace("\\","/")
        if not path.startswith("reconstructed_from_"):
            continue
        if event["public_score"] is None or event["is_alias"]:
            continue
        if path.startswith("reconstructed_from_48_"):
            key = path.removeprefix("reconstructed_from_48_")
            row = early.get(key)
            v = s48.copy()
            flips = json.loads(row["flipped_ids"]) if row is not None else None
        else:
            key = path.removeprefix("reconstructed_from_d1d2_")
            row = probes.get(key)
            v = d1.copy()
            flips = row.get("flipped_ids_vs_d1d2") if row is not None else None
        if flips is None:
            issues["missing_mask"] += 1
            continue
        assert len(flips)==len(set(flips))
        if flips:
            v[np.asarray(flips,dtype=int)-1] ^= True
        if normalized_hash(v) != event["normalized_vector_sha256"]:
            issues["hash_mismatch"] += 1
            continue
        mapped[event["event_id"]] = (event["public_score"],frozenset(anchor.index[v!=a]))
    return mapped, catalog, issues


def main():
    root=Path("ap_history_nov13")
    reconstructed,catalog,issues=reconstruct(root)
    history={}
    for path in sorted((old/"inputs").glob("block_vectors_*.json")):
        for r in json.loads(path.read_text()):
            history[r["id"]]=(r["score"],frozenset(r["flips"]))
    extra=len(set(reconstructed)-set(history))
    for key,value in reconstructed.items():
        if key in history and history[key]!=value:
            issues["same_event_vector_conflict"]+=1
            history.pop(key)
        else:
            history[key]=value
    event = next(r for r in json.loads((root/"comprehensive").read_text()) if r["event_id"]=="EVT-151")
    raw = chunks(root,"evt151")
    assert hashlib.sha256(raw.encode()).hexdigest() == event["sha256_lf"]
    history["EVT-151"] = (event["public_score"],frozenset(anchor.index[load_vector(raw)!=anchor.eq("used").to_numpy()]))
    confirmed=[(name,mask(old/"entrega_ap"/f"{name}.csv"),audit.hits(score)-HITS)
               for name,score in OLDER.items()]
    confirmed += [(k,mask(next(today.glob(f"ap_20261113_{k}_*.csv"))),audit.hits(score)-HITS)
                  for k,score in SCORES.items()]
    accepted=confirmed.copy()
    rejected={}
    duplicate=0
    for name,(score,ids) in sorted(history.items(),reverse=True):
        try:
            gain=audit.hits(score)-HITS
        except ValueError:
            rejected[name]="incompatible precision"
            continue
        existing=[d for _,v,d in accepted if ids==v]
        if existing:
            if gain not in existing:
                rejected[name]="same vector different score"
            else:
                duplicate+=1
            continue
        if any(abs(gain-d)>len(ids^v) for _,v,d in accepted):
            rejected[name]="pair distance contradiction"
            continue
        if audit.grouped_certificate([*accepted,(name,ids,gain)],feasibility_only=True)[0] is None:
            rejected[name]="joint aggregate contradiction"
            continue
        accepted.append((name,ids,gain))
    coverage=set().union(*(ids for _,ids,_ in accepted))
    pair_bounds=audit.pair_bounds([*accepted,("S4",frozenset(),0)])
    chosen,certificate=audit.grouped_certificate(accepted)
    stats={"catalog_rows":len(catalog),"catalog_with_score":sum(r["public_score"] is not None for r in catalog),
           "catalog_category_counts":dict(Counter(r["category"] for r in catalog)),
           "reconstructed_hash_verified":len(reconstructed),"new_vectors_vs_previous_audit":extra,
           "issues":dict(issues),"duplicate_vectors":duplicate,"accepted":len(accepted),
           "covered_ids":len(coverage),"uncovered_ids":30000-len(coverage),"rejected":rejected,
           "positive_pair_bounds":len(pair_bounds),
           "max_pair_lower_bound":max((r[1] for r in pair_bounds),default=0),
           "aggregate_certificate":certificate}
    root.joinpath("constraints_private.json").write_text(json.dumps([[e,sorted(v),d] for e,v,d in accepted]))
    if chosen and certificate["lower_hits"]>0:
        root.joinpath("whole_block_private.json").write_text(json.dumps(sorted(chosen)))
    root.joinpath("aggregate_report.json").write_text(json.dumps(stats,indent=2)+"\n")
    print(json.dumps(stats,indent=2))


if __name__=="__main__":
    main()
