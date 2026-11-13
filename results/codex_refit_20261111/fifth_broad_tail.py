"""Exact public-tail sensitivity for broad, previously untested fifth-trial masks."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import binom, hypergeom

N, PUBLIC, REQUIRED = 30000, 20998, 50


def tail(probabilities):
    fixes = np.array([1.])
    for p in probabilities:
        fixes = np.convolve(fixes, [1-p,p])
    k = len(probabilities)
    if k < REQUIRED:
        return 0.
    public = np.arange(REQUIRED,k+1)
    need = (public+REQUIRED+1)//2
    t = hypergeom.sf(need[None,:]-1,k,np.arange(k+1)[:,None],public[None,:])
    return float(fixes @ (t @ hypergeom.pmf(public,N,k,PUBLIC)))


def main():
    old,today,out = map(Path,sys.argv[1:])
    a=pd.read_csv(old/"entrega_ap/S4_EVT154_respaldo.csv")
    full=pd.read_csv("fifth_schema/full_pairs.csv")
    assert a.ID.equals(full.ID) and len(a)==N
    s=a.condition.eq("used").to_numpy()
    prior_names=("ap_20261111_01_consenso_ruteado","ap_proxima_disjunta_104",
                 "ap_quinta_catboost_disjunta","ap_probe_A_refit_exclusivo","ap_probe_B_ag_exclusivo")
    prior=[old/"entrega_ap"/f"{name}.csv" for name in prior_names]
    prior.extend(next(today.glob(f"ap_20261113_{k}_*.csv")) for k in "ABCD")

    def read_mask(path):
        f=pd.read_csv(path)
        assert f.ID.equals(a.ID) and f.condition.isin(["used","new"]).all()
        return f.condition.ne(a.condition).to_numpy()

    retired=np.logical_or.reduce([read_mask(p) for p in prior])
    assert retired.sum()==1514
    schema=(full.p_schema.ge(.5).to_numpy()!=s)&~retired
    proxy=full.m1.ge(.5).to_numpy()==s
    cross=read_mask(today/"ap_20261113_05_consenso_cruzado.csv")
    r=read_mask(today/"ap_20261113_R_consenso_fuera_proxy.csv")
    assert not ((cross|r)&retired).any()
    policies={"schema_proxy":schema&proxy,"schema_broad":schema,"cross":cross,
              "r":r,"schema_cross":schema|cross,"schema_cross_r":schema|cross|r}
    cold=pd.read_csv("fifth_schema/cold_pairs.csv")
    warm=pd.read_csv("fifth_schema/warm_pairs.csv")
    assert not set(cold.id)&set(warm.id[warm.fresh])
    cold_y=cold.p_schema.ge(.5)
    warm_y=warm.p_schema.ge(.5)
    vm=cold_y.ne(cold.line3.ge(.5))
    wm=warm.fresh & warm_y.ne(warm.attributes.ge(.5))
    def estimate(pred, y, mask):
        return (int((pred[mask]==y[mask]).sum())+1)/(int(mask.sum())+2)
    cv=(estimate(cold_y,cold.y,vm),estimate(warm_y,warm.y,wm))
    tested=set()
    for file in sorted((old/"inputs").glob("block_vectors_*.json")):
        tested.update(frozenset(x["flips"]) for x in json.loads(file.read_text()))
    report={}
    for name,m in policies.items():
        assert frozenset(a.ID[m]) not in tested and not (m&retired).any()
        informed=(m&schema&proxy)[m]
        by_panel={}
        for panel,p in zip(("cold","warm_exclusive"),cv):
            by_panel[panel]={}
            for unvalidated in (.45,.5,.55,.6):
                q=np.where(informed,p,unvalidated)
                by_panel[panel][str(unvalidated)]={"p":tail(q),"mean_net":float((2*q-1).sum()*PUBLIC/N)}
        report[name]={"flips":int(m.sum()),"proxy_schema_flips":int(informed.sum()),
                      "scenarios":by_panel,
                      "all_correct_ceiling":float(hypergeom.sf(REQUIRED-1,N,int(m.sum()),PUBLIC))}
    winner=max(policies,key=lambda n:min(v["0.45"]["p"] for v in report[n]["scenarios"].values()))
    m=policies[winner]
    path=out/"ap_20261113_05_final.csv"
    final=a.assign(condition=np.where(s^m,"used","new"))
    final.to_csv(path,index=False,lineterminator="\n")
    assert pd.read_csv(path).equals(final)
    result={"selected":winner,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "scope":"Sensitivity only. Fixed independent flip accuracy by evidence group, uniformly sampled public set. "
            "Replayed proxies differ from S4. Untested off-proxy accuracy is assumed, not estimated. "
            "The worst of two panel transfer estimates at assumed 45% on unvalidated rows ranks the six policies. "
            "The negative submissions informed block exclusion; their scores cannot establish individual correctness.",
            "cv_proxy":{"cold_n":int(vm.sum()),"warm_n":int(wm.sum()),"precision":cv},
            "models":json.loads(Path("fifth_schema/model_report.json").read_text()),
            "candidates":report}
    (out/"ap_20261113_05_final_report.json").write_text(json.dumps(result,indent=2)+"\n")
    print(winner,json.dumps(report[winner]),flush=True)


if __name__=="__main__":
    main()
