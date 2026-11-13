"""Select a fifth candidate by a conservative conditional public-tail objective."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd


N_PUBLIC = 20998
NEEDED = 50
PENALTIES = (0., .05, .10, .15)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def simulator(df, mask, target, counts, seed, draws):
    """Seller Bayesian bootstrap and exact-size public sample, conditional on transfer."""
    rng = np.random.default_rng(seed)
    d = df.loc[mask]
    correct = d.y.to_numpy() == target[mask]
    seller, index = np.unique(d.seller, return_inverse=True)
    strata = 2*d.seen.to_numpy(dtype=int) + target[mask]
    good = np.zeros((len(seller),4))
    bad = good.copy()
    np.add.at(good, (index[correct], strata[correct]), 1)
    np.add.at(bad, (index[~correct], strata[~correct]), 1)
    public = rng.multivariate_hypergeometric(np.r_[counts, 30000-sum(counts)], N_PUBLIC, size=draws)[:,:4]
    prior = (correct.sum()+1)/(len(correct)+2)
    predictions = []
    for start in range(0,draws,512):
        end = min(start+512,draws)
        w = rng.exponential(size=(end-start,len(seller)))
        predictions.append(rng.beta(w@good+4*prior, w@bad+4*(1-prior)))
    p = np.vstack(predictions)
    result = {}
    for penalty in PENALTIES:
        delta = (2*rng.binomial(public, np.clip(p-penalty,0,1))-public).sum(axis=1)
        wins = int((delta >= NEEDED).sum())
        result[str(penalty)] = {"wins": wins, "draws":draws, "probability":wins/draws,
                                "mean_net":float(delta.mean()),
                                "net_quantiles":np.quantile(delta,[.05,.5,.95]).tolist()}
    return result, {"correct":good.sum(axis=0).astype(int).tolist(),"wrong":bad.sum(axis=0).astype(int).tolist(),
                    "test_counts":list(map(int,counts))}


def main():
    archive, old, ext, today, out = map(Path, sys.argv[1:])
    out.mkdir(exist_ok=True, parents=True)
    model = Path("fifth_schema")
    source = json.loads((model/"model_report.json").read_text())
    names = ["cold","warm","full"]
    assert set(source) == set(names)
    frames = {}
    for name in names:
        path = model/f"{name}_pairs.csv"
        assert digest(path)==source[name]["pairs_sha256"]
        frames[name]=pd.read_csv(path,dtype={"id":str,"item":str})
    cold,warm,test = [frames[x] for x in names]
    cold=cold.rename(columns={"lgb":"attr"})
    warm=warm.rename(columns={"attributes":"attr"})
    cold["base"]=cold.line3
    # Warm CX-0178 did not store Line3 probabilities. Its stronger local
    # LightGBM reference is used for the warm proxy, and cold uses Line3.
    warm["base"]=warm.attr
    test["base"]=test.m1
    q=pd.read_csv(ext/"morning_attribute_test.csv",dtype={"item":str})
    assert q.ID.equals(test.ID) and q.item.equals(test.item)
    test["attr"]=q.p_attributes
    anchor_path=old/"entrega_ap/S4_EVT154_respaldo.csv"
    anchor=pd.read_csv(anchor_path)
    assert test.ID.equals(anchor.ID) and anchor.ID.tolist()==list(range(1,30001))
    s=anchor.condition.eq("used").to_numpy()
    earlier=("ap_20261111_01_consenso_ruteado","ap_proxima_disjunta_104",
             "ap_quinta_catboost_disjunta","ap_probe_A_refit_exclusivo","ap_probe_B_ag_exclusivo")
    prior=[old/"entrega_ap"/f"{k}.csv" for k in earlier]
    prior.extend(next(today.glob(f"ap_20261113_{k}_*.csv")) for k in "ABCD")
    retired=np.zeros(len(anchor),bool)
    for path in prior:
        frame=pd.read_csv(path)
        assert frame.ID.equals(anchor.ID)
        retired |= frame.condition.ne(anchor.condition)
    assert retired.sum()==1514
    eligible=~retired & (test.base.ge(.5).to_numpy()==s)
    unique=set(cold.id)|set(warm.id)
    with zipfile.ZipFile(archive) as z,z.open("train_data.jsonlines") as f:
        sellers={str(r["id"]):str(r.get("seller_id") or "MISSING") for r in map(json.loads,f) if str(r["id"]) in unique}
    for df in (cold,warm):
        df["seller"]=df.id.map(sellers)
        assert df.seller.notna().all()
    fresh=warm.loc[warm.fresh].copy()
    assert not set(fresh.id)&set(cold.id)
    held_sellers=set(fresh.seller)|set(cold.seller)
    tune=warm.loc[~warm.fresh & ~warm.seller.isin(held_sellers) & ~warm.id.isin(cold.id)].copy()
    assert len(tune)>1000
    frozen=[]
    for weight in (.25,.5,.75,1.):
        for margin in (0.,.025,.05):
            pred=(1-weight)*test.attr.to_numpy()+weight*test.p_schema.to_numpy()
            target=pred>=.5
            selected=eligible&(target!=s)&(abs(pred-.5)>=margin)
            counts=np.bincount((2*test.seen.to_numpy(dtype=int)+target)[selected],minlength=4)
            if not selected.any():
                continue
            p=(1-weight)*tune.attr.to_numpy()+weight*tune.p_schema.to_numpy()
            t=p>=.5
            diff=(t!=tune.base.ge(.5).to_numpy())&(abs(p-.5)>=margin)
            rr,support=simulator(tune,diff,t,counts,301,8000)
            objective=rr["0.1"]["probability"]
            frozen.append((objective,rr["0.1"]["mean_net"],weight,margin,counts,selected,target,rr,support))
    assert frozen, "No nonempty policy"
    obj,_,weight,margin,counts,selected,target,tune_result,tune_support=max(frozen,key=lambda x:x[:2])
    heldout={}
    for i,(name,df) in enumerate((("warm_exclusive",fresh),("cold",cold))):
        p=(1-weight)*df.attr.to_numpy()+weight*df.p_schema.to_numpy()
        target_val=p>=.5
        diff=(target_val!=df.base.ge(.5).to_numpy())&(abs(p-.5)>=margin)
        rr,support=simulator(df,diff,target_val,counts,1011+i,60000)
        heldout[name]={"scenarios":rr,"support":support,"proxy":"LightGBM attributes" if i==0 else "Line3 M1"}
    from scipy.stats import hypergeom
    optimistic=float(hypergeom.sf(NEEDED-1,30000,int(selected.sum()),N_PUBLIC))
    out_frame=anchor.assign(condition=np.where(selected,target,s))
    out_frame["condition"]=out_frame.condition.map({True:"used",False:"new"})
    assert out_frame.ID.equals(anchor.ID) and out_frame.condition.isin(("new","used")).all()
    ids=frozenset(out_frame.ID[selected])
    for file in sorted((old/"inputs").glob("block_vectors_*.json")):
        assert not any(ids==frozenset(x["flips"]) for x in json.loads(file.read_text()))
    path=out/"ap_20261113_05_modelo_esquema.csv"
    out_frame.to_csv(path,index=False,lineterminator="\n")
    assert pd.read_csv(path).equals(out_frame)
    report={"file":path.name,"sha256":digest(path),"anchor_sha256":digest(anchor_path),
            "source_model":source,"runner_sha256":digest(Path(__file__)),"flips":int(selected.sum()),
            "seen_flips":int((selected&test.seen).sum()),"prior_changed_ids_excluded":int(retired.sum()),
            "tune_rows":len(tune),"evaluated_grid":len(frozen),
            "grid":[{"schema_weight":x[2],"margin":x[3],"flips":int(x[5].sum()),
                     "p_tune_stress_10pp":x[0],"mean_tune_stress_10pp":x[1]} for x in frozen],
            "selected":{"schema_weight":weight,"margin":margin,"tune_stress_10pp_probability":obj},
            "optimistic_all_flips_correct_uniform_public":optimistic,"tune":tune_result,"tune_support":tune_support,"holdout":heldout,
            "assumptions":"Uniform public sampling without replacement, seller bootstrap, four pseudo-observations per stratum shrunk toward the pooled rate "
                          "within seen/direction cells, and hypothetical transport penalties. Train proxy differs "
                          "from S4. This is conditional sensitivity, not a calibrated public probability. Warm and "
                          "cold holdout IDs are disjoint; base fits and model development panels can overlap."}
    (out/"ap_20261113_05_reporte.json").write_text(json.dumps(report,indent=2)+"\n")
    print("Selected",report["selected"],"flips",report["flips"])
    for name,values in heldout.items():
        print(name,values)


if __name__=="__main__":
    main()
