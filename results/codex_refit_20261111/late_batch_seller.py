"""Compare causal and batch seller histories on an interleaved train-only split."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from evaluate_structural_blends import score, sha
from validate_variant_inventory import ids
from validate_structural_disjoint import (
    METADATA, PARAMS, PICTURES, StratifiedGroupKFold, ag, causal_features, load,
)


def batch_features(df, fit, val):
    """Use fit labels only, subtracting the fit row itself from its statistics."""
    y = df.y.to_numpy()
    outputs = []
    for keys in (["seller_id"], ["seller_id", "category_id"], ["seller_id", "brand_val"]):
        table = df.iloc[fit].groupby(keys, sort=False).y.agg(["size", "sum"])
        if len(keys) == 1:
            index = df[keys[0]]
        else:
            index = pd.MultiIndex.from_frame(df[keys])
        counts, used = table.reindex(index).fillna(0).to_numpy().T
        counts[fit] -= 1
        used[fit] -= y[fit]
        assert np.all((used >= 0) & (counts >= used))
        outputs.extend((np.log1p(counts), (used+2.5)/(counts+5)))
    return np.column_stack(outputs).astype(np.float32)


def main():
    archive=Path(sys.argv[1])
    prior=ids(list(map(Path,sys.argv[5:])))
    with zipfile.ZipFile(archive) as z:
        df, extra = load(z,"train")
    rng = np.random.default_rng(28111322)
    sellers = np.sort(df.seller_id.unique())
    rng.shuffle(sellers)
    cold = df.seller_id.isin(sellers[:len(sellers)//6]).to_numpy()
    unique_times = df.groupby("seller_id").start_time.nunique()
    eligible = df.seller_id.map(unique_times).ge(3).to_numpy() & ~cold
    order = df.groupby("seller_id").start_time.rank(method="dense").to_numpy()
    upper = df.seller_id.map(unique_times).to_numpy()
    warm = eligible & (order>1) & (order<upper) & (rng.random(len(df))<.50)
    held = cold | warm
    fit,val = np.flatnonzero(~held),np.flatnonzero(held)
    assert not set(df.seller_id[cold]) & set(df.seller_id.iloc[fit])
    fit_times = df.iloc[fit].groupby("seller_id").start_time.agg(["min","max"])
    seen=warm[val]
    stamp=df.start_time.iloc[val[seen]].to_numpy()
    links=fit_times.reindex(df.seller_id.iloc[val[seen]])
    assert ((links["min"]<stamp)&(links["max"]>stamp)).all()
    fresh=~df.id.iloc[val].isin(prior).to_numpy()
    digest=hashlib.sha256(np.r_[fit,val].tobytes()).hexdigest()
    out=Path("late_batch_seller");out.mkdir(exist_ok=True)
    print("fit / validation / seen / fresh",len(fit),len(val),int(seen.sum()),int(fresh.sum()),flush=True)
    tr=ag.precompute_item_intrinsic_arrays(df)
    inner=list(StratifiedGroupKFold(4,shuffle=True,random_state=61111).split(
        fit,tr["y"][fit],tr["seller_id"][fit]))
    assert all(not set(tr["seller_id"][fit[a]])&set(tr["seller_id"][fit[b]]) for a,b in inner)
    ag.compute_causal_seller_features=causal_features([tr["start_time"][val]])
    m=ag.build_fold_matrices(fit,[val],inner,tr,[tr],out/f"text_{digest[:16]}.npz")
    print("text OOF ready",flush=True)
    attrs=extra[:,sorted(METADATA+PICTURES)]
    hist=batch_features(df,fit,val)
    x=np.column_stack((m["X_m1_seen_causal_T"],attrs[fit]))
    xv=np.column_stack((m["X_m1_seen_causal_evals"][0],attrs[val]))
    models={}
    for name,a,b in (("causal",x,xv),("batch",np.column_stack((x,hist[fit])),
                                    np.column_stack((xv,hist[val])))):
        model=LGBMClassifier(**PARAMS).fit(a,m["y_T"])
        model.booster_.save_model(str(out/f"{name}.txt"))
        models[name]=model.predict_proba(b)[:,1]
    pp=out/"pairs_private.csv"
    df.iloc[val][["id","y"]].assign(seen=seen,fresh=fresh,**models).to_csv(pp,index=False)
    p,base=models["batch"],models["causal"]
    y=tr["y"][val]
    cohorts={"all":np.ones(len(val),bool),"seen":seen,"new":~seen,
             "fresh":fresh,"fresh_seen":fresh&seen,"fresh_new":fresh&~seen}
    r={"scope":"Offline batch classification diagnostic using labels from fit only; includes later fit listings "
               "from the same seller. Cold sellers disjoint, warm holdout is interleaved. Full panel is development.",
       "fit_rows":len(fit),"validation_rows":len(val),"archive_sha256":sha(archive),"split_sha256":digest,
       "runner_sha256":sha(Path(__file__)),"pairs_sha256":sha(pp),
       "checkpoint_sha256":{n:sha(out/f"{n}.txt") for n in models},
       "disagreement_ids_sha256":hashlib.sha256("\n".join(sorted(df.id.iloc[val[(p>=.5)!=(base>=.5)]])).encode()).hexdigest(),
       "metrics":{name:score(y[s],p[s],base[s],tr["seller_id"][val][s],tr["category_id"][val][s],iterations=2500)
                  for name,s in cohorts.items()}}
    (out/"report.json").write_text(json.dumps(r,indent=2)+"\n")
    print({k:(v["n"],v["net"],v["seller_category_q05_q95"]) for k,v in r["metrics"].items()},flush=True)


if __name__=="__main__":
    main()
