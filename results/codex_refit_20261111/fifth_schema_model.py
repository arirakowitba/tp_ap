"""Seller-group cross-fitted attribute-schema model for the fifth submission."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from confirm_structural_models import split as cold_split
from validate_recent_seller_history import split as warm_split
from validate_variant_inventory import ids
from validate_structural_disjoint import (
    METADATA, PARAMS, PICTURES, StratifiedGroupKFold, ag, causal_features, load,
)
from evaluate_structural_blends import sha, score


def schema_features(rows, df, fit, val, inner):
    groups = list(zip(df.category_id, df.listing_type_id))
    y = df.y.to_numpy()
    fields = ("attributes", "tags", "shipping", "layout")
    out = np.empty((len(df), 4*8), dtype=np.float32)

    def fill(train, query):
        prior = y[train].mean()
        for f, field in enumerate(fields):
            tables = [{}, {}]
            for i in train:
                keys = [(df.category_id.iat[i], token) for token in rows[i][field]]
                for token, key in zip(rows[i][field], keys):
                    for tbl, k in zip(tables, (token, key)):
                        stat = tbl.setdefault(k, [0, 0])
                        stat[0] += 1
                        stat[1] += y[i]
            for i in query:
                offset = 8*f
                tokens = rows[i][field]
                for j, table in enumerate(tables):
                    keys = tokens if j == 0 else [(groups[i][0], t) for t in tokens]
                    stats = np.asarray([table.get(k, (0, 0)) for k in keys] or [(0, 0)])
                    counts, used = stats.T
                    rates = (used+12*prior)/(counts+12)
                    out[i, offset+4*j:offset+4*j+4] = (
                        rates.min(), rates.max(), rates.mean(), np.log1p(counts.max()))

    for train, query in inner:
        fill(fit[train], fit[query])
    fill(fit, val)
    return out


def parse_schema(row):
    attrs = [str(a.get("id") or "") for a in row.get("attributes") or []]
    attrs = sorted(set(x for x in attrs if x and x != "ITEM_CONDITION"))
    shipping = row.get("shipping") or {}
    layout = tuple(str(int(bool(row.get(k)))) for k in (
        "warranty", "attributes", "variations", "pictures", "video_id", "descriptions",
        "original_price", "official_store_id", "differential_pricing", "coverage_areas"))
    return {
        "attributes": [*attrs, "schema="+"|".join(attrs)],
        "tags": sorted(set(row.get("tags") or [])),
        "shipping": sorted(set(shipping.get("tags") or [])),
        "layout": ["".join(layout), "".join(layout[:5]), "".join(layout[5:])],
    }


def main():
    archive, old = map(Path, sys.argv[1:3])
    previous = list(map(Path,sys.argv[5:]))
    out = Path("fifth_schema"); out.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        train,x = load(z,"train")
        test,xt = load(z,"test")
        rows=[]
        for kind in ("train","test"):
            with z.open(kind+"_data.jsonlines") as handle:
                rows.extend(parse_schema(json.loads(line)) for line in handle)
    assert len(train)==70000 and len(test)==30000 and len(rows)==100000
    assert train.id.is_unique and test.id.is_unique and not set(train.id)&set(test.id)
    df=pd.concat([train,test],ignore_index=True)
    extra=np.vstack((x,xt))[:,sorted(METADATA+PICTURES)]
    tr,te=map(ag.precompute_item_intrinsic_arrays,(train,test))
    full=np.arange(len(train))
    test_index=np.arange(len(test))
    report={}
    for name in ("cold","warm","full"):
        if name=="full":
            fit,val,arrays=full,test_index,te
            source=old/"line3_today"
            cache=source/"text_70000.npz"
            assert sha(source/"test_probs.csv")==json.loads((source/"manifest.json").read_text())["test_probs_sha256"]
            reference=pd.read_csv(source/"test_probs.csv",dtype={"item":str})
            assert np.array_equal(reference.item,test.id)
            seen=reference.seen.to_numpy(dtype=bool)
            cache_val=val+len(train)
        else:
            warm=name=="warm"
            source=Path("recent_seller_history" if warm else "structural_heldout_confirm")
            manifest=json.loads((source/"report.json").read_text())
            if warm:
                fit,val,seen,fresh=warm_split(train,ids(previous))
            else:
                fit,val,seen=cold_split(train,ids(previous[:-1]))
                fresh=np.ones(len(val),bool)
            digest=hashlib.sha256(np.r_[fit,val].tobytes()).hexdigest()
            assert digest==manifest["split_sha256"]
            cache=source/f"text_{digest[:16]}.npz"
            assert sha(source/"pairs_private.csv")==manifest["pairs_sha256"]
            reference=pd.read_csv(source/"pairs_private.csv",dtype={"id":str})
            assert np.array_equal(reference.id,train.id.iloc[val]) and np.array_equal(reference.y,train.y.iloc[val])
            arrays,cache_val=tr,val
        assert cache.exists()
        inner=list(StratifiedGroupKFold(4,shuffle=True,random_state=61111).split(fit,tr["y"][fit],tr["seller_id"][fit]))
        assert all(not set(tr["seller_id"][fit[a]])&set(tr["seller_id"][fit[b]]) for a,b in inner)
        ag.compute_causal_seller_features=causal_features([arrays["start_time"][val]])
        m=ag.build_fold_matrices(fit,[val],inner,tr,[arrays],cache)
        feat=schema_features(rows,df,fit,cache_val,inner)
        x_train=np.column_stack((m["X_m1_seen_causal_T"],extra[fit],feat[fit]))
        x_eval=np.column_stack((m["X_m1_seen_causal_evals"][0],extra[cache_val],feat[cache_val]))
        p=[]
        for seed in (119,271):
            model=LGBMClassifier(**(PARAMS|{"random_state":seed})).fit(x_train,m["y_T"])
            model.booster_.save_model(str(out/f"{name}_{seed}.txt"))
            p.append(model.predict_proba(x_eval)[:,1])
        reference["p_schema"]=np.mean(p,axis=0)
        path=out/f"{name}_pairs.csv"
        reference.to_csv(path,index=False)
        report[name]={"pairs_sha256":sha(path),"model_sha256":[sha(out/f"{name}_{seed}.txt") for seed in (119,271)]}
        if name!="full":
            baseline=reference["attributes" if warm else "lgb"].to_numpy()
            report[name]["metrics"]={
                group:score(tr["y"][val][s], reference.p_schema.to_numpy()[s],baseline[s],
                            tr["seller_id"][val][s],tr["category_id"][val][s])
                for group,s in (("all",np.ones(len(val),bool)),("fresh",fresh),("seen",seen),("new",~seen))}
        (out/"model_report.json").write_text(json.dumps(report,indent=2)+"\n")
        print(name,{g:m["net"] for g,m in report[name].get("metrics",{}).items()},flush=True)


if __name__=="__main__":
    main()
