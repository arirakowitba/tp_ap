"""Seller-group paired CatBoost with native publication categories."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from confirm_structural_models import split as cold_split
from validate_recent_seller_history import split as warm_split
from validate_variant_inventory import ids
from validate_structural_disjoint import (
    METADATA, PARAMS, PICTURES, StratifiedGroupKFold, ag, causal_features, load,
)
from evaluate_structural_blends import sha, score


CATEGORICAL = ("category_id", "brand_val", "model_val", "item_cond_attr",
               "listing_type_id", "state_id", "city_id", "ship_mode", "buying_mode", "currency_id")


def main():
    archive, old = map(Path, sys.argv[1:3])
    previous = list(map(Path,sys.argv[5:]))
    out = Path("late_native_categories"); out.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        train,x = load(z,"train")
        test,xt = load(z,"test")
    assert len(train)==70000 and len(test)==30000
    assert train.id.is_unique and test.id.is_unique and not set(train.id)&set(test.id)
    df=pd.concat([train,test],ignore_index=True)
    extra=np.vstack((x,xt))[:,sorted(METADATA+PICTURES)]
    tr,te=map(ag.precompute_item_intrinsic_arrays,(train,test))
    full=np.arange(len(train))
    test_index=np.arange(len(test))
    report={}
    for name in ("cold","warm"):
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
        def matrix(numeric, rows):
            out_frame = pd.DataFrame(np.column_stack((numeric, extra[rows])))
            out_frame.columns = out_frame.columns.astype(str)
            for col in CATEGORICAL:
                out_frame[col] = df[col].iloc[rows].astype(str).to_numpy()
            return out_frame

        x_train=matrix(m["X_m1_seen_causal_T"],fit)
        x_eval=matrix(m["X_m1_seen_causal_evals"][0],cache_val)
        model=CatBoostClassifier(iterations=900, depth=7, learning_rate=.06, l2_leaf_reg=6,
                                 random_seed=119, thread_count=5, verbose=False,
                                 allow_writing_files=False, cat_features=list(CATEGORICAL))
        model.fit(x_train,m["y_T"])
        model.save_model(str(out/f"{name}.cbm"))
        reference["p_native"]=model.predict_proba(x_eval)[:,1]
        path=out/f"{name}_pairs.csv"
        reference.to_csv(path,index=False)
        report[name]={"pairs_sha256":sha(path),"model_sha256":[sha(out/f"{name}.cbm")]}
        if name!="full":
            baseline=reference["attributes" if warm else "lgb"].to_numpy()
            report[name]["metrics"]={
                group:score(tr["y"][val][s], reference.p_native.to_numpy()[s],baseline[s],
                            tr["seller_id"][val][s],tr["category_id"][val][s])
                for group,s in (("all",np.ones(len(val),bool)),("fresh",fresh),("seen",seen),("new",~seen))}
        (out/"model_report.json").write_text(json.dumps(report,indent=2)+"\n")
        print(name,{g:m["net"] for g,m in report[name].get("metrics",{}).items()},flush=True)


if __name__=="__main__":
    main()
