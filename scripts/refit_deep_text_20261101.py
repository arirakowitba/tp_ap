"""Full 70k refit from frozen test-like tune settings; no public score used."""
import hashlib
import os
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.revision_submissions.deep_catboost_nbsvm_ablation_20261029 import extract_deep_features
from scripts.train_text_condition_20261101 import ZIP, OUT
from scripts.train_deep_text_20261101 import DROP_AGG

ITERATIONS=1246
THRESHOLD=.545
SEED=20261109


def text_array(rows):
    return np.array([str(r.get("title") or "") + " " + str(r.get("warranty") or "") +
                     " " + " ".join(str(v.get("value_name") or "") for v in r.get("attributes") or [])
                     for r in rows], dtype=object)


def vec():
    return TfidfVectorizer(analyzer="char",ngram_range=(3,5),min_df=3,
                           max_features=180000,sublinear_tf=True,dtype=np.float32)


def main():
    with zipfile.ZipFile(ZIP) as z:
        train=[json.loads(line) for line in z.open("train_data.jsonlines")]
        test=[json.loads(line) for line in z.open("test_data.jsonlines")]
    y=np.array([int(r["condition"]=="used") for r in train])
    sellers=np.array([str(r["seller_id"]) for r in train])
    ids=np.array([str(r["id"]) for r in train])
    assert len(train)==70000 and len(test)==30000 and len(set(ids))==70000
    ftr,ctr,*_=extract_deep_features(train)
    fte,cte,*_=extract_deep_features(test)
    keep=[c for c in ftr if not c.startswith(DROP_AGG)]
    base_tr=ftr[keep].copy()
    base_te=fte[keep].copy()
    for c in ("seller_id","category_id","city"):
        cnt=ctr[c].value_counts()
        base_tr["fit_freq_"+c]=ctr[c].map(cnt).fillna(1).astype(np.float32).values
        base_te["fit_freq_"+c]=cte[c].map(cnt).fillna(1).astype(np.float32).values
    med=base_tr.groupby(ctr.category_id).log_price.median()
    global_med=base_tr.log_price.median()
    base_tr["fit_rel_price_cat"]=base_tr.log_price-ctr.category_id.map(med).fillna(global_med)
    base_te["fit_rel_price_cat"]=base_te.log_price-cte.category_id.map(med).fillna(global_med)
    base_tr["seller_id_cat"]=ctr.seller_id
    base_te["seller_id_cat"]=cte.seller_id
    cats=["category_id","listing_type_id","buying_mode","state","ship_mode","seller_id_cat"]
    texts_tr=text_array(train);texts_te=text_array(test)
    oof=np.empty(len(train),np.float32)
    for tr,va in GroupKFold(n_splits=4).split(np.arange(len(y)),y,sellers):
        v=vec();x=v.fit_transform(texts_tr[tr])
        m=LogisticRegression(C=2,max_iter=180,solver="liblinear")
        m.fit(x,y[tr]);oof[va]=m.predict_proba(v.transform(texts_tr[va]))[:,1]
    v=vec();x=v.fit_transform(texts_tr)
    m=LogisticRegression(C=2,max_iter=180,solver="liblinear")
    m.fit(x,y)
    p_text_te=m.predict_proba(v.transform(texts_te))[:,1]
    base_tr["text_oof_p"]=oof
    base_te["text_oof_p"]=p_text_te
    model=CatBoostClassifier(iterations=ITERATIONS,depth=6,learning_rate=.05,
                             l2_leaf_reg=5,loss_function="Logloss",verbose=False,
                             thread_count=7,random_seed=SEED)
    model.fit(base_tr,y,cat_features=cats)
    prob=model.predict_proba(base_te)[:,1]
    pred=np.where(prob>=THRESHOLD,"used","new")
    out=OUT/"candidate_deep_text_full70k_20261101.csv"
    pd.DataFrame({"ID":np.arange(1,len(test)+1),"condition":pred}).to_csv(out,index=False)
    probs=OUT/"candidate_deep_text_full70k_prob.csv"
    pd.DataFrame({"ID":np.arange(1,len(test)+1),"item_id":[r["id"] for r in test],
                  "seller_id":[r["seller_id"] for r in test],"p_used":prob,
                  "p_text":p_text_te}).to_csv(probs,index=False)
    baseline=Path(os.environ.get("AP_BASELINE", str(Path(__file__).resolve().parents[1]/"submissions/candidate_d1_d2_combined_on48.csv")))
    flips=None
    if baseline.exists():
        ref=pd.read_csv(baseline).sort_values("ID")
        assert np.array_equal(ref.ID.values,np.arange(1,30001))
        flips=(ref.condition.to_numpy()!=pred)
    manifest={"train_zip_sha":hashlib.sha256(ZIP.read_bytes()).hexdigest(),
              "iterations":ITERATIONS,"threshold":THRESHOLD,"seed":SEED,
              "candidate_sha":hashlib.sha256(out.read_bytes()).hexdigest(),
              "prob_sha":hashlib.sha256(probs.read_bytes()).hexdigest(),
              "n_rows":len(test),"n_unique_test_item_id":len({r["id"] for r in test}),
              "flips_vs_D1D2":int(flips.sum()) if flips is not None else None,"test_seen_fraction":float(np.isin([str(r["seller_id"]) for r in test],sellers).mean())}
    (OUT/"refit_manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=="__main__":
    main()
