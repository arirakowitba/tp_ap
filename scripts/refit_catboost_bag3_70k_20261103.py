"""Full 70k refit of fixed equal-weight three-seed CatBoost; analysis only.

No Kaggle submission is emitted. 100k original JSONL provide unlabelled feature
context; only train_data labels enter target encodings and model training.
"""
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.revision_submissions.deep_catboost_nbsvm_ablation_20261029 import extract_deep_features

ZIP=Path(os.environ["AP_ZIP"])
OUT=ROOT/"results"/"catboost_bag3_full70k"
OUT.mkdir(parents=True,exist_ok=True)
SEED=20261102
ITERATIONS=900  # predeclared, no test selection
MODEL_SEEDS=[20261102,20262099,20263096]

def main():
    with zipfile.ZipFile(ZIP) as z:
        train=[json.loads(line) for line in z.open("train_data.jsonlines")]
        test=[json.loads(line) for line in z.open("test_data.jsonlines")]
    assert len(train)==70000 and len(test)==30000
    assert all(r.get("condition") in ("new","used") for r in train)
    assert all("condition" not in r for r in test)
    rows=train+test
    y=np.array([r["condition"]=="used" for r in train],dtype=np.int8)
    a,cat,titles,warranties,combos=extract_deep_features(rows)
    n=len(train); idx=np.arange(n); tidx=np.arange(n,len(rows))
    seller=cat.seller_id.astype(str).to_numpy()
    prior=float(y.mean())
    inner=list(GroupKFold(4).split(idx,y,seller[:n]))
    for col in ("seller_id","category_id","city"):
        oof=np.full(n,prior,dtype=np.float32)
        keys=cat[col].astype(str)
        for tr,va in inner:
            s=pd.DataFrame({"k":keys.iloc[tr].to_numpy(),"y":y[tr]}).groupby("k").y.agg(["sum","count"])
            smooth=(s["sum"]+5*prior)/(s["count"]+5)
            oof[va]=keys.iloc[va].map(smooth).fillna(prior).to_numpy()
        s=pd.DataFrame({"k":keys.iloc[:n].to_numpy(),"y":y}).groupby("k").y.agg(["sum","count"])
        smooth=(s["sum"]+5*prior)/(s["count"]+5)
        a["te_"+col]=np.r_[oof,keys.iloc[n:].map(smooth).fillna(prior).to_numpy()]

    word=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=120000,
                         sublinear_tf=True,strip_accents="unicode",lowercase=True)
    char=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=5,
                         max_features=150000,sublinear_tf=True)
    xtr=sp.hstack([word.fit_transform(combos[:n]),char.fit_transform(titles[:n])],format="csr")
    xte=sp.hstack([word.transform(combos[n:]),char.transform(titles[n:])],format="csr")
    txt_oof=np.empty(n,dtype=np.float32)
    for tr,va in inner:
        lr=LogisticRegression(C=3.,max_iter=300,solver="liblinear",random_state=SEED)
        lr.fit(xtr[tr],y[tr])
        txt_oof[va]=lr.predict_proba(xtr[va])[:,1]
    lr=LogisticRegression(C=3.,max_iter=300,solver="liblinear",random_state=SEED)
    lr.fit(xtr,y)
    txt_test=lr.predict_proba(xte)[:,1]
    a["txt_sparse_p"]=np.r_[txt_oof,txt_test]
    a["seller_id_cat"]=cat.seller_id.astype(str)
    cats=["category_id","listing_type_id","buying_mode","state","ship_mode","seller_id_cat"]
    probs=[]
    for seed in MODEL_SEEDS:
        model=CatBoostClassifier(iterations=ITERATIONS,depth=6,learning_rate=.07,l2_leaf_reg=6.,
                                 random_seed=seed,thread_count=7,verbose=False,loss_function="Logloss")
        model.fit(a.iloc[:n],y,cat_features=cats)
        probs.append(model.predict_proba(a.iloc[n:])[:,1])
        print(f"finished full70k seed={seed}",flush=True)
    p=np.mean(probs,axis=0)
    result=pd.DataFrame({"ID":np.arange(1,30001),"item_id":[str(r["id"]) for r in test],
                         "seller_id":seller[n:],"seller_seen":np.isin(seller[n:],seller[:n]).astype(int),
                         "p_bag3":p,
                         "p_seed0":probs[0],"p_seed1":probs[1],"p_seed2":probs[2]})
    assert result.ID.is_unique and result.item_id.is_unique
    out=OUT/"test_probabilities.csv";result.to_csv(out,index=False)
    manifest={"train_zip_sha256":hashlib.sha256(ZIP.read_bytes()).hexdigest(),
              "train_rows":n,"test_rows":len(test),"used_is_one":True,
              "iterations":ITERATIONS,"model_seeds":MODEL_SEEDS,"weights":[1/3]*3,
              "features_count":len(a.columns),
              "text_inner_folds":"GroupKFold(4) by seller; vocabulary fit on all 70k train, no test text",
              "te_inner_folds":"GroupKFold(4) by seller",
              "unlabelled_features":"extract_deep_features on original train+test 100k",
              "test_probs_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
              "test_id_order_sha256":hashlib.sha256(result.item_id.astype("U").to_numpy().tobytes()).hexdigest()}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)

if __name__=="__main__":main()
