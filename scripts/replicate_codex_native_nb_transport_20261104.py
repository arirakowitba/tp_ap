"""Paired CatBoost native text / nested NB text on original70k.
All vocabulary/supervised text fitting restricted to outer fit.
"""
import argparse,gc,hashlib,json,time,zipfile,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score,log_loss
from catboost import CatBoostClassifier
import lightgbm as lgb
try:
    from train_codex_raw_nested_text_histgb_20261104 import split,features,text_fit,digest,metrics
except ModuleNotFoundError:
    from train_raw_nested_text_histgb import split,features,text_fit,digest,metrics

def main():
    ap=argparse.ArgumentParser();ap.add_argument("zip");ap.add_argument("out");ap.add_argument("--seed",type=int,default=20261104);ap.add_argument("--pure",action="store_true");ap.add_argument("--deep",action="store_true");ap.add_argument("--replicate-only",action="store_true");args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);start=time.time()
    with zipfile.ZipFile(args.zip) as z: rows=[json.loads(l) for l in z.open("train_data.jsonlines")]
    y=np.asarray([int(r["condition"]=="used") for r in rows]);s=np.asarray([str(r["seller_id"]) for r in rows]);ids=np.asarray([str(r["id"]) for r in rows])
    assert len(rows)==70000 and len(set(ids))==70000
    fi,tu,co=split(s,args.seed,args.pure);print("SPLITS",len(fi),len(tu),len(co),flush=True)
    if args.deep:
        try:
            from src.revision_submissions.deep_catboost_nbsvm_ablation_20261029 import extract_deep_features
        except ModuleNotFoundError:
            from deep_reference import extract_deep_features
        af,cat,_,_,_=extract_deep_features(rows)
        drop=("freq_","seller_","sell_mean_","rel_price_cat")
        df=af[[c for c in af if not c.startswith(drop)]].copy()
        for col in ["seller_id","category_id","city"]:
            counts=cat.iloc[fi][col].value_counts()
            df["fit_freq_"+col]=cat[col].map(counts).fillna(0).astype(float)
        med=df.iloc[fi].groupby(cat.iloc[fi].category_id).log_price.median()
        df["fit_rel_price"]=df.log_price-cat.category_id.map(med).fillna(float(df.iloc[fi].log_price.median()))
        text=np.asarray([str(r.get("title") or "")+" WARRANTY "+str(r.get("warranty") or "")+" ATTR "+
                 " ".join(str(v.get("value_name") or v.get("name") or "") for v in r.get("attributes") or [] if isinstance(v,dict)) for r in rows],object)
    else:
        a,text,_,_=features(rows,fi,s)
        df=pd.DataFrame(a[:,:-7],columns=["n"+str(i) for i in range(a.shape[1]-7)])
    for key in ["category_id","listing_type_id","buying_mode","currency_id","status","seller_id"]:
        df[key]=[str(r.get(key) or "MISSING") for r in rows]
    df["state"]=[str(((r.get("seller_address") or {}).get("state") or {}).get("id") or "MISSING") for r in rows]
    df["ship_mode"]=[str((r.get("shipping") or {}).get("mode") or "MISSING") for r in rows]
    cats=["category_id","listing_type_id","buying_mode","currency_id","status","seller_id","state","ship_mode"]
    del rows;gc.collect()
    # Stateless hashing + strictly inner-fit supervised NB/LR.
    word=HashingVectorizer(n_features=2**16,alternate_sign=False,ngram_range=(1,2),norm="l2",dtype=np.float32)
    char=HashingVectorizer(n_features=2**16,alternate_sign=False,analyzer="char_wb",ngram_range=(3,5),norm="l2",dtype=np.float32)
    X=sparse.hstack([word.transform(text),char.transform(text)],format="csr");pt=np.full(len(y),np.nan)
    for f,(tr,va) in enumerate(GroupKFold(3).split(fi,y[fi],s[fi])):
        assert not set(s[fi[tr]])&set(s[fi[va]])
        pt[fi[va]]=text_fit(X,y,fi[tr],fi[va]);print("INNER_TEXT",f,time.time()-start,flush=True)
    pt[np.r_[tu,co]]=text_fit(X,y,fi,np.r_[tu,co]);del X;gc.collect()
    d_nb=df.assign(nested_nb_text=pt)
    d_native=df.assign(native_text=np.asarray([str(t).strip() or "empty" for t in text],object))
    probs={};tup={};metadata={}
    jobs=[("cb_tabular",df,[]),("cb_nested_nb",d_nb,[]),("cb_native_text",d_native,["native_text"])]
    if args.replicate_only:jobs=jobs[1:]
    for name,dd,txt in jobs:
        t0=time.time()
        m=CatBoostClassifier(iterations=900,depth=6,learning_rate=.06,l2_leaf_reg=6,random_seed=args.seed,
            thread_count=3,verbose=False,loss_function="Logloss",allow_writing_files=False)
        m.fit(dd.iloc[fi],y[fi],cat_features=cats,text_features=txt,
              eval_set=(dd.iloc[tu],y[tu]),early_stopping_rounds=90,use_best_model=True)
        tup[name]=m.predict_proba(dd.iloc[tu])[:,1];probs[name]=m.predict_proba(dd.iloc[co])[:,1]
        metadata[name]={"best_iteration":int(m.best_iteration_),"seconds":time.time()-t0}
        print("TRAINED",name,metadata[name],flush=True)
    ldf=d_nb.copy()
    for c in cats:ldf[c]=ldf[c].astype("category")
    m=lgb.LGBMClassifier(n_estimators=1000,num_leaves=45,max_depth=7,learning_rate=.05,subsample=.8,subsample_freq=1,colsample_bytree=.8,
          reg_lambda=5,random_state=args.seed,n_jobs=3,verbosity=-1)
    t0=time.time();m.fit(ldf.iloc[fi],y[fi],eval_set=[(ldf.iloc[tu],y[tu])],callbacks=[lgb.early_stopping(90,verbose=False)])
    tup["lgb_nested_nb"]=m.predict_proba(ldf.iloc[tu])[:,1];probs["lgb_nested_nb"]=m.predict_proba(ldf.iloc[co])[:,1]
    metadata["lgb_nested_nb"]={"best_iteration":int(m.best_iteration_),"seconds":time.time()-t0}
    for name,parts in {"blend_native_nb":(.5,"cb_native_text","cb_nested_nb"),
                       "blend_cb_lgb80":(.8,"cb_nested_nb","lgb_nested_nb")}.items():
        w,k1,k2=parts;tup[name]=w*tup[k1]+(1-w)*tup[k2];probs[name]=w*probs[k1]+(1-w)*probs[k2]
    # Candidate recipes predeclared above; freeze thresholds using tune only.
    thresholds={name:float(max(np.arange(.40,.601,.01),key=lambda t:accuracy_score(y[tu],p>=t))) for name,p in tup.items()}
    baseline_name="cb_nested_nb" if args.replicate_only else "cb_tabular"
    base=probs[baseline_name]>=.5;seen=np.isin(s[co],s[fi])
    res={name:metrics(y[co],p,base,s[co],seen) for name,p in probs.items()}
    tuned_base=probs[baseline_name]>=thresholds[baseline_name]
    tuned={}
    for name,p in probs.items():
        th=thresholds[name];delta=((p>=th)==y[co]).astype(int)-(tuned_base==y[co]).astype(int)
        tuned[name]={"threshold":th,"tune_accuracy":float(accuracy_score(y[tu],tup[name]>=th)),
                     "hits":int(((p>=th)==y[co]).sum()),"net_vs_tuned_base":int(delta.sum())}
    pairs=pd.DataFrame({"item_id":ids[co],"seller_id":s[co],"is_seen":seen.astype(int),"y_true":y[co],
                        **{"p_"+n:p for n,p in probs.items()}})
    pairs.to_csv(out/"pairs.csv",index=False)
    tune_pairs=pd.DataFrame({"item_id":ids[tu],"seller_id":s[tu],"y_true":y[tu],**{"p_"+n:p for n,p in tup.items()}})
    tune_pairs.to_csv(out/"tune_pairs.csv",index=False)
    report={"seed":args.seed,"baseline_name":baseline_name,"deep":args.deep,"pure":args.pure,"split_sizes":dict(zip(["fit","tune","confirm"],map(len,[fi,tu,co]))),
     "split_hashes":{n:digest(ids[ix]) for n,ix in [("fit",fi),("tune",tu),("confirm",co)]},
     "results_at_050":res,"tune_selected":tuned,
     "incremental_vs_nested_nb":{name:metrics(y[co],probs[name],probs["cb_nested_nb"]>=.5,s[co],seen)
          for name in ["cb_native_text","blend_native_nb","blend_cb_lgb80"]},
     "frozen_policy":{"blend_threshold":.51,"nb_threshold":.53,"blend_hits":int(((probs["blend_native_nb"]>=.51)==y[co]).sum()),"nb_hits":int(((probs["cb_nested_nb"]>=.53)==y[co]).sum())},
     "source_zip_sha256":hashlib.sha256(Path(args.zip).read_bytes()).hexdigest(),"training":metadata,"duration_seconds":time.time()-start,
     "pairs_sha256":hashlib.sha256((out/"pairs.csv").read_bytes()).hexdigest(),
     "limitation":"Reused development split. Native CatBoost text handled by library within fit. No end-to-end outer confirmation or public improvement demonstrated."}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()
