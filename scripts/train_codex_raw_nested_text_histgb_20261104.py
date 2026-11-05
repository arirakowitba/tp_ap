"""Train-only raw-data experiment: rich HistGB and strictly nested hash NB text.
No leaderboard scores used. All confirm results are development due historical reuse.
"""
import argparse,hashlib,json,time,zipfile,gc
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score,log_loss

def digest(v): return hashlib.sha256("\n".join(map(str,v)).encode()).hexdigest()
def split(s,seed,pure=False):
    rng=np.random.default_rng(seed); u=np.unique(s); rng.shuffle(u)
    if pure:
        k=int(.15*len(u)); tu=np.flatnonzero(np.isin(s,u[:k]));co=np.flatnonzero(np.isin(s,u[k:2*k]));fi=np.flatnonzero(np.isin(s,u[2*k:]));return fi,tu,co
    k=int(.05*len(u))
    tu_new=set(u[:k]); co_new=set(u[k:2*k]); groups={}
    for i,x in enumerate(s): groups.setdefault(x,[]).append(i)
    eligible=[]
    for x in u[2*k:]:
        ix=np.asarray(groups[x]); rng.shuffle(ix); eligible.extend(ix[1:])
    eligible=np.asarray(eligible,int); rng.shuffle(eligible); n=min(9000,len(eligible)//2)
    tu=np.r_[np.flatnonzero(np.isin(s,list(tu_new))),eligible[:n]]
    co=np.r_[np.flatnonzero(np.isin(s,list(co_new))),eligible[n:2*n]]
    fi=np.setdiff1d(np.arange(len(s)),np.r_[tu,co])
    assert not (set(fi)&set(tu) or set(fi)&set(co) or set(tu)&set(co))
    return fi,tu,co
def num(x):
    try: return float(x) if x is not None else np.nan
    except (ValueError,TypeError): return np.nan
def features(rows,fit,sellers):
    cats=["category_id","listing_type_id","buying_mode","currency_id","status","state","shipping"]
    nums=[]; catdata=[]; text=[]
    for r in rows:
        title=str(r.get("title") or ""); warranty=str(r.get("warranty") or "")
        attrs=r.get("attributes") or []; shipping=r.get("shipping") or {}; address=r.get("seller_address") or {}
        price=num(r.get("price")); original=num(r.get("original_price"))
        qs=[num(r.get(k)) for k in ["initial_quantity","sold_quantity","available_quantity","price","base_price","original_price"]]
        qs=[np.log1p(max(0,q)) if np.isfinite(q) else np.nan for q in qs]
        lens=[len(r.get(k) or []) for k in ["attributes","pictures","variations","tags","non_mercado_pago_payment_methods","descriptions","deal_ids","sub_status"]]
        textvals=[str(a.get("value_name") or a.get("name") or "") for a in attrs if isinstance(a,dict)]
        text.append(title+" WARRANTY "+warranty+" ATTR "+" ".join(textvals))
        core=qs+lens+[len(title),len(title.split()),len(warranty),int(bool(warranty)),
             int(bool(r.get("accepts_mercadopago"))),int(bool(r.get("automatic_relist"))),
             int(r.get("official_store_id") is not None),int(bool(shipping.get("free_shipping"))),
             int(bool(shipping.get("local_pick_up"))),int(bool(r.get("video_id"))),
             int(bool(r.get("parent_item_id")))]
        times=[num(pd.to_datetime(r.get(k),errors="coerce",utc=True).timestamp()) if r.get(k) else np.nan for k in ["date_created","last_updated"]]
        # Invalid timestamps become nan; no labels or test statistics involved.
        times=[t if np.isfinite(t) else np.nan for t in times]
        nums.append(core+times+[(times[1]-times[0])/86400 if all(np.isfinite(times)) else np.nan,
                      original/price if np.isfinite(price) and price>0 else np.nan])
        catdata.append([str(r.get(k) or "MISSING") for k in cats[:5]]+
          [str((address.get("state") or {}).get("id") or "MISSING"),str(shipping.get("mode") or "MISSING")])
    a=np.asarray(nums,float); catdata=np.asarray(catdata,object)
    codes=[]
    for j in range(len(cats)):
        counts=Counter(catdata[fit,j]); mp={v:i+1 for i,(v,_) in enumerate(counts.most_common(180))}
        codes.append(np.asarray([mp.get(v,0) for v in catdata[:,j]],float))
    counts=Counter(sellers[fit]); seller_freq=np.asarray([np.log1p(counts.get(s,0)) for s in sellers])
    # Fit-only category price summaries.
    lp=a[:,3]; tmp=pd.DataFrame({"c":catdata[fit,0],"p":lp[fit]})
    med=tmp.groupby("c").p.median().to_dict(); globalmed=float(np.nanmedian(lp[fit]))
    relative=lp-np.asarray([med.get(c,globalmed) for c in catdata[:,0]])
    a=np.column_stack([a,seller_freq,relative]+codes)
    core_idx=list(range(27))+[a.shape[1]-len(cats)+i for i in range(len(cats))]
    return a,np.asarray(text,object),list(range(a.shape[1]-len(cats),a.shape[1])),core_idx
def text_fit(X,y,tr,va):
    # Supervised NB log-count ratio strictly within the corresponding fit fold.
    pos=np.asarray(X[tr][y[tr]==1].sum(axis=0)).ravel()+1
    neg=np.asarray(X[tr][y[tr]==0].sum(axis=0)).ravel()+1
    ratio=np.log((pos/pos.sum())/(neg/neg.sum()))
    m=LogisticRegression(C=2.,solver="liblinear",max_iter=300,random_state=104)
    m.fit(X[tr].multiply(ratio),y[tr])
    return m.predict_proba(X[va].multiply(ratio))[:,1]
def metrics(y,p,base,seller,seen):
    pred=p>=.5; delta=(pred==y).astype(int)-(base==y).astype(int)
    by=pd.Series(delta).groupby(seller).sum().to_numpy(); rng=np.random.default_rng(104)
    boot=np.asarray([by[rng.integers(0,len(by),len(by))].sum() for _ in range(3000)])
    return {"accuracy":float(accuracy_score(y,pred)),"logloss":float(log_loss(y,p)),
      "hits":int((pred==y).sum()),"net":int(delta.sum()),"fixes":int((delta==1).sum()),
      "breaks":int((delta==-1).sum()),"seller_ci95":np.quantile(boot,[.025,.975]).tolist(),
      "seen_new":{name:{"n":int(mask.sum()),"accuracy":float(accuracy_score(y[mask],pred[mask])) if mask.any() else None,
        "net":int(delta[mask].sum())} for name,mask in [("seen",seen),("new",~seen)]}}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("zip");ap.add_argument("out");ap.add_argument("--seed",type=int,default=20261104);ap.add_argument("--pure",action="store_true");args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True); t0=time.time()
    with zipfile.ZipFile(args.zip) as z: rows=[json.loads(l) for l in z.open("train_data.jsonlines")]
    assert len(rows)==70000
    y=np.asarray([int(r["condition"]=="used") for r in rows]); sellers=np.asarray([str(r["seller_id"]) for r in rows])
    ids=np.asarray([str(r["id"]) for r in rows]);assert len(set(ids))==len(ids)
    fi,tu,co=split(sellers,args.seed,args.pure); print("SPLITS",len(fi),len(tu),len(co),flush=True)
    a,text,cats,core_idx=features(rows,fi,sellers); del rows;gc.collect()
    print("FEATURES",a.shape,"elapsed",time.time()-t0,flush=True)
    # Stateless hashing avoids vocabulary fitting on validation.
    word=HashingVectorizer(n_features=2**16,alternate_sign=False,ngram_range=(1,2),norm="l2",dtype=np.float32)
    char=HashingVectorizer(n_features=2**16,alternate_sign=False,analyzer="char_wb",ngram_range=(3,5),norm="l2",dtype=np.float32)
    X=sparse.hstack([word.transform(text),char.transform(text)],format="csr")
    poof=np.full(len(y),np.nan)
    for fold,(tr,va) in enumerate(GroupKFold(3).split(fi,y[fi],sellers[fi])):
        assert not set(sellers[fi[tr]])&set(sellers[fi[va]])
        poof[fi[va]]=text_fit(X,y,fi[tr],fi[va]);print("TEXT_INNER",fold,time.time()-t0,flush=True)
    poof[np.r_[tu,co]]=text_fit(X,y,fi,np.r_[tu,co]);del X;gc.collect()
    print("TEXT_DONE",time.time()-t0,flush=True)
    aset={"core":(a[:,core_idx],[core_idx.index(i) for i in cats]),
          "rich":(a,cats),"rich_nested_text":(np.column_stack([a,poof]),cats)}
    probs={}; tuned={}; times={}
    for name,(xx,cc) in aset.items():
        start=time.time()
        m=HistGradientBoostingClassifier(max_iter=250,max_leaf_nodes=23,learning_rate=.08,l2_regularization=5,
             min_samples_leaf=30,categorical_features=cc,early_stopping=False,random_state=args.seed)
        m.fit(xx[fi],y[fi]); ptu=m.predict_proba(xx[tu])[:,1];pco=m.predict_proba(xx[co])[:,1]
        threshold=max(np.arange(.40,.601,.01),key=lambda th:accuracy_score(y[tu],ptu>=th))
        probs[name]=pco;tuned[name]={"threshold":float(threshold),"hits":int(((pco>=threshold)==y[co]).sum()),
           "tune_accuracy":float(accuracy_score(y[tu],ptu>=threshold))};times[name]=time.time()-start
        print("MODEL",name,"acc",accuracy_score(y[co],pco>=.5),"tuned",tuned[name],"seconds",times[name],flush=True)
    probs["nested_text_only"]=poof[co]
    seen=np.isin(sellers[co],sellers[fi]);base=probs["core"]>=.5
    results={name:metrics(y[co],p,base,sellers[co],seen) for name,p in probs.items()}
    pairs=pd.DataFrame({"item_id":ids[co],"seller_id":sellers[co],"is_seen":seen.astype(int),"y_true":y[co],
       **{"p_"+name:p for name,p in probs.items()}})
    pairs.to_csv(out/"pairs.csv",index=False)
    report={"seed":args.seed,"pure_seller_group":args.pure,"n_train":len(y),"split_sizes":dict(zip(["fit","tune","confirm"],map(len,[fi,tu,co]))),
      "split_id_hashes":{name:digest(ids[ix]) for name,ix in [("fit",fi),("tune",tu),("confirm",co)]},
      "source_zip_sha256":hashlib.sha256(Path(args.zip).read_bytes()).hexdigest(),
      "pair_sha256":hashlib.sha256((out/"pairs.csv").read_bytes()).hexdigest(),"results":results,"tune_selected":tuned,
      "model_duration_seconds":times,"duration_seconds":time.time()-t0,
      "limitation":"New reproducible development experiment; historical confirm overlap not yet audited. Compare official strong baseline on same IDs before promotion."}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()
