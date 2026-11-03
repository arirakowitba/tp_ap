"""Screen an orthogonal ExtraTrees base with clean seller-group OOF features.

The candidate uses only 70k labelled training rows. Text probabilities and
categorical target encodings are OOF by seller inside fit. Threshold selection
is tune-only; confirm is joined by item_id to the existing strong CatBoost pair.
"""
import argparse, json, re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold


def split_groups(seller, seed):
    rng = np.random.default_rng(seed)
    unique = np.unique(seller); rng.shuffle(unique)
    n_new = len(unique) // 10
    new_tune, new_confirm = set(unique[:n_new]), set(unique[n_new:2*n_new])
    seen = ~np.isin(seller, list(new_tune | new_confirm))
    seen_rows = np.where(seen)[0]; rng.shuffle(seen_rows)
    n_seen = len(seen_rows) // 10
    tune = np.r_[np.flatnonzero(np.isin(seller, list(new_tune))), seen_rows[:n_seen]]
    confirm = np.r_[np.flatnonzero(np.isin(seller, list(new_confirm))), seen_rows[n_seen:2*n_seen]]
    fit = np.setdiff1d(np.arange(len(seller)), np.r_[tune, confirm])
    return fit, tune, confirm


def scalar_features(rows):
    def num(r, k, default=0.):
        try: return float(r.get(k) if r.get(k) is not None else default)
        except Exception: return default
    data=[]
    cats={k:[] for k in ("seller_id","category_id","listing_type_id","buying_mode","state","city","ship_mode")}
    texts=[]
    for r in rows:
        title=str(r.get("title") or ""); warranty=str(r.get("warranty") or "")
        attrs=r.get("attributes") or []
        aval=" ".join(str(a.get("value_name") or "") for a in attrs if isinstance(a,dict))
        text=f"{title} {warranty} {aval}".lower()
        initial=max(num(r,"initial_quantity"),0.); sold=max(num(r,"sold_quantity"),0.)
        available=max(num(r,"available_quantity"),0.); price=max(num(r,"price"),0.)
        shipping=r.get("shipping") or {}; pics=r.get("pictures") or []
        data.append([
            np.log1p(price), np.log1p(max(num(r,"base_price"),0.)), np.log1p(max(num(r,"original_price"),0.)),
            np.log1p(initial), np.log1p(sold), np.log1p(available), sold/max(initial,1.),
            len(title), len(title.split()), sum(c.isdigit() for c in title),
            sum(c.isupper() for c in title)/max(len(title),1), len(warranty), len(attrs), len(pics),
            len(r.get("variations") or []), len(r.get("tags") or []), len(r.get("descriptions") or []),
            float(bool(r.get("accepts_mercadopago"))), float(bool(r.get("automatic_relist"))),
            float(bool(shipping.get("free_shipping"))), float(bool(shipping.get("local_pick_up"))),
            float(bool(re.search(r"usad|segunda mano|reacond",text))),
            float(bool(re.search(r"nuev|sin uso|sellad",text))),
            float(bool(re.search(r"antig|vintage|colecci|retro",text))),
            float(bool(re.search(r"repuesto|reparar|no funciona",text))),
        ])
        addr=r.get("seller_address") or {}
        cats["seller_id"].append(str(r.get("seller_id") or ""))
        cats["category_id"].append(str(r.get("category_id") or ""))
        cats["listing_type_id"].append(str(r.get("listing_type_id") or ""))
        cats["buying_mode"].append(str(r.get("buying_mode") or ""))
        cats["state"].append(str((addr.get("state") or {}).get("id") or ""))
        cats["city"].append(str((addr.get("city") or {}).get("id") or ""))
        cats["ship_mode"].append(str(shipping.get("mode") or ""))
        texts.append(text)
    return np.asarray(data,np.float32), pd.DataFrame(cats), np.asarray(texts,dtype=object)


def clean_te(cat, y, seller, fit, tune, confirm, m=10.):
    out=np.zeros((len(y),2*len(cat.columns)),np.float32); gkf=GroupKFold(4)
    for j,col in enumerate(cat.columns):
        freq=cat.iloc[fit][col].value_counts()
        out[:,2*j]=cat[col].map(freq).fillna(0).to_numpy(np.float32)
        for tr,va in gkf.split(fit,y[fit],seller[fit]):
            trr, var=fit[tr],fit[va]; prior=float(y[trr].mean())
            s=pd.DataFrame({"c":cat.iloc[trr][col].to_numpy(),"y":y[trr]}).groupby("c").y.agg(["count","mean"])
            mp=(s["count"]*s["mean"]+m*prior)/(s["count"]+m)
            out[var,2*j+1]=cat.iloc[var][col].map(mp).fillna(prior)
        prior=float(y[fit].mean())
        s=pd.DataFrame({"c":cat.iloc[fit][col].to_numpy(),"y":y[fit]}).groupby("c").y.agg(["count","mean"])
        mp=(s["count"]*s["mean"]+m*prior)/(s["count"]+m)
        for ix in (tune,confirm): out[ix,2*j+1]=cat.iloc[ix][col].map(mp).fillna(prior)
    return out


def text_oof(text,y,seller,fit,tune,confirm,seed):
    word=HashingVectorizer(ngram_range=(1,2),n_features=2**18,alternate_sign=False)
    char=HashingVectorizer(analyzer="char_wb",ngram_range=(3,5),n_features=2**18,alternate_sign=False)
    X=sp.hstack([word.transform(text),char.transform(text)],format="csr")
    out=np.zeros(len(y),np.float32); gkf=GroupKFold(4)
    for tr,va in gkf.split(fit,y[fit],seller[fit]):
        model=SGDClassifier(loss="log_loss",alpha=2e-6,max_iter=40,tol=1e-4,random_state=seed)
        model.fit(X[fit[tr]],y[fit[tr]]); out[fit[va]]=model.predict_proba(X[fit[va]])[:,1]
    model=SGDClassifier(loss="log_loss",alpha=2e-6,max_iter=50,tol=1e-4,random_state=seed)
    model.fit(X[fit],y[fit])
    out[tune]=model.predict_proba(X[tune])[:,1]; out[confirm]=model.predict_proba(X[confirm])[:,1]
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("jsonl",type=Path); ap.add_argument("baseline",type=Path)
    ap.add_argument("--seed",type=int,default=20261143); ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args(); rows=[json.loads(x) for x in args.jsonl.open(encoding="utf-8")]
    y=np.asarray([int(r["condition"]=="used") for r in rows],np.int8)
    seller=np.asarray([str(r["seller_id"]) for r in rows]); ids=np.asarray([str(r["id"]) for r in rows])
    fit,tune,confirm=split_groups(seller,args.seed)
    dense,cat,text=scalar_features(rows); te=clean_te(cat,y,seller,fit,tune,confirm); tp=text_oof(text,y,seller,fit,tune,confirm,args.seed)
    X=np.c_[dense,te,tp]
    model=ExtraTreesClassifier(n_estimators=500,min_samples_leaf=2,max_features=.75,class_weight="balanced",n_jobs=-1,random_state=args.seed)
    model.fit(X[fit],y[fit]); ptu=model.predict_proba(X[tune])[:,1]; pco=model.predict_proba(X[confirm])[:,1]
    grid=np.arange(.40,.601,.005); th=float(max(grid,key=lambda t:accuracy_score(y[tune],ptu>=t))); pred=(pco>=th).astype(int)
    base=pd.read_csv(args.baseline); key="id" if "id" in base else "item_id"
    cand=pd.DataFrame({"id":ids[confirm],"seller_id":seller[confirm],"y_true":y[confirm],"p_extra":pco,"pred_extra":pred})
    pair=cand.merge(base[[key,"y_true","p_dual","pred_dual"]].rename(columns={key:"id","y_true":"y_base"}),on="id",validate="one_to_one")
    assert len(pair)==len(confirm) and np.array_equal(pair.y_true,pair.y_base)
    delta=(pair.pred_extra==pair.y_true).astype(int)-(pair.pred_dual==pair.y_true).astype(int); pair["delta"]=delta
    by=pair.groupby("seller_id").delta.sum().to_numpy(); rng=np.random.default_rng(args.seed)
    boot=by[rng.integers(0,len(by),(3000,len(by)))].sum(1)
    args.out.mkdir(parents=True,exist_ok=True); pair.to_csv(args.out/f"paired_{args.seed}.csv",index=False)
    report={"seed":args.seed,"sizes":[len(fit),len(tune),len(confirm)],"threshold":th,
            "tune_acc":float(accuracy_score(y[tune],ptu>=th)),"extra_acc":float(accuracy_score(y[confirm],pred)),
            "extra_logloss":float(log_loss(y[confirm],pco)),"strong_acc":float((pair.pred_dual==pair.y_true).mean()),
            "net_vs_strong":int(delta.sum()),"fixes":int((delta==1).sum()),"breaks":int((delta==-1).sum()),
            "disagreements":int((pair.pred_extra!=pair.pred_dual).sum()),"seller_ci95":np.quantile(boot,[.025,.975]).tolist()}
    (args.out/f"report_{args.seed}.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

if __name__=="__main__": main()
