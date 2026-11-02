"""Complementary LightGBM on fold-safe text feature, tune-only blend weight."""
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.revision_submissions.deep_catboost_nbsvm_ablation_20261029 import extract_deep_features
from scripts.train_text_condition_20261101 import OUT, text_features
from scripts.train_deep_text_20261101 import read_raw, split_testlike, add_fit_aggregates, DROP_AGG

SEED=20261110


def main():
    rows=read_raw()
    y=np.array([int(r["condition"]=="used") for r in rows])
    sellers=np.array([str(r["seller_id"]) for r in rows])
    ids=np.array([str(r["id"]) for r in rows])
    fit,tune,confirm=split_testlike(sellers,SEED)
    features,cat,*_=extract_deep_features(rows)
    keep=[c for c in features if not c.startswith(DROP_AGG)]
    a=add_fit_aggregates(features[keep],cat,fit)
    cats=["category_id","listing_type_id","buying_mode","state","ship_mode"]
    a["seller_id_cat"]=cat.seller_id;cats.append("seller_id_cat")
    texts=np.array([str(r.get("title") or "")+" "+str(r.get("warranty") or "")+
                    " "+" ".join(str(v.get("value_name") or "") for v in r.get("attributes") or [])
                    for r in rows],dtype=object)
    oo,tu,co=text_features(texts,y,sellers,fit,tune,confirm)
    a["text_oof_p"]=np.nan
    a.loc[fit,"text_oof_p"]=oo;a.loc[tune,"text_oof_p"]=tu;a.loc[confirm,"text_oof_p"]=co
    cb=CatBoostClassifier(iterations=1250,depth=6,learning_rate=.05,l2_leaf_reg=5,
                          loss_function="Logloss",thread_count=7,random_seed=SEED,verbose=False)
    cb.fit(a.iloc[fit],y[fit],cat_features=cats,eval_set=(a.iloc[tune],y[tune]),
           early_stopping_rounds=100,use_best_model=True)
    p_cb_tu=cb.predict_proba(a.iloc[tune])[:,1]
    p_cb_co=cb.predict_proba(a.iloc[confirm])[:,1]
    b=a.copy()
    for c in cats:
        b[c]=pd.Categorical(b[c],categories=pd.Index(b[c].iloc[fit].unique())).codes.astype("int32")
    m=lgb.LGBMClassifier(n_estimators=1800,learning_rate=.035,num_leaves=15,
                         min_child_samples=70,colsample_bytree=.85,subsample=.85,
                         reg_lambda=8,verbosity=-1,random_state=SEED,n_jobs=7)
    m.fit(b.iloc[fit],y[fit],eval_set=[(b.iloc[tune],y[tune])],
          eval_metric="binary_logloss",categorical_feature=cats,
          callbacks=[lgb.early_stopping(100,verbose=False)])
    p_lgb_tu=m.predict_proba(b.iloc[tune])[:,1]
    p_lgb_co=m.predict_proba(b.iloc[confirm])[:,1]
    weights=np.arange(0,.501,.05)
    w=min(weights,key=lambda v:log_loss(y[tune],(1-v)*p_cb_tu+v*p_lgb_tu))
    bp_tu=(1-w)*p_cb_tu+w*p_lgb_tu
    bp_co=(1-w)*p_cb_co+w*p_lgb_co
    thresholds=np.arange(.44,.561,.005)
    th_cb=max(thresholds,key=lambda t:accuracy_score(y[tune],p_cb_tu>=t))
    th_bl=max(thresholds,key=lambda t:accuracy_score(y[tune],bp_tu>=t))
    pred_cb=p_cb_co>=th_cb;pred_bl=bp_co>=th_bl
    seen=np.isin(sellers[confirm],sellers[fit])
    delta=(pred_bl==y[confirm]).astype(int)-(pred_cb==y[confirm]).astype(int)
    paired=pd.DataFrame({"item_id":ids[confirm],"seller_id":sellers[confirm],
                         "is_seen":seen.astype(int),"y_true":y[confirm],"p_cb":p_cb_co,
                         "p_lgb":p_lgb_co,"p_blend":bp_co,"pred_cb":pred_cb.astype(int),
                         "pred_blend":pred_bl.astype(int)})
    out=OUT/f"paired_lgb_{SEED}.csv";paired.to_csv(out,index=False)
    by=paired.assign(delta=delta).groupby("seller_id").delta.sum().to_numpy()
    rng=np.random.default_rng(SEED);boot=by[rng.integers(0,len(by),(2000,len(by)))].sum(axis=1)
    report={"seed":SEED,"sizes":[len(fit),len(tune),len(confirm)],
            "hash_ids":{k:hashlib.sha256(ids[ix].astype("U").tobytes()).hexdigest()
                        for k,ix in (("fit",fit),("tune",tune),("confirm",confirm))},
            "best_iterations":[cb.best_iteration_,m.best_iteration_],
            "weight_lgb":float(w),"thresholds":[float(th_cb),float(th_bl)],
            "tune_accuracy":[accuracy_score(y[tune],p_cb_tu>=th_cb),accuracy_score(y[tune],bp_tu>=th_bl)],
            "confirm_accuracy":[accuracy_score(y[confirm],pred_cb),accuracy_score(y[confirm],pred_bl)],
            "confirm_logloss":[log_loss(y[confirm],p_cb_co),log_loss(y[confirm],bp_co)],
            "net":int(delta.sum()),"ci95_seller":np.quantile(boot,[.025,.975]).tolist(),
            "seen_new":{str(k):{"n":int(sum(seen==k)),"net":int(delta[seen==k].sum())} for k in (0,1)},
            "paired_sha":hashlib.sha256(out.read_bytes()).hexdigest()}
    (OUT/f"report_lgb_{SEED}.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
