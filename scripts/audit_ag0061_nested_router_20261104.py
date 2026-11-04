"""Reproducible development audit of AG0061 pairs; no Kaggle/test labels.
Base predictions are a reused development holdout, not nested base OOF.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

def model(c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=104))
def features(d):
    a=d.p_cb.to_numpy(); b=d.p_lgb.to_numpy()
    return np.column_stack([a,b,np.abs(a-.5),np.abs(b-.5),b-a,np.abs(b-.5)/(np.abs(a-.5)+1e-4)])
def pair(y,a,b,g):
    delta=(b==y).astype(int)-(a==y).astype(int)
    sums=pd.Series(delta).groupby(np.asarray(g)).sum().to_numpy()
    rng=np.random.default_rng(104)
    boot=np.array([sums[rng.integers(0,len(sums),len(sums))].sum() for _ in range(5000)])
    return {"base_hits":int((a==y).sum()),"candidate_hits":int((b==y).sum()),
            "fixes":int((delta==1).sum()),"breaks":int((delta==-1).sum()),
            "net":int(delta.sum()),"seller_ci95":np.quantile(boot,[.025,.975]).tolist()}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("pairs"); ap.add_argument("output"); args=ap.parse_args()
    d=pd.read_csv(args.pairs); y=d.y_true.to_numpy(int); g=d.seller_id.astype(str).to_numpy()
    assert len(d)==12264 and d.confirm_idx.is_unique
    assert set(np.unique(y))=={0,1}
    assert np.allclose(d.p_blend80,.8*d.p_cb+.2*d.p_lgb)
    base=(d.p_cb.to_numpy()>=.5).astype(int)
    challenger=(d.p_lgb.to_numpy()>=.5).astype(int)
    dis=base!=challenger
    dx=d.loc[dis].reset_index(drop=True)
    X=features(dx); target=(challenger[dis]==y[dis]).astype(int); groups=g[dis]
    oof=np.full(len(dx),np.nan); accept=np.zeros(len(dx),bool); folds=[]
    for fold,(tr,te) in enumerate(GroupKFold(5).split(X,target,groups)):
        assert not set(groups[tr]) & set(groups[te])
        # Select C and threshold solely from inner group-OOF predictions.
        candidates=[]
        for c in [.01,.1,1.]:
            inner=np.full(len(tr),np.nan)
            for itr,iva in GroupKFold(3).split(X[tr],target[tr],groups[tr]):
                m=model(c); m.fit(X[tr[itr]],target[tr[itr]])
                inner[iva]=m.predict_proba(X[tr[iva]])[:,1]
            assert np.isfinite(inner).all()
            for threshold in [.5,.55,.6,.65,.7,.8,1.01]:
                route=inner>=threshold
                net=int(np.where(target[tr][route]==1,1,-1).sum())
                # Tie-break towards fewer interventions then stronger regularization.
                candidates.append((net,-int(route.sum()),-c,threshold,c))
        best=max(candidates); _,_,_,threshold,c=best
        m=model(c); m.fit(X[tr],target[tr]); p=m.predict_proba(X[te])[:,1]
        oof[te]=p; accept[te]=p>=threshold
        folds.append({"fold":fold,"n_train":len(tr),"n_test":len(te),"C":c,
                      "threshold":threshold,"inner_net":best[0],"outer_routed":int(accept[te].sum()),
                      "outer_net":int(np.where(target[te][accept[te]]==1,1,-1).sum())})
    assert np.isfinite(oof).all()
    routed=base.copy(); dis_idx=np.flatnonzero(dis)
    routed[dis_idx[accept]]=challenger[dis_idx[accept]]
    result={"source_sha256":hashlib.sha256(Path(args.pairs).read_bytes()).hexdigest(),
        "n":len(d),"n_sellers":len(np.unique(g)),"n_disagreements":int(dis.sum()),
        "warning":"Reused base development holdout; outer router folds do not remove upstream base-model dependence.",
        "common_threshold_050":pair(y,base,(d.p_blend80.to_numpy()>=.5).astype(int),g),
        "saved_tune_selected":pair(y,d.pred_cb.to_numpy(),d.pred_blend80.to_numpy(),g),
        "nested_router":pair(y,base,routed,g),
        "router_auc":float(roc_auc_score(target,oof)),
        "router_logloss":float(log_loss(target,oof)),
        "router_folds":folds,
        "threshold_table":[{"threshold":t,"cb_hits":int(((d.p_cb.to_numpy()>=t)==y).sum()),
              "blend_hits":int(((d.p_blend80.to_numpy()>=t)==y).sum())} for t in [.5,.53,.56]]}
    Path(args.output).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
