"""Development-only outer-group calibration audit; upstream holdout is reused."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, roc_auc_score

def logit(p):
    p=np.clip(np.asarray(p),1e-6,1-1e-6)
    return np.log(p/(1-p))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("pairs"); ap.add_argument("output"); args=ap.parse_args()
    d=pd.read_csv(args.pairs)
    assert d.confirm_idx.is_unique and len(d)==12264
    y=d.y_true.to_numpy(int); g=d.seller_id.astype(str).to_numpy()
    assert np.isfinite(d[["p_cb","p_lgb","p_blend80"]]).all().all()
    cb=d.p_cb.to_numpy(); bl=d.p_blend80.to_numpy()
    designs={"platt_cb":logit(cb)[:,None],"platt_blend":logit(bl)[:,None],
             "stack_cb_lgb":np.column_stack([logit(cb),logit(d.p_lgb)])}
    preds={"raw_cb":cb,"raw_blend":bl}; fold_ids=np.full(len(d),-1)
    for name,X in designs.items():
        p=np.full(len(d),np.nan)
        for fold,(tr,te) in enumerate(GroupKFold(5).split(X,y,g)):
            assert not set(g[tr]) & set(g[te])
            m=make_pipeline(StandardScaler(),LogisticRegression(C=1.,solver="lbfgs",max_iter=2000,random_state=104))
            m.fit(X[tr],y[tr]); p[te]=m.predict_proba(X[te])[:,1]; fold_ids[te]=fold
        assert np.isfinite(p).all()
        preds[name]=p
    base=(cb>=.5)==y
    rng=np.random.default_rng(104); unique,inv=np.unique(g,return_inverse=True); n=len(unique)
    # Same seller resamples for every model and metric.
    sampled=rng.integers(0,n,(3000,n))
    base_ll=-(y*np.log(np.clip(cb,1e-12,1))+(1-y)*np.log(np.clip(1-cb,1e-12,1)))
    results={}
    for name,p in preds.items():
        correct=(p>=.5)==y; delta=correct.astype(int)-base.astype(int)
        ll=-(y*np.log(np.clip(p,1e-12,1))+(1-y)*np.log(np.clip(1-p,1e-12,1)))
        hit_by=np.bincount(inv,weights=delta,minlength=n)
        ll_by=np.bincount(inv,weights=ll-base_ll,minlength=n)
        counts=np.bincount(inv,minlength=n)
        boot_hit=hit_by[sampled].sum(axis=1)
        boot_ll=ll_by[sampled].sum(axis=1)/counts[sampled].sum(axis=1)
        results[name]={"hits":int(correct.sum()),"accuracy":float(correct.mean()),"logloss":float(log_loss(y,p)),
          "auc":float(roc_auc_score(y,p)),"fixes":int((delta==1).sum()),"breaks":int((delta==-1).sum()),
          "net_vs_cb":int(delta.sum()),"net_ci95":np.quantile(boot_hit,[.025,.975]).tolist(),
          "logloss_delta_vs_cb":float((ll-base_ll).mean()),"logloss_delta_ci95":np.quantile(boot_ll,[.025,.975]).tolist(),
          "fold_nets":[int(delta[fold_ids==f].sum()) for f in range(5)]}
    out={"source_sha256":hashlib.sha256(Path(args.pairs).read_bytes()).hexdigest(),
      "n":len(d),"sellers":n,"method":"5 outer GroupKFold, fixed LR C=1, threshold=.5, no threshold search",
      "limitation":"Development only. Upstream base models and this confirm have been repeatedly used; these folds do not make end-to-end nested OOF.",
      "results":results}
    Path(args.output).write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2))
if __name__=="__main__": main()
