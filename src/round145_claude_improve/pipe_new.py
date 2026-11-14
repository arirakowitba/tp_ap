"""Round145 'new-seller' specialist: same B10 lgb_tpe_2 recipe, but every label-derived OOF feature for training rows is
computed with SELLER-grouped folds (so training rows look like unseen sellers, as new-seller test rows do), seller-identity
features are dropped (seller_id, seller_category, seller_listing and their target rates), plus round145 new features
(seller-grouped seller-free text OOF, seller-pooled text prob, unlabeled seller aggregates, category price, keywords).
usage: python pipe_new.py MODE STACKDIR [lgb|cb|both]   (requires pipe.py MODE STACKDIR already run: reuses stack_MODE.pkl)"""
import os, sys, time, gc
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[k] = "1"
sys.path.insert(0, "/home/claude/wt_main/src")
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from optimize_stack import group_rates, text_predict
MODE, D = sys.argv[1], Path(sys.argv[2]); WHAT = sys.argv[3] if len(sys.argv) > 3 else "lgb"
H2 = Path("/home/claude/work/holdout2"); T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
d = pd.read_pickle(H2 / ("feats_full.pkl" if MODE == "full" else "feats_holdout.pkl"))
if MODE == "inner":
    iv = np.load(D / "inner_eval_rows.npy"); it = np.setdiff1d(np.arange(len(d["y"])), iv)
    x, z, ytr = d["xa"].iloc[it].reset_index(drop=True), d["xa"].iloc[iv].reset_index(drop=True), d["y"][it]
    sell = d["ea"].seller_id.values[it]
else:
    x, z, ytr, sell = d["xa"], d["xb"], d["y"], d["ea"].seller_id.values
del d; gc.collect()
s = pd.read_pickle(D / f"stack_{MODE}.pkl"); assert (s["ytr"] == ytr).all()
drop = [c for c in s["st"].columns if c.endswith(("_oof_rate", "_oof_count")) or c == "text_oof_probability"]
tr, va = s["st"].drop(columns=drop), s["sv"].drop(columns=drop)
st, sv = group_rates(tr, va, ytr, sell)          # seller-grouped folds
oof = np.zeros(len(ytr))
for k, (tt, vv) in enumerate(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=2026).split(np.zeros(len(ytr)), ytr, sell)):
    oof[vv] = text_predict(x.iloc[tt], ytr[tt], x.iloc[vv]); log("text oof (seller groups)", k)
st["text_oof_probability"] = oof; sv["text_oof_probability"] = text_predict(x, ytr, z)
sid = [c for c in st.columns if c.startswith(("seller_id", "seller_category", "seller_listing"))]
a = pd.concat([st.drop(columns=sid), s["new_tr"]], axis=1); b = pd.concat([sv.drop(columns=sid), s["new_ev"]], axis=1)
cats = [c for c in s["cats"] if c not in sid]
log("dropped", sid, "shape", a.shape)
if WHAT in ("lgb", "both"):
    params = {"n_estimators": 430, "learning_rate": 0.0228919761888824, "num_leaves": 94, "min_child_samples": 30, "reg_lambda": 0.6140703790090526,
              "reg_alpha": 0.07617361848894641, "colsample_bytree": 0.8889021191105235, "subsample": 0.9638149477566389, "subsample_freq": 1,
              "max_cat_threshold": 8, "cat_smooth": 22.90498629058358, "n_jobs": 1, "verbosity": -1, "random_state": 42}
    a2, b2 = a.copy(), b.copy()
    for c in cats:
        dt = pd.CategoricalDtype(categories=pd.Index(a2[c].unique())); a2[c] = a2[c].astype(dt); b2[c] = b2[c].astype(dt)
    m = LGBMClassifier(**params); m.fit(a2, ytr); np.save(D / "newspec_lgb.npy", m.predict_proba(b2)[:, 1]); log("lgb done")
    pd.Series(m.booster_.feature_importance("gain"), index=a2.columns).sort_values(ascending=False).head(40).to_csv(D / "newspec_lgb_importance.csv")
if WHAT in ("cb", "both"):
    m = CatBoostClassifier(iterations=600, depth=6, learning_rate=0.07, l2_leaf_reg=6, random_seed=145, loss_function="Logloss", thread_count=1,
                           allow_writing_files=False, one_hot_max_size=10, random_strength=.5, verbose=200)
    m.fit(a, ytr, cat_features=cats); np.save(D / "newspec_cb.npy", m.predict_proba(b)[:, 1]); log("cb done")
log("ALL DONE")
