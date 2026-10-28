"""LightGBM / XGBoost sobre la misma tabla que CatBoost (frame.py), 1 hilo.
usage: python tab_gbm.py NAME lgb|xgb [CONFIG_KEY]
Variantes por env: DROP_HICARD=1 saca category_id/city/seller_id crudos (quedan TE OOF y frecuencias)."""
import sys, os, time, json, numpy as np, pandas as pd
from common import *
from frame import build
name, algo = sys.argv[1], sys.argv[2]
STK = os.environ.get("STACKS", "sparse_all,nbsvm").split(","); EXT = [e for e in os.environ.get("EXTRAS", "knn_title.pkl").split(",") if e]
o = build(tuple(STK), tuple(EXT)); F, y, ntr, CAT = o["F"], o["y"], o["ntr"], o["CAT"]
fo = folds(y)
F = F.copy()
SO = os.environ.get("SELLEROTHER", "orig")
if SO != "orig":
    codes = pd.factorize(F.seller_id.astype(str))[0]; ns = codes.max() + 1
    def mates_mean(v):
        sm = np.bincount(codes, v, ns); sc = np.bincount(codes, None, ns)
        return np.where(sc[codes] > 1, (sm[codes] - v) / np.maximum(sc[codes] - 1, 1), np.nan)
    for s in STK:
        if SO == "none": F = F.drop(columns=["stk_sellerother_" + s]); continue
        PF = np.load(D + f"b_perfold_{s}.npy").astype(float); v = np.full(len(F), np.nan)
        for kk, (aa, bb) in enumerate(fo): v[bb] = mates_mean(PF[kk])[bb]
        v[ntr:] = mates_mean(PF.mean(0))[ntr:]; F["stk_sellerother_" + s] = v
drop = ["seller_id"]
if os.environ.get("DROP_HICARD", "1") == "1": drop += ["category_id", "city"]
cols = [c for c in F.columns if c not in drop]
X = F[cols].copy()
for c in cols:
    if c in CAT: X[c] = X[c].astype(str).astype("category")
Xtr, Xte = X.iloc[:ntr], X.iloc[ntr:]
SEED = int(os.environ.get("SEED", "2026")); LR = float(os.environ.get("LR", "0.02"))
oof = np.zeros(ntr); te = np.zeros(len(Xte)); its = []; t0 = time.time()
for k, (a, b) in enumerate(fo):
    if algo == "lgb":
        import lightgbm as lgb
        p = dict(objective="binary", learning_rate=LR, num_leaves=int(os.environ.get("LEAVES", "31")), max_depth=int(os.environ.get("MAXD", "-1")),
                 min_child_samples=int(os.environ.get("MCS", "40")), feature_fraction=float(os.environ.get("FF", "0.4")), bagging_fraction=0.8, bagging_freq=1,
                 lambda_l2=float(os.environ.get("L2", "10")), lambda_l1=float(os.environ.get("L1", "0")), min_gain_to_split=0.0,
                 cat_smooth=50, cat_l2=20, max_cat_to_onehot=8, min_data_per_group=50, path_smooth=float(os.environ.get("PSMOOTH", "0")),
                 verbose=-1, seed=SEED + k, num_threads=1, metric="binary_logloss")
        m = lgb.train(p, lgb.Dataset(Xtr.iloc[a], y[a]), 20000, valid_sets=[lgb.Dataset(Xtr.iloc[b], y[b])], callbacks=[lgb.early_stopping(400, verbose=False)])
        oof[b] = m.predict(Xtr.iloc[b], num_iteration=m.best_iteration); te += m.predict(Xte, num_iteration=m.best_iteration) / 5; its.append(m.best_iteration)
    else:
        import xgboost as xgb
        dtr = xgb.DMatrix(Xtr.iloc[a], y[a], enable_categorical=True); dva = xgb.DMatrix(Xtr.iloc[b], y[b], enable_categorical=True)
        p = dict(objective="binary:logistic", eval_metric="logloss", eta=LR, max_depth=int(os.environ.get("MAXD", "6")), min_child_weight=float(os.environ.get("MCW", "3")),
                 subsample=0.8, colsample_bytree=float(os.environ.get("FF", "0.5")), reg_lambda=float(os.environ.get("L2", "5")), tree_method="hist",
                 max_cat_to_onehot=8, max_bin=256, nthread=1, seed=SEED + k)
        m = xgb.train(p, dtr, 20000, evals=[(dva, "va")], early_stopping_rounds=400, verbose_eval=False)
        r = (0, m.best_iteration + 1)
        oof[b] = m.predict(dva, iteration_range=r); te += m.predict(xgb.DMatrix(Xte, enable_categorical=True), iteration_range=r) / 5; its.append(m.best_iteration)
    print(f"{name} fold {k} acc {((oof[b] > .5) == y[b]).mean():.4f} it {its[-1]} t {time.time()-t0:.0f}s", flush=True)
acc, ll = ev(y, oof); print(f"{name} OOF acc {acc:.5f} logloss {ll:.5f}", flush=True)
save_pred(name, oof, te)
json.dump({"name": name, "algo": algo, "env": {k: v for k, v in os.environ.items() if k in ["SEED", "LR", "LEAVES", "MAXD", "MCS", "FF", "L2", "L1", "MCW", "DROP_HICARD", "STACKS", "EXTRAS", "PSMOOTH", "SELLEROTHER"]},
           "oof_acc": acc, "logloss": ll, "iters": its}, open(D + f"report_{name}.json", "w"))
