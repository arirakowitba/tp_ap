"""Proxy rápido (LightGBM, 1 hilo) sobre la MISMA tabla que cat_b.py (frame.build de round127, sin stk_sellerother_*),
para ablaciones de grupos de features y medir OOF vs holdout.
usage: python diag_lgb.py NAME
env: STACKS=sparse_all,nbsvm (o sparse_fix,nbsvm), DROP=prefijo1,prefijo2 (columnas que empiezan con esos prefijos se quitan), ES=val|fixed (val = early stopping sobre el
     fold de validación como en cat_b.py; fixed = iteraciones fijas FIXED_IT), FRAME_EXTRA=archivo.pkl con columnas a reemplazar/agregar"""
import sys, os, json, time, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/tp_ap/src/round127_agente_b")
from common import D, folds, ev
from frame import build
import lightgbm as lgb
name = sys.argv[1]
STK = tuple(s for s in os.environ.get("STACKS", "sparse_all,nbsvm").split(",") if s)
o = build(STK, ("knn_title.pkl",)); F, y, ntr, CAT = o["F"].copy(), o["y"], o["ntr"], o["CAT"]
F = F.drop(columns=[c for c in F.columns if c.startswith("stk_sellerother_")])
if os.environ.get("FRAME_EXTRA"):
    E = pd.read_pickle(D + os.environ["FRAME_EXTRA"])
    for c in E.columns: F[c] = E[c].values
drops = [p for p in os.environ.get("DROP", "").split(",") if p]
F = F.drop(columns=[c for c in F.columns if any(c.startswith(p) for p in drops)])
F = F.drop(columns=[c for c in ["seller_id", "category_id", "city"] if c in F.columns])  # alta cardinalidad: sólo vía TE
for c in F.columns:
    if str(F[c].dtype) == "category": F[c] = F[c].cat.codes
fo = folds(y); ES = os.environ.get("ES", "val"); FIX = int(os.environ.get("FIXED_IT", "800"))
P = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=30, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
         lambda_l2=10, verbose=-1, seed=7, num_threads=1)
Xtr, Xte = F.iloc[:ntr], F.iloc[ntr:]
oof = np.zeros(ntr); te = np.zeros(len(Xte)); its = []; t0 = time.time()
for k, (a, b) in enumerate(fo):
    if ES == "val":
        m = lgb.train(P, lgb.Dataset(Xtr.iloc[a], y[a]), 5000, valid_sets=[lgb.Dataset(Xtr.iloc[b], y[b])], callbacks=[lgb.early_stopping(200, verbose=False)])
        it = m.best_iteration
    else:
        m = lgb.train(P, lgb.Dataset(Xtr.iloc[a], y[a]), FIX); it = FIX
    oof[b] = m.predict(Xtr.iloc[b], num_iteration=it); te += m.predict(Xte, num_iteration=it) / 5; its.append(it)
acc, ll = ev(y, oof)
print(f"{name} OOF acc {acc:.5f} ll {ll:.5f} its {its} nfeat {F.shape[1]} t {time.time()-t0:.0f}s", flush=True)
np.savez(D + f"pred_{name}.npz", oof=oof, te=te)
