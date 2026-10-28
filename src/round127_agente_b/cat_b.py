"""CatBoost con la configuración de cat_d5_ll (prof. 5, corte por Logloss, od_wait 400, l2 5, one_hot 8) sobre frame.py,
con opciones del agente B. Mismas semillas por fold (SEED + k) que train_gbm.py.
usage: python cat_b.py NAME
env: STACKS (coma), EXTRAS (coma), SELLEROTHER=orig|perfold|none, NEWSTK (stacks extra sin sellerother, coma),
     SEED, CAT_LR, CAT_DEPTH, FOLDS (ej. "0,1" para screening rápido)"""
import sys, os, time, json, numpy as np, pandas as pd
from common import *
from frame import build
name = sys.argv[1]
STK = [s for s in os.environ.get("STACKS", "sparse_all,nbsvm").split(",") if s]
EXT = [e for e in os.environ.get("EXTRAS", "knn_title.pkl").split(",") if e]
SO = os.environ.get("SELLEROTHER", "orig"); NEW = [s for s in os.environ.get("NEWSTK", "").split(",") if s]
o = build(tuple(STK), tuple(EXT)); F, y, ntr, CAT = o["F"].copy(), o["y"], o["ntr"], o["CAT"]
fo = folds(y); FOLDS = [int(k) for k in os.environ.get("FOLDS", "0,1,2,3,4").split(",")]
seller = F.seller_id.astype(str).values
codes = pd.factorize(seller)[0]; ns = codes.max() + 1
def mates_mean(v):
    sm = np.bincount(codes, v, ns); sc = np.bincount(codes, None, ns)
    return np.where(sc[codes] > 1, (sm[codes] - v) / np.maximum(sc[codes] - 1, 1), np.nan)
perfold = {}
for s in STK:
    if SO == "none": F = F.drop(columns=["stk_sellerother_" + s])
    elif SO == "perfold":
        PF = np.load(D + f"b_perfold_{s}.npy").astype(float)   # (5, N): modelo de fold k sobre todas las filas
        perfold[s] = [mates_mean(PF[k]) for k in range(5)]     # versión sin fuga para las filas del fold k
        perfold[s].append(mates_mean(PF.mean(0)))              # test: promedio de los 5 modelos
for s in NEW:
    d = np.load(D + f"pred_{s}.npz"); F["stk_" + s] = np.r_[d["oof"], d["te"]]
cols = [c for c in F.columns if c != "__"]
SEED = int(os.environ.get("SEED", "4141"))
from catboost import CatBoostClassifier, Pool
catc = [c for c in cols if c in CAT]
X = F[cols].copy()
for c in catc: X[c] = X[c].astype(str)
oof = np.full(ntr, np.nan); te = np.zeros(len(F) - ntr); its = []; t0 = time.time()
for k in FOLDS:
    a, b = fo[k]; Xk = X
    if SO == "perfold":
        Xk = X.copy()
        for s in STK:
            # filas de entrenamiento a: su valor "OOF" honesto es el del modelo de su propio fold;
            # para entrenar el CatBoost del fold k usamos, en cada fila de train, la versión de su fold (como en test-time del 2.º nivel)
            v = np.full(len(F), np.nan)
            for kk, (aa, bb) in enumerate(fo): v[bb] = perfold[s][kk][bb]
            v[ntr:] = perfold[s][5][ntr:]
            Xk["stk_sellerother_" + s] = v
    Xtr, Xte = Xk.iloc[:ntr], Xk.iloc[ntr:]
    m = CatBoostClassifier(iterations=6000, learning_rate=float(os.environ.get("CAT_LR", "0.06")), depth=int(os.environ.get("CAT_DEPTH", "5")),
                           l2_leaf_reg=float(os.environ.get("CAT_L2", "5")), eval_metric="Logloss", od_type="Iter", od_wait=400,
                           random_seed=SEED + k, thread_count=1, verbose=0, one_hot_max_size=8, train_dir=D + "b_catboost_info")
    m.fit(Pool(Xtr.iloc[a], y[a], cat_features=catc), eval_set=Pool(Xtr.iloc[b], y[b], cat_features=catc), use_best_model=True)
    oof[b] = m.predict_proba(Pool(Xtr.iloc[b], cat_features=catc))[:, 1]; te += m.predict_proba(Pool(Xte, cat_features=catc))[:, 1] / len(FOLDS); its.append(m.get_best_iteration())
    print(f"{name} fold {k} acc {((oof[b] > .5) == y[b]).mean():.4f} ll {ev(y[b], oof[b])[1]:.4f} it {its[-1]} t {time.time()-t0:.0f}s", flush=True)
msk = ~np.isnan(oof); acc, ll = ev(y[msk], oof[msk]); print(f"{name} OOF({len(FOLDS)} folds) acc {acc:.5f} logloss {ll:.5f}", flush=True)
if len(FOLDS) == 5: save_pred(name, oof, te)
else: np.savez(D + f"b_partial_{name}.npz", oof=oof, te=te)
json.dump({"name": name, "env": {k: v for k, v in os.environ.items() if k in ["STACKS", "EXTRAS", "SELLEROTHER", "NEWSTK", "SEED", "CAT_LR", "CAT_DEPTH", "CAT_L2", "FOLDS"]},
           "oof_acc": acc, "logloss": ll, "iters": its}, open(D + f"report_{name}.json", "w"))
