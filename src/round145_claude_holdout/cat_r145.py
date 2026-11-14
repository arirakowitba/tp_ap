"""CatBoost = cat_b.py de round127 con SELLEROTHER=none (b_cat_noso_b: prof. 5, l2 5, one_hot 8, od Logloss 400, semilla SEED+k),
con opciones para diagnosticar y mejorar.
usage: python cat_r145.py NAME
env: STACKS (def. sparse_fix,nbsvm), EXTRAS (def. knn_title.pkl; agregar p.ej. sknn.pkl), DROP=prefijos a quitar,
     ES=val (como cat_b.py: early stopping sobre el fold de validación) | inner (early stopping sobre 10% interno del train del fold,
     el fold de validación no se toca) | fixed (FIXED_IT iteraciones), SEED (4242), CAT_LR (0.06), NOCAT=1 (seller/category/city fuera de cat_features),
     TEXT=1 (título y garantía crudos como text_features de CatBoost), SKNN_NESTED=1 (sknn anidado por fold, ver sknn_nested.py),
     TE_NESTED=1 (te_*/tecnt_* anidados por fold externo: ninguna fila de entrenamiento usa etiquetas de la validación)"""
import sys, os, time, json, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/tp_ap/src/round127_agente_b")
from common import D, folds, ev
from frame import build
from catboost import CatBoostClassifier, Pool
name = sys.argv[1]
STK = tuple(s for s in os.environ.get("STACKS", "sparse_fix,nbsvm").split(",") if s)
EXT = tuple(e for e in os.environ.get("EXTRAS", "knn_title.pkl").split(",") if e)
o = build(STK, ("knn_title.pkl",)); F, y, ntr, CAT = o["F"].copy(), o["y"], o["ntr"], o["CAT"]
F = F.drop(columns=[c for c in F.columns if c.startswith("stk_sellerother_")])
for e in EXT:
    if e == "knn_title.pkl": continue
    E = pd.read_pickle(D + e)
    for c in E.columns: F[c] = E[c].values
drops = [p for p in os.environ.get("DROP", "").split(",") if p]
F = F.drop(columns=[c for c in F.columns if any(c.startswith(p) for p in drops)])
TEXT = os.environ.get("TEXT", "0") == "1"; TXT = []
if TEXT:
    P = pd.read_pickle(D + "feats.pkl"); F["title_raw"] = P["text"]; F["warranty_raw"] = P["warranty"]; TXT = ["title_raw", "warranty_raw"]
cols = list(F.columns)
NOCAT = os.environ.get("NOCAT", "0") == "1"
catc = [c for c in cols if c in CAT and not (NOCAT and c in ("seller_id", "category_id", "city"))]
X = F[cols].copy()
for c in catc: X[c] = X[c].astype(str)
for c in cols:
    if c not in catc and c not in TXT and str(X[c].dtype) == "category": X[c] = X[c].cat.codes
fo = folds(y); SEED = int(os.environ.get("SEED", "4242")); ES = os.environ.get("ES", "val"); FIX = int(os.environ.get("FIXED_IT", "2000"))
Xtr, Xte = X.iloc[:ntr], X.iloc[ntr:]
pool = lambda A, yy=None: Pool(A, yy, cat_features=catc, text_features=TXT or None)
oof = np.full(ntr, np.nan); te = np.zeros(len(Xte)); its = []; t0 = time.time()
TEN = os.environ.get("TE_NESTED", "0") == "1"
if TEN:  # TE anidado: para el modelo del fold k, las filas de entrenamiento (fold f) usan sólo folds no en {k,f}; val y test usan todo folds != k
    G = o["F"]; KEYS = {"seller": G.seller_id.astype(str).values, "cat": G.category_id.astype(str).values, "city": G.city.astype(str).values,
                        "seller_cat": (G.seller_id.astype(str) + "|" + G.category_id.astype(str)).values,
                        "seller_lt": (G.seller_id.astype(str) + "|" + G.listing_type_id.astype(str)).values}
    fold_of = np.full(len(F), -1)
    for kk, (aa, bb) in enumerate(fo): fold_of[bb] = kk
    def te_nested(keys, k, alpha=5):
        out = np.zeros(len(F)); cnt = np.zeros(len(F)); trk = np.flatnonzero((fold_of >= 0) & (fold_of != k)); prior = y[trk].mean()
        def enc(src, dst, sc):
            st = pd.DataFrame({"k": keys[src], "y": y[src]}).groupby("k").y.agg(["sum", "count"]).reindex(keys[dst])
            out[dst] = ((st["sum"].fillna(0) + alpha * prior) / (st["count"].fillna(0) + alpha)).values; cnt[dst] = st["count"].fillna(0).values * sc
        for f in range(5):
            if f == k: continue
            enc(trk[fold_of[trk] != f], np.flatnonzero(fold_of == f), 1.0)
        enc(trk, np.flatnonzero((fold_of == k) | (fold_of == -1)), 0.75)
        return out, cnt
    TEMAP = [("te_seller", "tecnt_seller", "seller"), ("te_cat", "tecnt_cat", "cat"), ("te_city", None, "city"),
             ("te_seller_cat", "tecnt_seller_cat", "seller_cat"), ("te_seller_lt", None, "seller_lt")]
    TEN_K = {}
    for k in range(5):
        d = {}
        for c, cc, key in TEMAP:
            v, n = te_nested(KEYS[key], k); d[c] = v
            if cc: d[cc] = n
        TEN_K[k] = {c: v for c, v in d.items() if c in X.columns}
if TEN and os.environ.get("DRYRUN"):
    a0, b0 = fo[0]
    for c in TEN_K[0]:
        print(c, "val rows max|nested-std|", float(np.abs(TEN_K[0][c][b0] - X[c].values[b0].astype(float)).max().round(4)),
              "train rows corr", float(np.corrcoef(TEN_K[0][c][a0], X[c].values[a0].astype(float))[0, 1].round(4)))
    sys.exit(0)
SKN = os.environ.get("SKNN_NESTED", "0") == "1"
if SKN:  # sknn anidado (sknn_nested.py): las columnas sknn_* se reemplazan por la versión del fold externo k
    NEST = np.load(D + "sknn_nested.npy"); SCOLS = ["sknn_ncand", "sknn_maxsim", "sknn_lab1", "sknn_wmean", "sknn_wmean_p4", "sknn_cnt08", "sknn_mean08", "sknn_cnt05", "sknn_mean05"]
    assert all(c in X.columns for c in SCOLS), "SKNN_NESTED=1 requiere EXTRAS con sknn.pkl"
for k, (a, b) in enumerate(fo):
    if SKN or TEN:
        Xk = X.copy()
        if SKN: Xk[SCOLS] = NEST[k]
        if TEN:
            for c, v in TEN_K[k].items(): Xk[c] = v
        Xtr, Xte = Xk.iloc[:ntr], Xk.iloc[ntr:]
    kw = dict(iterations=6000 if ES != "fixed" else FIX, learning_rate=float(os.environ.get("CAT_LR", "0.06")), depth=5, l2_leaf_reg=5,
              eval_metric="Logloss", random_seed=SEED + k, thread_count=1, verbose=0, one_hot_max_size=8, train_dir=D + "r145_catboost_info")
    if ES == "val":
        m = CatBoostClassifier(od_type="Iter", od_wait=400, **kw)
        m.fit(pool(Xtr.iloc[a], y[a]), eval_set=pool(Xtr.iloc[b], y[b]), use_best_model=True)
    elif ES == "inner":
        rng = np.random.RandomState(SEED + k); sh = rng.permutation(a); n_es = len(a) // 10; es, fit = sh[:n_es], sh[n_es:]
        m = CatBoostClassifier(od_type="Iter", od_wait=400, **kw)
        m.fit(pool(Xtr.iloc[fit], y[fit]), eval_set=pool(Xtr.iloc[es], y[es]), use_best_model=True)
    else:
        m = CatBoostClassifier(**kw); m.fit(pool(Xtr.iloc[a], y[a]))
    oof[b] = m.predict_proba(pool(Xtr.iloc[b]))[:, 1]; te += m.predict_proba(pool(Xte))[:, 1] / 5
    its.append(int(m.get_best_iteration()) if ES != "fixed" else FIX)
    print(f"{name} fold {k} acc {((oof[b] > .5) == y[b]).mean():.4f} ll {ev(y[b], oof[b])[1]:.4f} it {its[-1]} t {time.time()-t0:.0f}s", flush=True)
acc, ll = ev(y, oof); print(f"{name} OOF(5 folds) acc {acc:.5f} logloss {ll:.5f}", flush=True)
np.savez(D + f"pred_{name}.npz", oof=oof, te=te)
json.dump({"name": name, "env": {k: os.environ.get(k) for k in ["STACKS", "EXTRAS", "DROP", "ES", "FIXED_IT", "SEED", "CAT_LR", "NOCAT", "TEXT", "SKNN_NESTED", "TE_NESTED"]},
           "oof_acc": acc, "logloss": ll, "iters": its, "n_features": len(cols), "seconds": time.time() - t0}, open(D + f"report_{name}.json", "w"))
