"""Modelos sklearn no-GBM sobre la tabla de train_gbm.py (cache de B: b_frame_sparse_all_nbsvm_knn_title.pkl).
usage: python c_sk.py NAME et|rf|knn [params...]"""
import os, sys, time, numpy as np, pandas as pd
from c_common import *
name, algo = sys.argv[1], sys.argv[2]
B = pd.read_pickle(D + "b_frame_sparse_all_nbsvm_knn_title.pkl"); F, y, ntr, CAT = B["F"], B["y"], B["ntr"], B["CAT"]
F = F.copy()
if os.environ.get("SELLEROTHER", "orig") == "perfold":  # versión sin fuga de B (b_perfold_<s>.npy)
    _codes = pd.factorize(F.seller_id.astype(str))[0]; _ns = _codes.max() + 1; _fo = folds(y)
    def _mates(v):
        sm = np.bincount(_codes, v, _ns); sc = np.bincount(_codes, None, _ns)
        return np.where(sc[_codes] > 1, (sm[_codes] - v) / np.maximum(sc[_codes] - 1, 1), np.nan)
    for _s in ["sparse_all", "nbsvm"]:
        PF = np.load(D + f"b_perfold_{_s}.npy").astype(float); v = np.full(len(F), np.nan)
        for kk, (aa, bb) in enumerate(_fo): v[bb] = _mates(PF[kk])[bb]
        v[ntr:] = _mates(PF.mean(0))[ntr:]; F["stk_sellerother_" + _s] = v
    print("stk_sellerother: perfold", flush=True)
if os.environ.get("SELLEROTHER") == "none":
    F = F.drop(columns=[c for c in F.columns if c.startswith("stk_sellerother_")]); print("stk_sellerother: none", flush=True)
num = [c for c in F.columns if c not in CAT]
SELLER_COLS = ("te_seller", "tecnt_seller", "sell_", "stk_sellerother", "knn_same_seller", "knn_other_seller", "freq_seller", "seller_")
if os.environ.get("NOSELLER") == "1":
    num = [c for c in num if not c.startswith(SELLER_COLS)]; CAT = [c for c in CAT if c != "seller_id"]
print("n num", len(num), "cats", CAT, flush=True)
X = F[num].astype(np.float32).values
fo = folds(y); oof = np.zeros(ntr); te = np.zeros(len(F) - ntr); t0 = time.time()
if algo in ("et", "rf"):
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    X = np.nan_to_num(X, nan=-999.0)
    kw = dict(n_estimators=int(sys.argv[3]) if len(sys.argv) > 3 else 300, min_samples_leaf=int(sys.argv[4]) if len(sys.argv) > 4 else 2,
              max_features=float(sys.argv[5]) if len(sys.argv) > 5 else 0.3, n_jobs=1, random_state=2026)
    M = ExtraTreesClassifier if algo == "et" else RandomForestClassifier
    for k, (a, b) in enumerate(fo):
        m = M(**kw).fit(X[a], y[a]); oof[b] = m.predict_proba(X[b])[:, 1]; te += m.predict_proba(X[ntr:])[:, 1] / 5
        print(f"{name} fold {k} acc {((oof[b]>.5)==y[b]).mean():.4f} t {time.time()-t0:.0f}s", flush=True); del m
elif algo == "knn":
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.neighbors import KNeighborsClassifier
    Xq = QuantileTransformer(n_quantiles=200, output_distribution="normal", subsample=100000, random_state=0).fit_transform(np.nan_to_num(X, nan=-999.0)).astype(np.float32)
    k_ = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    for k, (a, b) in enumerate(fo):
        m = KNeighborsClassifier(k_, weights="distance", n_jobs=1).fit(Xq[a], y[a]); oof[b] = m.predict_proba(Xq[b])[:, 1]; te += m.predict_proba(Xq[ntr:])[:, 1] / 5
        print(f"{name} fold {k} acc {((oof[b]>.5)==y[b]).mean():.4f} t {time.time()-t0:.0f}s", flush=True)
oof = np.clip(oof, 1e-4, 1 - 1e-4); te = np.clip(te, 1e-4, 1 - 1e-4)
print(name, "OOF acc %.5f ll %.5f" % ev(y, oof)); blend_eval(y, oof); save_pred(name, oof, te)
