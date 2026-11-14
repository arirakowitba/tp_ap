"""sknn ANIDADO: para el CatBoost del fold externo k, las features de vecinos del mismo vendedor se calculan como en test:
- filas de validación (fold k) y filas de test: candidatos = filas de train del vendedor en folds != k (pool del modelo k), conteos x0,75
- filas de entrenamiento del modelo k (fold f != k): candidatos = filas de train del vendedor en folds no en {k, f} (nunca la validación)
Así ninguna feature de una fila de entrenamiento usa etiquetas de la validación (igual que en test real, donde ninguna fila de train
usa etiquetas de test). Salida: D/sknn_nested.npy con forma (5, N, 9), mismas columnas que seller_knn.py."""
import os, numpy as np, pandas as pd, scipy.sparse as sp, time
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
D = os.environ.get("AP2026Q2_DATA_DIR").rstrip("/") + "/"
P = pd.read_pickle(D + "feats.pkl"); y = P["y"].astype(float); ntr = len(y); N = len(P["text"])
seller = pd.factorize(P["F"]["seller_id"].astype(str))[0]
fold = np.full(N, -1)
for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=2026).split(np.zeros(ntr), y)): fold[b] = k
t = P["text"]; t0 = time.time()
Xw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode").fit_transform(t)
Xc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, strip_accents="unicode", max_features=200000).fit_transform(t)
X = normalize(sp.hstack([Xw, Xc]).tocsr()).astype(np.float32)
yy = np.r_[y, np.full(N - ntr, np.nan)]
OUT = np.full((5, N, 9), np.nan); OUT[:, :, 0] = 0; OUT[:, :, 5] = 0; OUT[:, :, 7] = 0
order = np.argsort(seller, kind="stable"); bounds = np.flatnonzero(np.diff(seller[order])) + 1
groups = [g for g in np.split(order, bounds) if len(g) >= 2]
for g in groups:
    S = (X[g] @ X[g].T).toarray(); fg = fold[g]; L = yy[g]
    for k in range(5):
        pool = (fg >= 0) & (fg != k)                      # filas de train del modelo k
        is_tr = pool                                       # filas de entrenamiento del modelo k
        # fila de entrenamiento (fold f): candidatos en pool con fold != f ; fila val/test: todo el pool
        V = pool[None, :] & (~is_tr[:, None] | (fg[:, None] != fg[None, :]))
        np.fill_diagonal(V, False)
        for ii in range(len(g)):
            v = V[ii]
            if not v.any(): continue
            s = S[ii, v]; l = L[v]; sc = 1.0 if is_tr[ii] else 0.75
            j = np.argmax(s); w = np.maximum(s, 1e-3); m8 = s >= 0.8; m5 = s >= 0.5
            OUT[k, g[ii]] = [v.sum() * sc, s[j], l[j], (w * l).sum() / w.sum(), (w ** 4 * l).sum() / (w ** 4).sum(),
                             m8.sum() * sc, l[m8].mean() if m8.any() else np.nan, m5.sum() * sc, l[m5].mean() if m5.any() else np.nan]
np.save(D + "sknn_nested.npy", OUT.astype(np.float32))
print("sknn_nested done", round(time.time() - t0), "s")
