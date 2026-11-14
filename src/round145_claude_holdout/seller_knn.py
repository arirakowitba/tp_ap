"""Vecinos de título DENTRO del mismo vendedor (OOF, mismos 5 folds semilla 2026).
Fila de train del fold f: candidatos = filas de train del mismo vendedor en folds != f. Fila de test: todas las filas de train del vendedor.
Nunca entra la propia fila ni otras filas de test. Conteos de test escalados x0,8 (misma convención que tecnt_* de train_gbm.py).
Salida: D/sknn.pkl (columnas sknn_*)."""
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
cols = ["sknn_ncand", "sknn_maxsim", "sknn_lab1", "sknn_wmean", "sknn_wmean_p4", "sknn_cnt08", "sknn_mean08", "sknn_cnt05", "sknn_mean05"]
out = np.full((N, len(cols)), np.nan); out[:, 0] = 0; out[:, 5] = 0; out[:, 7] = 0
order = np.argsort(seller, kind="stable"); bounds = np.flatnonzero(np.diff(seller[order])) + 1
for g in np.split(order, bounds):
    if len(g) < 2: continue
    S = (X[g] @ X[g].T).toarray(); fg = fold[g]; tr = fg >= 0
    # candidato j válido para i: j es de train, j != i, y (i test) o fold distinto
    V = tr[None, :] & ((fg[:, None] == -1) | (fg[:, None] != fg[None, :]))
    np.fill_diagonal(V, False)
    L = yy[g]
    for ii in range(len(g)):
        v = V[ii]
        if not v.any(): continue
        s = S[ii, v]; l = L[v]; sc = 0.8 if fg[ii] == -1 else 1.0
        j = np.argmax(s); w = np.maximum(s, 1e-3)
        m8 = s >= 0.8; m5 = s >= 0.5
        out[g[ii]] = [v.sum() * sc, s[j], l[j], (w * l).sum() / w.sum(), (w ** 4 * l).sum() / (w ** 4).sum(),
                      m8.sum() * sc, l[m8].mean() if m8.any() else np.nan, m5.sum() * sc, l[m5].mean() if m5.any() else np.nan]
E = pd.DataFrame(out, columns=cols); E.to_pickle(D + "sknn.pkl")
m = ~np.isnan(E.sknn_lab1.values[:ntr])
print("sknn done", round(time.time() - t0), "s; train rows with candidate", m.mean().round(4), "acc lab1 there", ((E.sknn_lab1.values[:ntr][m] > .5) == y[m]).mean().round(4))
