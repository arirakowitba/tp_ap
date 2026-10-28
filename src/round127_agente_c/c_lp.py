"""Propagación de etiquetas transductiva (Zhou et al.) en un grafo de avisos train+test:
aristas = vecinos de título (c_knn_all.npz, top K, peso sim^p) + opcional aristas de vendedor (mismo seller_id, peso ws/grado).
Para el fold k sólo se fijan las etiquetas de los otros 4 folds; fold k y test quedan sin etiqueta (OOF limpio).
Para test se fijan las 70k etiquetas. Salida: score propagado (y probabilidad por logística anidada) -> pred_c_<name>.npz
usage: python c_lp.py NAME [K=20] [alpha=0.9] [ws=0] [iters=30]"""
import sys, numpy as np, pandas as pd, scipy.sparse as sp
from c_common import *
name = sys.argv[1]; K = int(sys.argv[2]) if len(sys.argv) > 2 else 20; alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
ws = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0; iters = int(sys.argv[5]) if len(sys.argv) > 5 else 30
Z = np.load(D + "c_knn_all.npz"); NI, NS = Z["NI"][:, :K], Z["NS"][:, :K]; n = NI.shape[0]
y = load_y(); ntr = len(y); fo = folds(y)
W = sp.csr_matrix((np.clip(NS, 0, 1).ravel() ** 2, (np.repeat(np.arange(n), K), NI.ravel())), shape=(n, n)); W = W.maximum(W.T)
if ws > 0:
    s = pd.factorize(pd.read_pickle(D + "feats.pkl")["F"]["seller_id"].astype(str))[0]
    G = sp.csr_matrix((np.ones(n), (np.arange(n), s))); deg = np.asarray(G.sum(0)).ravel()
    Sm = (G @ sp.diags(ws / np.maximum(deg - 1, 1)) @ G.T).tocsr(); Sm.setdiag(0); Sm.eliminate_zeros(); W = (W + Sm).tocsr()
d = np.asarray(W.sum(1)).ravel(); Dm = sp.diags(1 / np.sqrt(np.maximum(d, 1e-9))); S = (Dm @ W @ Dm).tocsr()
def prop(lab_idx):
    Y0 = np.zeros((n, 2)); Y0[lab_idx, 0] = y[lab_idx] == 1; Y0[lab_idx, 1] = y[lab_idx] == 0
    f = Y0.copy()
    for _ in range(iters): f = alpha * (S @ f) + (1 - alpha) * Y0
    return np.log((f[:, 0] + 1e-4) / (f[:, 1] + 1e-4)), np.log(f.sum(1) + 1e-6)
sc = np.zeros(ntr); mass = np.zeros(ntr)
for a, b in fo:
    lab = a; s_, m_ = prop(lab); sc[b] = s_[b]; mass[b] = m_[b]
s_, m_ = prop(np.arange(ntr)); tsc, tmass = s_[ntr:], m_[ntr:]
# calibrar con logística anidada sobre [score, masa, score*masa]
X = np.c_[sc, mass, sc * mass]; XT = np.c_[tsc, tmass, tsc * tmass]; oof = np.zeros(ntr); te = np.zeros(n - ntr)
for a, b in fo:
    m = LogisticRegression(max_iter=2000).fit(X[a], y[a]); oof[b] = m.predict_proba(X[b])[:, 1]; te += m.predict_proba(XT)[:, 1] / 5
print(name, f"K={K} alpha={alpha} ws={ws}", "OOF acc %.5f ll %.5f" % ev(y, oof), flush=True); blend_eval(y, oof); save_pred(name, oof, te)
np.savez(D + f"c_lpraw_{name}.npz", sc=np.r_[sc, tsc], mass=np.r_[mass, tmass])
