# round145: copia de round126_modelos/sparse_all.py con UN cambio: binning por VALOR (los empates quedan en el mismo bin).
# El original usaba rank(method="first"): los empates se rompen por orden de fila (train primero, test al final), así que las
# filas de test caen en bins que casi no tienen filas de train (77/107 columnas) -> predicción de test degradada.
"""Big sparse linear model on everything (text n-grams + one-hot categoricals + binned numerics + seller one-hot).
Produces OOF/test probabilities used as a stacking feature (same 5 folds as train_v1)."""
import pandas as pd, numpy as np, scipy.sparse as sp, sys
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

import os
D = os.environ.get("AP2026Q2_DATA_DIR", "data/ap_2026_q_2").rstrip("/") + "/"
P = pd.read_pickle(D + "feats.pkl")
F, y, is_test, CAT = P["F"], P["y"], P["is_test"], P["CAT"]
ntr = int((is_test == 0).sum())
folds = list(StratifiedKFold(5, shuffle=True, random_state=2026).split(np.zeros(ntr), y))

def tfidf(texts, **kw):
    return TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", lowercase=True, **kw).fit_transform(texts)
blocks = [
    tfidf(P["text"], ngram_range=(1, 2), min_df=2),
    tfidf(P["text"], analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=300000),
    tfidf(P["warranty"], ngram_range=(1, 3), min_df=2),
    tfidf(P["attr"], ngram_range=(1, 1), min_df=2),
]
num = [c for c in F.columns if c not in CAT and F[c].dtype != object]
def vbin(v):  # pocos valores distintos: el valor mismo; si no, 20 cuantiles por valor (empates juntos, NaN aparte)
    v = v.astype(float)
    return v.astype(str) if v.nunique() <= 20 else pd.qcut(v, 20, labels=False, duplicates="drop").astype(str)
binned = pd.DataFrame({c: vbin(F[c]) for c in num})
cats = pd.concat([F[CAT].astype(str), binned], axis=1)
oh = OneHotEncoder(handle_unknown="ignore", min_frequency=2).fit_transform(cats)
X = sp.hstack(blocks + [oh]).tocsr()
print("X", X.shape, flush=True)
Xtr, Xte = X[:ntr], X[ntr:]
C = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
oof = np.zeros(ntr); te = np.zeros(X.shape[0] - ntr)
for a, b in folds:
    m = LogisticRegression(C=C, max_iter=3000, solver="liblinear")
    m.fit(Xtr[a], y[a]); oof[b] = m.predict_proba(Xtr[b])[:, 1]; te += m.predict_proba(Xte)[:, 1] / 5
print("sparse_fix C", C, "OOF acc", ((oof > .5) == y).mean().round(5), flush=True)
np.savez(D + f"pred_sparse_fix.npz", oof=oof, te=te)
