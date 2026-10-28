"""Re-entrena sparse_all y nbsvm (mismas features y C) guardando la predicción de CADA modelo de fold sobre TODAS las filas.
Sirve para armar `stk_sellerother` sin fuga: para la fila i del fold k se promedian las predicciones del modelo k
(que no vio y_i) sobre los otros avisos del vendedor. En el original se usan predicciones OOF de los compañeros,
que vienen de modelos entrenados con y_i (fuga: test no la tiene).
Salida: D/b_perfold_<modelo>.npy  (5, 100000)"""
import sys, numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from common import *
P = pd.read_pickle(D + "feats.pkl"); F, y, CAT = P["F"], P["y"], P["CAT"]; ntr = len(y); fo = folds(y)
which = sys.argv[1]
if which == "sparse_all":
    tf = lambda t, **kw: TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", lowercase=True, **kw).fit_transform(t)
    blocks = [tf(P["text"], ngram_range=(1, 2), min_df=2), tf(P["text"], analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=300000),
              tf(P["warranty"], ngram_range=(1, 3), min_df=2), tf(P["attr"], ngram_range=(1, 1), min_df=2)]
    num = [c for c in F.columns if c not in CAT and F[c].dtype != object]
    binned = pd.DataFrame({c: pd.qcut(F[c].rank(method="first"), 20, labels=False, duplicates="drop").astype(str) for c in num})
    oh = OneHotEncoder(handle_unknown="ignore", min_frequency=2).fit_transform(pd.concat([F[CAT].astype(str), binned], axis=1))
    X = sp.hstack(blocks + [oh]).tocsr(); R = None; C = 1.0
else:
    txt = np.array([f"{a} __w__ {b}" for a, b in zip(P["text"], P["warranty"])])
    Xw = CountVectorizer(ngram_range=(1, 3), min_df=2, binary=True, strip_accents="unicode").fit_transform(txt)
    Xc = CountVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, binary=True, strip_accents="unicode", max_features=400000).fit_transform(P["text"])
    X = sp.hstack([Xw, Xc]).tocsr().astype(np.float32); C = 0.5
out = np.zeros((5, X.shape[0]), np.float32); ref = load_pred(which)[0]; chk = np.zeros(ntr)
for k, (a, b) in enumerate(fo):
    Xa = X[:ntr][a]; M = X
    if which == "nbsvm":
        p = 1 + Xa[y[a] == 1].sum(0); q = 1 + Xa[y[a] == 0].sum(0)
        Rd = sp.diags(np.asarray(np.log((p / p.sum()) / (q / q.sum()))).ravel()); Xa = Xa @ Rd; M = X @ Rd
    m = LogisticRegression(C=C, max_iter=3000, solver="liblinear").fit(Xa, y[a])
    out[k] = m.predict_proba(M)[:, 1]; chk[b] = out[k][:ntr][b]; print(which, "fold", k, flush=True)
print("max |oof reproducido - original|", np.abs(chk - ref).max())
np.save(D + f"b_perfold_{which}.npy", out)
