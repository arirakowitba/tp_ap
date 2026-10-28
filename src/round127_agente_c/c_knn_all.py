"""Vecinos por título (TF-IDF palabra 1-2 + char_wb 3-5, coseno) entre TODOS los 100k avisos (train+test), top 20 sin uno mismo.
No usa etiquetas: es una representación no supervisada para propagación de etiquetas transductiva. Guarda D/c_knn_all.npz"""
import numpy as np, pandas as pd, scipy.sparse as sp, time
from c_common import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
P = pd.read_pickle(D + "feats.pkl"); t = P["text"]
Xw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode").fit_transform(t)
Xc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, strip_accents="unicode", max_features=200000).fit_transform(t)
X = normalize(sp.hstack([Xw, Xc]).tocsr()).astype(np.float32); XT = X.T.tocsc(); K = 20; n = X.shape[0]
NI = np.zeros((n, K), np.int32); NS = np.zeros((n, K), np.float32); t0 = time.time()
for st in range(0, n, 500):
    M = (X[st:st + 500] @ XT).toarray(); M[np.arange(M.shape[0]), np.arange(st, st + M.shape[0])] = -1
    part = np.argpartition(-M, K, axis=1)[:, :K]; ss = np.take_along_axis(M, part, 1); o = np.argsort(-ss, 1)
    NI[st:st + 500] = np.take_along_axis(part, o, 1); NS[st:st + 500] = np.take_along_axis(ss, o, 1)
    if st % 10000 == 0: print(st, round(time.time() - t0), flush=True)
path = D + "c_knn_all.npz"; assert not __import__("os").path.exists(path); np.savez(path, NI=NI, NS=NS); print("done", round(time.time() - t0))
