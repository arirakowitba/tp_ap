"""NB-SVM (log-count-ratio scaled binary n-grams + logistic) per text field. OOF on the shared 5 folds."""
import pandas as pd, numpy as np, scipy.sparse as sp, sys
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

import os
D = os.environ.get("AP2026Q2_DATA_DIR", "data/ap_2026_q_2").rstrip("/") + "/"
P = pd.read_pickle(D + "feats.pkl"); y = P["y"]; ntr = int((P["is_test"] == 0).sum())
folds = list(StratifiedKFold(5, shuffle=True, random_state=2026).split(np.zeros(ntr), y))
txt = np.array([f"{a} __w__ {b}" for a, b in zip(P["text"], P["warranty"])])
Xw = CountVectorizer(ngram_range=(1, 3), min_df=2, binary=True, strip_accents="unicode").fit_transform(txt)
Xc = CountVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, binary=True, strip_accents="unicode", max_features=400000).fit_transform(P["text"])
X = sp.hstack([Xw, Xc]).tocsr().astype(np.float32)
Xtr, Xte = X[:ntr], X[ntr:]
oof = np.zeros(ntr); te = np.zeros(X.shape[0] - ntr)
for a, b in folds:
    p = 1 + Xtr[a][y[a] == 1].sum(0); q = 1 + Xtr[a][y[a] == 0].sum(0)
    r = np.asarray(np.log((p / p.sum()) / (q / q.sum()))).ravel()
    R = sp.diags(r)
    m = LogisticRegression(C=0.5, max_iter=3000, solver="liblinear").fit(Xtr[a] @ R, y[a])
    oof[b] = m.predict_proba(Xtr[b] @ R)[:, 1]; te += m.predict_proba(Xte @ R)[:, 1] / 5
print("nbsvm OOF acc", ((oof > .5) == y).mean().round(5), flush=True)
np.savez(D + "pred_nbsvm.npz", oof=oof, te=te)
