"""Separate OOF text models used as CatBoost stacks (the stacks sparse_all+nbsvm are worth ~+0.35 points OOF).
Each one sees a single view so their errors differ: title words, title chars, warranty, attributes, title+warranty+attr."""
import os, numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

D = os.environ.get("AP2026Q2_DATA_DIR", "data/ap_2026_q_2").rstrip("/") + "/"
P = pd.read_pickle(D + "feats.pkl"); y = P["y"]; ntr = len(y)
folds = list(StratifiedKFold(5, shuffle=True, random_state=2026).split(np.zeros(ntr), y))
tv = lambda **kw: TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", lowercase=True, **kw)
views = {
    "txt_title_word": (P["text"], tv(ngram_range=(1, 3), min_df=2), 4.0),
    "txt_title_char": (P["text"], tv(analyzer="char_wb", ngram_range=(2, 6), min_df=3, max_features=400000), 4.0),
    "txt_warranty": (P["warranty"], tv(ngram_range=(1, 3), min_df=2), 2.0),
    "txt_attr": (P["attr"], tv(ngram_range=(1, 2), min_df=2), 2.0),
    "txt_combo": (np.array([f"{a} || {b} || {c}" for a, b, c in zip(P["text"], P["warranty"], P["attr"])]), tv(ngram_range=(1, 2), min_df=2), 4.0),
}
for name, (texts, vec, C) in views.items():
    if os.path.exists(D + f"pred_{name}.npz"): continue
    X = vec.fit_transform(texts).tocsr(); Xtr, Xte = X[:ntr], X[ntr:]
    oof = np.zeros(ntr); te = np.zeros(X.shape[0] - ntr)
    for a, b in folds:
        m = LogisticRegression(C=C, max_iter=3000, solver="liblinear").fit(Xtr[a], y[a])
        oof[b] = m.predict_proba(Xtr[b])[:, 1]; te += m.predict_proba(Xte)[:, 1] / 5
    print(name, "OOF acc", ((oof > .5) == y).mean().round(4), flush=True)
    np.savez(D + f"pred_{name}.npz", oof=oof, te=te)
