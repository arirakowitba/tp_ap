"""Variante de sparse_all.py con tokens crudos extra: valores de atributos (id=valor), combinaciones de variaciones,
tags, sub_status, tamaños de fotos, modo de envío y tablas a_feats_raw binneadas. Logística liblinear, 1 hilo, 5 folds compartidos.
usage: python sparse_raw.py [C]  -> pred_a_sparse_raw.npz"""
import os, sys, re, numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
D = os.environ.get("AP2026Q2_DATA_DIR", "/home/claude/data").rstrip("/") + "/"
P = pd.read_pickle(D + "feats.pkl"); F, y, CAT = P["F"], P["y"], P["CAT"]; ntr = len(y)
tr = pd.read_pickle(D + "tr.pkl"); te = pd.read_pickle(D + "te.pkl"); df = pd.concat([tr, te], ignore_index=True)
R = pd.read_pickle(D + "a_feats_raw.pkl")
folds = list(StratifiedKFold(5, shuffle=True, random_state=2026).split(np.zeros(ntr), y))
lst = lambda x: x if isinstance(x, list) else []
clean = lambda s: re.sub(r"[^a-z0-9]+", "_", str(s).lower())[:40]
def toks(i):
    r = df.iloc[i]; out = []
    for a in lst(r.attributes): out.append(f"A_{clean(a.get('id'))}={clean(a.get('value_name'))}")
    for v in lst(r.variations):
        for c in lst(v.get("attribute_combinations")): out.append(f"V_{clean(c.get('name'))}={clean(c.get('value_name'))}")
    out += [f"T_{t}" for t in lst(r.tags)] + [f"S_{t}" for t in lst(r.sub_status)]
    out += [f"P_{p.get('size')}" for p in lst(r.pictures)[:3]]
    sh = r.shipping if isinstance(r.shipping, dict) else {}
    out += [f"SH_{clean(t)}" for t in lst(sh.get("tags"))] + [f"SM_{sh.get('mode')}"]
    return " ".join(out)
raw_txt = [toks(i) for i in range(len(df))]
tv = lambda **kw: TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", lowercase=True, **kw)
LIGHT = os.environ.get("LIGHT", "1") == "1"   # sin char n-gramas (memoria); sparse_all ya los tiene
blocks = [tv(ngram_range=(1, 2), min_df=2).fit_transform(P["text"]),
          tv(ngram_range=(1, 3), min_df=2).fit_transform(P["warranty"]),
          TfidfVectorizer(token_pattern=r"\S+", min_df=2, sublinear_tf=True, lowercase=False).fit_transform(raw_txt)]
if not LIGHT: blocks.append(tv(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=300000).fit_transform(P["text"]))
del df, tr, te
G = pd.concat([F, R], axis=1)
num = [c for c in G.columns if c not in CAT and str(G[c].dtype) not in ("object", "category")]
binned = pd.DataFrame({c: pd.qcut(G[c].rank(method="first"), 20, labels=False, duplicates="drop").astype(str) for c in num})
cats = pd.concat([G[[c for c in G.columns if c in CAT or str(G[c].dtype) == "category"]].astype(str), binned], axis=1)
oh = OneHotEncoder(handle_unknown="ignore", min_frequency=2).fit_transform(cats)
X = sp.hstack(blocks + [oh]).tocsr().astype(np.float32); print("X", X.shape, flush=True)
Xtr, Xte = X[:ntr], X[ntr:]
C = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
oof = np.zeros(ntr); tp = np.zeros(X.shape[0] - ntr)
for a, b in folds:
    m = LogisticRegression(C=C, max_iter=3000, solver="liblinear").fit(Xtr[a], y[a])
    oof[b] = m.predict_proba(Xtr[b])[:, 1]; tp += m.predict_proba(Xte)[:, 1] / 5; print("fold", flush=True)
ll = -np.mean(y * np.log(np.clip(oof, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - oof, 1e-6, 1)))
print("a_sparse_raw C", C, "OOF acc", ((oof > .5) == y).mean().round(5), "logloss", round(ll, 5), flush=True)
np.savez(D + "pred_a_sparse_raw.npz", oof=oof, te=tp)
