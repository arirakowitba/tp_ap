"""OOF k-nearest-neighbour label features on title TF-IDF (cosine).
Train rows in fold b only see neighbours from the other folds; test rows see all train rows
(neighbour labels then come from 5/5 of train, i.e. slightly more neighbours, same as any OOF encoding)."""
import os, numpy as np, pandas as pd, scipy.sparse as sp, time
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

D = os.environ.get("AP2026Q2_DATA_DIR", "data/ap_2026_q_2").rstrip("/") + "/"
P = pd.read_pickle(D + "feats.pkl"); y = P["y"].astype(float); ntr = len(y)
seller = pd.factorize(P["F"]["seller_id"].astype(str))[0]
folds = list(StratifiedKFold(5, shuffle=True, random_state=2026).split(np.zeros(ntr), y))
t = P["text"]
Xw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode").fit_transform(t)
Xc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, strip_accents="unicode", max_features=200000).fit_transform(t)
X = normalize(sp.hstack([Xw, Xc]).tocsr()).astype(np.float32)
K = 30
def topk(Q, I, idx_rows):
    out_i = np.zeros((Q.shape[0], K), np.int64); out_s = np.zeros((Q.shape[0], K), np.float32)
    IT = I.T.tocsc()
    for st in range(0, Q.shape[0], 1000):
        M = (Q[st:st + 1000] @ IT).toarray()
        part = np.argpartition(-M, K, axis=1)[:, :K]
        ss = np.take_along_axis(M, part, 1); o = np.argsort(-ss, 1)
        out_i[st:st + 1000] = idx_rows[np.take_along_axis(part, o, 1)]; out_s[st:st + 1000] = np.take_along_axis(ss, o, 1)
    return out_i, out_s

NI = np.zeros((len(t), K), np.int64); NS = np.zeros((len(t), K), np.float32)
t0 = time.time()
for a, b in folds:
    NI[b], NS[b] = topk(X[b], X[a], a); print("fold done", round(time.time() - t0), flush=True)
te_idx = np.arange(ntr, len(t)); allr = np.arange(ntr)
NI[te_idx], NS[te_idx] = topk(X[te_idx], X[:ntr], allr); print("test done", round(time.time() - t0), flush=True)

np.savez(D + "knn_title_neighbors.npz", NI=NI, NS=NS)
L = y[NI]                     # neighbour labels
feat = {}
for k in (1, 5, 10, 30):
    w = NS[:, :k]
    feat[f"knn_mean{k}"] = L[:, :k].mean(1)
    feat[f"knn_wmean{k}"] = (L[:, :k] * w).sum(1) / (w.sum(1) + 1e-6)
feat["knn_sim1"] = NS[:, 0]; feat["knn_sim5"] = NS[:, 4]; feat["knn_sim30"] = NS[:, 29]
same = seller[np.minimum(NI, ntr - 1)] == seller[:, None]
feat["knn_same_seller_frac"] = same.mean(1)
feat["knn_same_seller_label"] = np.where(same.sum(1) > 0, (L * same).sum(1) / np.maximum(same.sum(1), 1), np.nan)
feat["knn_other_seller_wmean"] = np.where((~same).sum(1) > 0, (L * NS * ~same).sum(1) / ((NS * ~same).sum(1) + 1e-6), np.nan)
for s in (0.9, 0.7, 0.5):
    msk = NS >= s
    feat[f"knn_cnt_sim{s}"] = msk.sum(1); feat[f"knn_mean_sim{s}"] = np.where(msk.sum(1) > 0, (L * msk).sum(1) / np.maximum(msk.sum(1), 1), np.nan)
pd.DataFrame(feat).to_pickle(D + "knn_title.pkl")
print("knn_mean10 OOF acc", ((feat["knn_wmean10"][:ntr] > .5) == y).mean().round(4))
