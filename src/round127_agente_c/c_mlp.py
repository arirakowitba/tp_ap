"""MLP tabular (torch, 1 hilo) sobre la tabla de train_gbm.py: numéricas con QuantileTransformer->normal + indicador de NaN,
embeddings para categóricas (vendedor/categoría/ciudad con mínimo de frecuencia en train+test), y opcionalmente una parte
'wide' lineal sobre TF-IDF del título (palabra+char). Mismos 5 folds; early stopping por logloss en un 10 % interno del fold de train.
usage: python c_mlp.py NAME [--wide] [--hidden 512] [--epochs 30] [--seeds 1] [--drop 0.3] [--nostk]"""
import os, sys, time, argparse
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np, pandas as pd, torch, torch.nn as nn
from c_common import *
torch.set_num_threads(1)
ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--wide", action="store_true"); ap.add_argument("--hidden", type=int, default=512)
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--seeds", type=int, default=1); ap.add_argument("--drop", type=float, default=0.3)
ap.add_argument("--nostk", action="store_true"); ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--wd", type=float, default=1e-5)
ap.add_argument("--emb", type=int, default=16); ap.add_argument("--mincnt", type=int, default=2); ap.add_argument("--bs", type=int, default=512)
A = ap.parse_args()
B = pd.read_pickle(D + "b_frame_sparse_all_nbsvm_knn_title.pkl"); F, y, ntr, CAT = B["F"], B["y"], B["ntr"], B["CAT"]
num = [c for c in F.columns if c not in CAT]
if A.nostk: num = [c for c in num if not c.startswith(("stk_", "knn_"))]
from sklearn.preprocessing import QuantileTransformer
Xn = F[num].astype(np.float64).values; nanm = np.isnan(Xn); nancols = nanm.any(0)
Xq = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=200000, random_state=0).fit_transform(Xn)  # no supervisado, train+test
Xq = np.nan_to_num(Xq, nan=0.0); Xq = np.c_[Xq, nanm[:, nancols].astype(float)].astype(np.float32)
codes, cards = [], []
for c in CAT:
    s = F[c].astype(str); vc = s.value_counts(); s = s.where(s.map(vc) >= A.mincnt, "__rare__")
    cd = pd.factorize(s)[0]; codes.append(cd); cards.append(cd.max() + 1)
Xc = np.stack(codes, 1).astype(np.int64)
Xw = None
if A.wide:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import scipy.sparse as sp
    P = pd.read_pickle(D + "feats.pkl"); t = P["text"]
    Xw = sp.hstack([TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, strip_accents="unicode", max_features=100000).fit_transform(t),
                    TfidfVectorizer(ngram_range=(1, 1), min_df=3, sublinear_tf=True, strip_accents="unicode").fit_transform(P["warranty"])]).tocsr().astype(np.float32)
print("num", Xq.shape, "cards", cards, "wide", None if Xw is None else Xw.shape, flush=True)

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.embs = nn.ModuleList([nn.Embedding(n, min(A.emb, max(2, n // 2))) for n in cards])
        d = Xq.shape[1] + sum(e.embedding_dim for e in s.embs); H = A.hidden
        s.mlp = nn.Sequential(nn.Linear(d, H), nn.BatchNorm1d(H), nn.SiLU(), nn.Dropout(A.drop), nn.Linear(H, H // 2), nn.BatchNorm1d(H // 2), nn.SiLU(), nn.Dropout(A.drop), nn.Linear(H // 2, 1))
        s.edrop = nn.Dropout(0.1)
        if Xw is not None: s.wide = nn.EmbeddingBag(Xw.shape[1], 1, mode="sum")
    def forward(s, xn, xc, wi=None, wo=None, ww=None):
        e = torch.cat([s.edrop(m(xc[:, j])) for j, m in enumerate(s.embs)], 1)
        o = s.mlp(torch.cat([xn, e], 1))[:, 0]
        if Xw is not None: o = o + s.wide(wi, wo, per_sample_weights=ww)[:, 0]
        return o

def wbatch(idx):
    M = Xw[idx]; return torch.from_numpy(M.indices.astype(np.int64)), torch.from_numpy(M.indptr[:-1].astype(np.int64)), torch.from_numpy(M.data)
def predict(m, idx):
    m.eval(); out = []
    with torch.no_grad():
        for st in range(0, len(idx), 4096):
            ii = idx[st:st + 4096]; args = wbatch(ii) if Xw is not None else ()
            out.append(torch.sigmoid(m(torch.from_numpy(Xq[ii]), torch.from_numpy(Xc[ii]), *args)).numpy())
    return np.concatenate(out)
def fit(tr_idx, va_idx, seed):
    torch.manual_seed(seed); np.random.seed(seed); m = Net()
    opt = torch.optim.AdamW(m.parameters(), lr=A.lr, weight_decay=A.wd); lossf = nn.BCEWithLogitsLoss()
    nb = int(np.ceil(len(tr_idx) / A.bs)); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=A.lr, total_steps=A.epochs * nb, pct_start=0.2)
    yt = torch.from_numpy(y.astype(np.float32)); best, bstate, bad = 9, None, 0
    for ep in range(A.epochs):
        m.train(); perm = np.random.permutation(tr_idx)
        for st in range(0, len(perm), A.bs):
            ii = perm[st:st + A.bs]; args = wbatch(ii) if Xw is not None else ()
            opt.zero_grad(); l = lossf(m(torch.from_numpy(Xq[ii]), torch.from_numpy(Xc[ii]), *args), yt[ii]); l.backward(); opt.step(); sched.step()
        ll = ev(y[va_idx], predict(m, va_idx))[1]
        if ll < best: best, bstate, bad, bep = ll, {k: v.clone() for k, v in m.state_dict().items()}, 0, ep
        else: bad += 1
        if bad >= 6: break
    m.load_state_dict(bstate); return m, bep, best

fo = folds(y); oof = np.zeros(ntr); te = np.zeros(len(F) - ntr); t0 = time.time(); teidx = np.arange(ntr, len(F))
for k, (a, b) in enumerate(fo):
    for s in range(A.seeds):
        rng = np.random.RandomState(100 + k + 10 * s); va = rng.rand(len(a)) < 0.1
        m, bep, bll = fit(a[~va], a[va], 1000 * s + k)
        oof[b] += predict(m, b) / A.seeds; te += predict(m, teidx) / A.seeds / 5
        print(f"{A.name} fold {k} seed {s} ep {bep} inner ll {bll:.4f} t {time.time()-t0:.0f}s", flush=True)
    print(f"{A.name} fold {k} acc {((oof[b]>.5)==y[b]).mean():.4f} ll {ev(y[b], oof[b])[1]:.4f}", flush=True)
print(A.name, "OOF acc %.5f ll %.5f" % ev(y, oof), flush=True); blend_eval(y, oof); save_pred(A.name, oof, te)
