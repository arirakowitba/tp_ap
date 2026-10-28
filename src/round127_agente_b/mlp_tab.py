"""MLP (torch, 1 hilo) sobre la tabla numérica (frame.py, sin sellerother) estandarizada con rangos→normal,
+ embeddings de categóricas de baja cardinalidad. OOF en los 5 folds compartidos; época elegida en 10 % interno.
usage: python mlp_tab.py NAME"""
import sys, os, time, numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.special import erfinv
from common import *
from frame import build
torch.set_num_threads(1); torch.set_num_interop_threads(1)
name = sys.argv[1]; SEED = int(os.environ.get("SEED", "3")); EPOCHS = int(os.environ.get("EPOCHS", "20"))
o = build(); F, y, ntr, CAT = o["F"].copy(), o["y"], o["ntr"], o["CAT"]; fo = folds(y)
F = F.drop(columns=[c for c in F.columns if c.startswith("stk_sellerother_")])
lowc = [c for c in CAT if F[c].nunique() < 100]
num = [c for c in F.columns if c not in CAT]
Xn = np.zeros((len(F), len(num) * 2), np.float32)
for j, c in enumerate(num):
    v = F[c].astype(float).values; na = np.isnan(v)
    r = pd.Series(np.where(na, np.nanmedian(v), v)).rank(pct=True).values
    Xn[:, j] = erfinv(np.clip(r * 2 - 1, -0.999, 0.999)); Xn[:, len(num) + j] = na
Xn = Xn[:, np.r_[np.arange(len(num)), len(num) + np.where(Xn[:, len(num):].any(0))[0]]]
Xc = np.stack([pd.factorize(F[c].astype(str))[0] for c in lowc], 1).astype(np.int64); card = Xc.max(0) + 1
print("num", Xn.shape, "cats", lowc, flush=True)
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.embs = nn.ModuleList([nn.Embedding(int(k), 8) for k in card])
        d = Xn.shape[1] + 8 * len(card)
        s.f = nn.Sequential(nn.Linear(d, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.25), nn.Linear(256, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.25), nn.Linear(128, 1))
    def forward(s, xn, xc): return s.f(torch.cat([xn] + [e(xc[:, i]) for i, e in enumerate(s.embs)], 1)).squeeze(1)
XN, XC, YT = torch.from_numpy(Xn), torch.from_numpy(Xc), torch.from_numpy(y.astype(np.float32))
def pred(m, idx):
    m.eval()
    with torch.no_grad(): return np.concatenate([torch.sigmoid(m(XN[idx[i:i + 4096]], XC[idx[i:i + 4096]])).numpy() for i in range(0, len(idx), 4096)])
oof = np.zeros(ntr); te = np.zeros(len(F) - ntr); tidx = np.arange(ntr, len(F)); t0 = time.time()
for k, (a, b) in enumerate(fo):
    torch.manual_seed(SEED + k); rng = np.random.default_rng(SEED + k); perm = rng.permutation(a); va, tr_ = perm[:len(a) // 10], perm[len(a) // 10:]
    m = Net(); opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=EPOCHS * (len(tr_) // 512 + 1)); best = (9, None, 0)
    for ep in range(EPOCHS):
        m.train(); rng.shuffle(tr_)
        for i in range(0, len(tr_), 512):
            j = tr_[i:i + 512]; loss = nn.functional.binary_cross_entropy_with_logits(m(XN[j], XC[j]), YT[j])
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        ll = ev(y[va], pred(m, va))[1]
        if ll < best[0]: best = (ll, {kk: v.clone() for kk, v in m.state_dict().items()}, ep)
    m.load_state_dict(best[1]); oof[b] = pred(m, b); te += pred(m, tidx) / 5
    print(f"{name} fold {k} acc {((oof[b] > .5) == y[b]).mean():.4f} best_ep {best[2]} t {time.time()-t0:.0f}s", flush=True)
acc, ll = ev(y, oof); print(f"{name} OOF acc {acc:.5f} logloss {ll:.5f}", flush=True); save_pred(name, oof, te)
