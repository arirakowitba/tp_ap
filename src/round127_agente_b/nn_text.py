"""Modelo neuronal de texto entrenado desde cero (torch CPU, 1 hilo), OOF en los 5 folds compartidos.
Entradas: título (CNN de caracteres + bolsa de palabras/bigramas embebidos), garantía (bolsa de palabras),
nombres de atributos y tokens categóricos (categoría, tipo de publicación, modo de compra, envío, estado).
usage: python nn_text.py NAME [VARIANT]   VARIANT: full (default) | title (sólo título)"""
import sys, os, time, re, unicodedata, numpy as np, pandas as pd, torch, torch.nn as nn
from common import *
torch.set_num_threads(1); torch.set_num_interop_threads(1)
name = sys.argv[1]; VAR = sys.argv[2] if len(sys.argv) > 2 else "full"
SEED = int(os.environ.get("SEED", "7")); EPOCHS = int(os.environ.get("EPOCHS", "6"))
P = pd.read_pickle(D + "feats.pkl"); y = P["y"]; ntr = len(y); fo = folds(y); F = P["F"]
def norm(s):
    s = unicodedata.normalize("NFKD", str(s).lower()); return "".join(c for c in s if not unicodedata.combining(c))
title = [norm(t) for t in P["text"]]; warr = [norm(w) for w in P["warranty"]]
tok = lambda s: re.findall(r"[a-z0-9]+", s)
# --- bolsa de tokens (palabras + bigramas de título, palabras de garantía, atributos, categóricas) ---
docs = []
for i in range(len(title)):
    tw = tok(title[i]); d = ["t:" + w for w in tw] + ["tb:" + a + "_" + b for a, b in zip(tw, tw[1:])]
    if VAR == "full":
        ww = tok(warr[i])[:40]; d += ["w:" + w for w in ww] + ["wb:" + a + "_" + b for a, b in zip(ww, ww[1:])]
        d += ["w:__null__"] if not warr[i] else []
        d += ["a:" + a for a in str(P["attr"][i]).split()]
        d += [f"c:{c}={F[c].iat[i]}" for c in ["category_id", "listing_type_id", "buying_mode", "ship_mode", "status", "sub_status", "state"]]
    docs.append(d)
from collections import Counter
cnt = Counter(t for d in docs for t in set(d))
vocab = {t: i + 1 for i, t in enumerate(sorted(t for t, c in cnt.items() if c >= 2))}
MAXT = 96
B = np.zeros((len(docs), MAXT), np.int64)
for i, d in enumerate(docs):
    ids = [vocab[t] for t in d if t in vocab][:MAXT]; B[i, :len(ids)] = ids
# --- caracteres del título ---
chars = Counter(c for t in title for c in t); cv = {c: i + 1 for i, c in enumerate(sorted(c for c, n in chars.items() if n >= 5))}
MAXC = 80
C = np.zeros((len(title), MAXC), np.int64)
for i, t in enumerate(title):
    ids = [cv.get(c, len(cv) + 1) for c in t[:MAXC]]; C[i, :len(ids)] = ids
print("vocab", len(vocab), "chars", len(cv), flush=True)

class Net(nn.Module):
    def __init__(s, nv, nc, E=64):
        super().__init__()
        s.emb = nn.EmbeddingBag(nv + 1, E, mode="mean", padding_idx=0)
        s.cemb = nn.Embedding(nc + 2, 32, padding_idx=0)
        s.convs = nn.ModuleList([nn.Conv1d(32, 64, k, padding=k // 2) for k in (3, 5)])
        s.head = nn.Sequential(nn.Linear(E + 128, 96), nn.ReLU(), nn.Dropout(0.3), nn.Linear(96, 1))
    def forward(s, b, c):
        e = s.emb(b)
        x = s.cemb(c).transpose(1, 2); m = (c > 0).unsqueeze(1).float()
        h = torch.cat([(torch.relu(cv(x)) * m - (1 - m) * 1e4).amax(2) for cv in s.convs], 1)
        return s.head(torch.cat([e, h], 1)).squeeze(1)

def predict(model, idx):
    model.eval(); out = []
    with torch.no_grad():
        for st in range(0, len(idx), 2048):
            j = idx[st:st + 2048]; out.append(torch.sigmoid(model(torch.from_numpy(B[j]), torch.from_numpy(C[j]))).numpy())
    return np.concatenate(out)

oof = np.zeros(ntr); te = np.zeros(len(title) - ntr); te_idx = np.arange(ntr, len(title)); t0 = time.time()
for k, (a, b) in enumerate(fo):
    torch.manual_seed(SEED + k); rng = np.random.default_rng(SEED + k)
    model = Net(len(vocab), len(cv)); opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    # 10 % de los folds de entrenamiento como validación interna para elegir la época (no se mira el fold b)
    perm = rng.permutation(a); nv = len(a) // 10; va, tr_ = perm[:nv], perm[nv:]
    yt = torch.from_numpy(y.astype(np.float32)); best = (9, None, 0); steps = EPOCHS * (len(tr_) // 256 + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    for ep in range(EPOCHS):
        model.train(); rng.shuffle(tr_)
        for st in range(0, len(tr_), 256):
            j = tr_[st:st + 256]
            loss = nn.functional.binary_cross_entropy_with_logits(model(torch.from_numpy(B[j]), torch.from_numpy(C[j])), yt[j])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        pv = predict(model, va); _, ll = ev(y[va], pv)
        if ll < best[0]: best = (ll, {kk: v.clone() for kk, v in model.state_dict().items()}, ep)
    model.load_state_dict(best[1])
    oof[b] = predict(model, b); te += predict(model, te_idx) / 5
    print(f"{name} fold {k} acc {((oof[b] > .5) == y[b]).mean():.4f} best_ep {best[2]} t {time.time()-t0:.0f}s", flush=True)
acc, ll = ev(y, oof); print(f"{name} OOF acc {acc:.5f} logloss {ll:.5f}", flush=True)
save_pred(name, oof, te)
