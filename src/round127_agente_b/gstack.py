"""Apilado logístico con compuertas por segmento: logits × indicadores (vendedor sin avisos en otros folds,
vendedor con 1–4, buying_mode ≠ buy_it_now). OOF con CV anidada; test con apilador ajustado en los 70k.
Los indicadores de test usan el conteo en todo train (tecnt_seller de frame.py, escalado 0,8 → se recalcula aquí).
usage: python gstack.py NAME model1 model2 ... [--save]"""
import sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from common import *
from frame import build
args = [a for a in sys.argv[1:] if not a.startswith("--")]; name, models = args[0], args[1:]
o = build(); F, y, ntr = o["F"], o["y"], o["ntr"]; fo = folds(y)
P = pd.read_pickle(D + "feats.pkl"); s = P["F"].seller_id.astype(str).values
# conteo de avisos del vendedor en train fuera del fold (train) / en todo train (test)
cnt = np.zeros(len(F)); vc_all = pd.Series(s[:ntr]).value_counts()
for a, b in fo:
    vc = pd.Series(s[:ntr][a]).value_counts(); cnt[b] = pd.Series(s[:ntr][b]).map(vc).fillna(0).values
cnt[ntr:] = pd.Series(s[ntr:]).map(vc_all).fillna(0).values
bm = (F.buying_mode.astype(str).values != "buy_it_now").astype(float)
G = np.c_[(cnt == 0), (cnt > 0) & (cnt < 5), bm].astype(float)
L = np.r_[np.array([logit(load_pred(m)[0]) for m in models]).T, np.array([logit(load_pred(m)[1]) for m in models]).T]
X = np.c_[L, *[L * G[:, [g]] for g in range(G.shape[1])], G]
Xtr, Xte = X[:ntr], X[ntr:]
for C in [0.03, 0.1, 0.3]:
    oof = np.zeros(ntr)
    for a, b in fo: oof[b] = LogisticRegression(C=C, max_iter=5000).fit(Xtr[a], y[a]).predict_proba(Xtr[b])[:, 1]
    print(f"gstack C={C} acc %.5f ll %.5f" % ev(y, oof))
    if C == 0.1 and "--save" in sys.argv:
        te = LogisticRegression(C=C, max_iter=5000).fit(Xtr, y).predict_proba(Xte)[:, 1]; save_pred(name, oof, te)
        print("guardado", name, "test frac used %.4f" % (te > .5).mean())
