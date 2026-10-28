"""Apilado logístico con compuertas: logits de cada modelo interactuados con indicadores de segmento
(vendedor sin historial en train-OOF, vendedor con 1-10 avisos, buying_mode != buy_it_now). CV anidada en los mismos 5 folds.
usage: python c_gstack.py m1 m2 ... [--save NAME] [--C 1]"""
import sys, numpy as np, pandas as pd
from c_common import *
args = sys.argv[1:]; save = None; C = 1.0
if "--save" in args: i = args.index("--save"); save = args[i + 1]; del args[i:i + 2]
if "--C" in args: i = args.index("--C"); C = float(args[i + 1]); del args[i:i + 2]
B = pd.read_pickle(D + "b_frame_sparse_all_nbsvm_knn_title.pkl"); F, y, ntr = B["F"], B["y"], B["ntr"]
cnt = F.tecnt_seller.values
seg = np.c_[cnt == 0, (cnt > 0) & (cnt <= 10), F.buying_mode.astype(str).values != "buy_it_now"].astype(float)
fo = folds(y)
O = np.r_[np.array([load_pred(m)[0] for m in args]).T, np.array([load_pred(m)[1] for m in args]).T]
L = logit(O); X = np.c_[L, seg] if True else L
for j in range(seg.shape[1]): X = np.c_[X, L * seg[:, [j]]]
Xtr, Xte = X[:ntr], X[ntr:]; oof = np.zeros(ntr); te = np.zeros(len(Xte))
for a, b in fo:
    m = LogisticRegression(C=C, max_iter=5000).fit(Xtr[a], y[a]); oof[b] = m.predict_proba(Xtr[b])[:, 1]; te += m.predict_proba(Xte)[:, 1] / 5
print("gstack", " ".join(args), "acc %.5f ll %.5f" % ev(y, oof))
if save: save_pred(save, oof, te)
