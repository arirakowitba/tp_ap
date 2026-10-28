"""Apilado logístico sobre logits con CV anidada (mismos 5 folds). usage: python c_stack.py base m1 m2 ... [--C 1] [--save NAME]"""
import sys, numpy as np
from c_common import *
args = sys.argv[1:]; save = None; C = 1.0
if "--save" in args: i = args.index("--save"); save = args[i + 1]; del args[i:i + 2]
if "--C" in args: i = args.index("--C"); C = float(args[i + 1]); del args[i:i + 2]
y = load_y(); fo = folds(y)
O = np.array([load_pred(m)[0] for m in args]).T; T = np.array([load_pred(m)[1] for m in args]).T
L, LT = logit(O), logit(T); oof = np.zeros(len(y)); te = np.zeros(len(T)); ws = []
for a, b in fo:
    m = LogisticRegression(C=C, max_iter=3000).fit(L[a], y[a]); oof[b] = m.predict_proba(L[b])[:, 1]; te += m.predict_proba(LT)[:, 1] / 5; ws.append(m.coef_[0])
print("stack", " ".join(args), "acc %.5f ll %.5f" % ev(y, oof), "w", np.round(np.mean(ws, 0), 2))
if save: save_pred(save, oof, te)
