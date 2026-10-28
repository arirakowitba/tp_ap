"""Evalúa mezclas sobre OOF existentes: promedio simple, promedio de logits, rank-average,
y logística sobre logits con CV anidada (los mismos 5 folds: el apilador se ajusta en 4 folds y predice el 5.º).
usage: python stack_eval.py NAME model1 model2 ... [--save]"""
import sys, numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from common import *
args = [a for a in sys.argv[1:] if not a.startswith("--")]; save = "--save" in sys.argv
name, models = args[0], args[1:]
y = load_y(); fo = folds(y)
O = np.array([load_pred(m)[0] for m in models]).T; T = np.array([load_pred(m)[1] for m in models]).T
for m, j in zip(models, range(len(models))): print(f"  {m:28s} acc {ev(y, O[:, j])[0]:.5f} ll {ev(y, O[:, j])[1]:.5f}")
print("mean prob   acc %.5f ll %.5f" % ev(y, O.mean(1)))
pl = 1 / (1 + np.exp(-logit(O).mean(1))); print("mean logit  acc %.5f ll %.5f" % ev(y, pl))
r = np.mean([rankdata(O[:, j]) / len(y) for j in range(O.shape[1])], 0); thr = np.quantile(r, 1 - y.mean())
print("rank avg (umbral = prevalencia) acc %.5f" % ((r > thr) == y).mean())
L, LT = logit(O), logit(T); oof = np.zeros(len(y)); te = np.zeros(len(T))
for C in [0.01, 0.1, 1.0]:
    oof = np.zeros(len(y)); te = np.zeros(len(T)); ws = []
    for a, b in fo:
        m = LogisticRegression(C=C, max_iter=2000).fit(L[a], y[a]); oof[b] = m.predict_proba(L[b])[:, 1]; te += m.predict_proba(LT)[:, 1] / 5; ws.append(m.coef_[0])
    print(f"logreg C={C} acc %.5f ll %.5f" % ev(y, oof), "w", np.round(np.mean(ws, 0), 2))
# umbral ajustado anidado sobre promedio de logits
pr = np.zeros(len(y), bool)
for a, b in fo:
    ts = np.linspace(.4, .6, 81); best = ts[np.argmax([((pl[a] > t) == y[a]).mean() for t in ts])]; pr[b] = pl[b] > best
print("umbral anidado acc %.5f" % (pr == y).mean())
if save:
    save_pred(name, oof, te); print("guardado", name)
