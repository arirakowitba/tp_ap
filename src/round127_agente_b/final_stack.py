"""Ensamble final: logística (C=0,1) sobre logits de los modelos dados.
OOF con CV anidada en los 5 folds compartidos; para test se ajusta el apilador en los 70k OOF completos.
usage: python final_stack.py NAME model1 model2 ..."""
import sys, json, numpy as np
from sklearn.linear_model import LogisticRegression
from common import *
name, models = sys.argv[1], sys.argv[2:]
y = load_y(); fo = folds(y)
L = np.array([logit(load_pred(m)[0]) for m in models]).T; LT = np.array([logit(load_pred(m)[1]) for m in models]).T
oof = np.zeros(len(y))
for a, b in fo: oof[b] = LogisticRegression(C=0.1, max_iter=3000).fit(L[a], y[a]).predict_proba(L[b])[:, 1]
m = LogisticRegression(C=0.1, max_iter=3000).fit(L, y); te = m.predict_proba(LT)[:, 1]
acc, ll = ev(y, oof); print(name, "OOF acc %.5f ll %.5f" % (acc, ll), "w", dict(zip(models, np.round(m.coef_[0], 3))), "b", round(m.intercept_[0], 3))
print("test: frac used %.4f" % (te > .5).mean(), "| oof frac %.4f" % (oof > .5).mean())
save_pred(name, oof, te)
json.dump({"name": name, "models": models, "oof_acc": acc, "logloss": ll, "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0])}, open(D + f"report_{name}.json", "w"))
