"""Level-2 blend of OOF predictions (logistic regression on logits, nested CV on the shared folds).
usage: python blend.py NAME model1 model2 ...  -> pred_NAME.npz + report"""
import os, sys, json, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

D = os.environ.get("AP2026Q2_DATA_DIR", "data/ap_2026_q_2").rstrip("/") + "/"
name, models = sys.argv[1], sys.argv[2:]
y = pd.read_pickle(D + "feats.pkl")["y"]; ntr = len(y)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
O = np.column_stack([lg(np.load(D + f"pred_{m}.npz")["oof"]) for m in models])
T = np.column_stack([lg(np.load(D + f"pred_{m}.npz")["te"]) for m in models])
folds = list(StratifiedKFold(5, shuffle=True, random_state=7).split(O, y))   # different seed from level 1
oof = np.zeros(ntr)
for a, b in folds:
    m = LogisticRegression(C=1.0, max_iter=1000).fit(O[a], y[a]); oof[b] = m.predict_proba(O[b])[:, 1]
m = LogisticRegression(C=1.0, max_iter=1000).fit(O, y); te = m.predict_proba(T)[:, 1]
acc = ((oof > .5) == y).mean()
for mm in models: print(mm, round(((np.load(D + f"pred_{mm}.npz")["oof"] > .5) == y).mean(), 5))
print(name, "blend OOF acc", round(acc, 5), "weights", dict(zip(models, m.coef_[0].round(3))))
np.savez(D + f"pred_{name}.npz", oof=oof, te=te)
json.dump({"name": name, "models": models, "oof_acc": acc, "weights": m.coef_[0].tolist()}, open(D + f"report_{name}.json", "w"))
