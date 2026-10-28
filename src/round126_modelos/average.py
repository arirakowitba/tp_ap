"""Simple probability average of several models (reduces test-prediction variance across seeds/depths).
usage: python average.py NAME model1 model2 ..."""
import os, sys, json, numpy as np, pandas as pd
D = os.environ.get("AP2026Q2_DATA_DIR", "data/ap_2026_q_2").rstrip("/") + "/"
name, models = sys.argv[1], sys.argv[2:]
y = pd.read_pickle(D + "feats.pkl")["y"]
oof = np.mean([np.load(D + f"pred_{m}.npz")["oof"] for m in models], 0)
te = np.mean([np.load(D + f"pred_{m}.npz")["te"] for m in models], 0)
acc = ((oof > .5) == y).mean()
ll = -np.mean(y * np.log(np.clip(oof, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - oof, 1e-6, 1)))
print(name, "OOF acc", round(acc, 5), "logloss", round(ll, 5))
np.savez(D + f"pred_{name}.npz", oof=oof, te=te)
json.dump({"name": name, "models": models, "oof_acc": acc, "logloss": ll}, open(D + f"report_{name}.json", "w"))
