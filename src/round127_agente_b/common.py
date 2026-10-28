"""Utilidades compartidas del agente B: datos, folds (5 estratificados, semilla 2026) y métricas."""
import os, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
D = os.environ.get("AP2026Q2_DATA_DIR", "/home/claude/data").rstrip("/") + "/"

def load_y():
    return pd.read_pickle(D + "feats.pkl")["y"]

def folds(y):
    return list(StratifiedKFold(5, shuffle=True, random_state=2026).split(np.zeros(len(y)), y))

def ev(y, p):
    acc = ((p > .5) == y).mean()
    ll = -np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
    return acc, ll

def load_pred(name):
    d = np.load(D + f"pred_{name}.npz"); return d["oof"], d["te"]

def save_pred(name, oof, te):
    assert name.startswith("b_")
    np.savez(D + f"pred_{name}.npz", oof=oof, te=te)

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
