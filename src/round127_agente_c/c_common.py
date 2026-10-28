"""Utilidades del agente C: datos, folds (5 estratificados, semilla 2026), métricas y evaluación de mezcla con ens_ll8."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
D = os.environ.get("AP2026Q2_DATA_DIR", "/home/claude/data").rstrip("/") + "/"

def load_y():
    return pd.read_pickle(D + "feats.pkl")["y"]

def folds(y):
    return list(StratifiedKFold(5, shuffle=True, random_state=2026).split(np.zeros(len(y)), y))

def ev(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return ((p > .5) == y).mean(), -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

def load_pred(name):
    d = np.load(D + f"pred_{name}.npz"); return d["oof"], d["te"]

def save_pred(name, oof, te):
    assert name.startswith("c_")
    path = D + f"pred_{name}.npz"
    assert not os.path.exists(path), path
    np.savez(path, oof=oof, te=te)

def blend_eval(y, oof, base="ens_ll8", verbose=True):
    """Mezcla con la base: (1) promedio de logits con peso w elegido en 4 folds y aplicado al 5.º;
    (2) logística sobre [logit base, logit modelo] con CV anidada. Devuelve dict de métricas."""
    fo = folds(y); b = load_pred(base)[0]
    Lb, Lm = logit(b), logit(oof)
    ws = np.linspace(0, 1, 41); pw = np.zeros(len(y)); wsel = []
    for a, t in fo:
        lls = [ev(y[a], 1 / (1 + np.exp(-((1 - w) * Lb[a] + w * Lm[a]))))[1] for w in ws]
        w = ws[int(np.argmin(lls))]; wsel.append(w)
        pw[t] = 1 / (1 + np.exp(-((1 - w) * Lb[t] + w * Lm[t])))
    ps = np.zeros(len(y)); coefs = []
    X = np.c_[Lb, Lm]
    for a, t in fo:
        m = LogisticRegression(C=1.0, max_iter=1000).fit(X[a], y[a]); ps[t] = m.predict_proba(X[t])[:, 1]; coefs.append(m.coef_[0])
    r = {"own": ev(y, oof), "base": ev(y, b), "wavg": ev(y, pw), "w": float(np.mean(wsel)), "stack": ev(y, ps), "coef": np.round(np.mean(coefs, 0), 3).tolist(),
         "corr": float(np.corrcoef(Lb, Lm)[0, 1])}
    if verbose:
        print(f"own acc {r['own'][0]:.5f} ll {r['own'][1]:.5f} | base {r['base'][0]:.5f}/{r['base'][1]:.5f} | wavg(w={r['w']:.3f}) {r['wavg'][0]:.5f}/{r['wavg'][1]:.5f} | stack {r['stack'][0]:.5f}/{r['stack'][1]:.5f} coef {r['coef']} | corr logit {r['corr']:.3f}", flush=True)
    return r
