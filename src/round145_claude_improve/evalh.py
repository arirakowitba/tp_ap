"""Metrics helpers. holdout: labels from holdout_labels.csv, seen = seller_id of holdout row present in the 56k train."""
import numpy as np, pandas as pd, json
from sklearn.metrics import log_loss
H = "/home/claude/work/holdout"; H2 = "/home/claude/work/holdout2"
W_SEEN = 0.7162

def holdout_ctx():
    lab = pd.read_csv(f"{H}/holdout_labels.csv").sort_values("ID")
    y = (lab.condition.values == "used").astype(int)
    d = pd.read_pickle(f"{H2}/feats_holdout.pkl")
    seen = np.isin(d["eb"].seller_id.values, d["ea"].seller_id.values)
    return y, seen

def inner_ctx():
    d = pd.read_pickle(f"{H2}/feats_holdout.pkl")
    iv = np.load(f"{H2}/inner/inner_eval_rows.npy"); y = np.load(f"{H2}/inner/inner_eval_y.npy")
    s = d["ea"].seller_id.values; tr = np.setdiff1d(np.arange(len(s)), iv)
    return y, np.isin(s[iv], s[tr])

def metrics(p, y, seen, ref=None, thr=0.5):
    q = p >= thr; ok = q == y
    r = dict(acc=float(ok.mean()), seen_acc=float(ok[seen].mean()), new_acc=float(ok[~seen].mean()), seen_frac=float(seen.mean()),
             logloss=float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))))
    r["reweighted"] = W_SEEN * r["seen_acc"] + (1 - W_SEEN) * r["new_acc"]
    if ref is not None:
        rok = (ref >= 0.5) == y
        r.update(fixes=int((ok & ~rok).sum()), breaks=int((~ok & rok).sum()))
        r["net"] = r["fixes"] - r["breaks"]
        r.update(fixes_seen=int((ok & ~rok & seen).sum()), breaks_seen=int((~ok & rok & seen).sum()),
                 fixes_new=int((ok & ~rok & ~seen).sum()), breaks_new=int((~ok & rok & ~seen).sum()))
    return r

def fmt(name, r):
    s = f"{name:32s} acc {r['acc']:.5f} seen {r['seen_acc']:.5f} new {r['new_acc']:.5f} rew {r['reweighted']:.5f} ll {r['logloss']:.5f}"
    if "net" in r: s += f"  +{r['fixes']}/-{r['breaks']} net {r['net']:+d} (seen {r['fixes_seen']-r['breaks_seen']:+d}, new {r['fixes_new']-r['breaks_new']:+d})"
    return s
