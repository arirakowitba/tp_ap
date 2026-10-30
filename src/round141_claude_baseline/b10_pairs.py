"""Paired comparison of B10 (10-Oct ensemble refit on dev=70k-confirm) against Antigravity models on the same confirm IDs.
usage: python b10_pairs.py B10_DIR AG_DIR SEED [SEED...]  -> prints JSON, writes B10_DIR/pairs_summary.json"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
b10, ag = Path(sys.argv[1]), Path(sys.argv[2]); seeds = [int(s) for s in sys.argv[3:]]
def ll(y, p): p = np.clip(p, 1e-6, 1 - 1e-6); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
def boot(y, a, b, sellers, n=2000, seed=42):
    d = (b == y).astype(int) - (a == y).astype(int)
    s = pd.Series(d).groupby(sellers).sum(); v = s.values; rng = np.random.default_rng(seed)
    bs = np.array([v[rng.integers(0, len(v), len(v))].sum() for _ in range(n)])
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], float((bs > 0).mean())
out = {}
for s in seeds:
    p = pd.read_csv(b10 / f"seed{s}" / f"paired_b10_vs_A_hist_seed{s}.csv.gz")
    a = pd.read_csv(ag / f"paired_all_{s}.csv.gz")
    m = p.merge(a[["row_idx", "p_CB_Bag3_hist", "p_Blend075_Bag3_hist", "y_true", "is_seen"]], on="row_idx", suffixes=("", "_ag"))
    assert len(m) == len(p) == len(a) and (m.y_used == m.y_true).all() and (m.is_seen == m.is_seen_ag).all()
    y = m.y_used.values; seen = m.is_seen.values == 1; sel = m.seller_id.astype(str).values
    models = {"B10": m.p_b10.values, "A_hist": m.p_A_hist.values, "CB_Bag3": m.p_CB_Bag3_hist.values, "Blend075_Bag3": m.p_Blend075_Bag3_hist.values}
    r = {"n": len(m), "n_seen": int(seen.sum()), "n_new": int((~seen).sum()), "metrics@0.5": {}, "pairs_B10_vs": {}}
    for k, q in models.items():
        pr = q >= .5
        r["metrics@0.5"][k] = {"acc": round(float((pr == y).mean()), 5), "acc_seen": round(float((pr == y)[seen].mean()), 5),
                               "acc_new": round(float((pr == y)[~seen].mean()), 5), "logloss": round(ll(y, q), 5),
                               "ll_seen": round(ll(y[seen], q[seen]), 5), "ll_new": round(ll(y[~seen], q[~seen]), 5)}
    bp = models["B10"] >= .5
    for k in ["A_hist", "CB_Bag3", "Blend075_Bag3"]:
        ap = models[k] >= .5
        fx = (bp == y) & (ap != y); br = (bp != y) & (ap == y)
        ci, pp = boot(y, ap, bp, sel)
        r["pairs_B10_vs"][k] = {"fixes": int(fx.sum()), "breaks": int(br.sum()), "net": int(fx.sum() - br.sum()),
                                "net_seen": int(fx[seen].sum() - br[seen].sum()), "net_new": int(fx[~seen].sum() - br[~seen].sum()),
                                "seller_boot_ci95": ci, "P(net>0)": round(pp, 3)}
    out[s] = r
(b10 / "pairs_summary.json").write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))
