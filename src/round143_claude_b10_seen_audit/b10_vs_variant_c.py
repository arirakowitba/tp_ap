"""Paired B10 (fit-only, 10-Oct ensemble) vs Antigravity variant_c (CatBoost + seller_id categorical) on the same
tune/confirm rows of the 3 seller-group cuts. Confirm is REUSED (development). Prints acc/logloss seen/new, paired net with
seller bootstrap, and, on disagreement rows, who is right by segment (tune and confirm separately).
usage: python b10_vs_variant_c.py B10_TC_DIR B10_FIT_DIR AG_DIR"""
import sys, os, json, numpy as np, pandas as pd
tcd, fitd, ag = sys.argv[1:4]
def ll(y, p): p = np.clip(p, 1e-6, 1 - 1e-6); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
out = {}
for s in [2026, 314159, 20261030]:
    vc = pd.read_csv(f"{ag}/vc_tc_{s}.csv.gz")
    co = pd.read_csv(f"{fitd}/seed{s}_fit/paired_b10_vs_A_hist_seed{s}.csv.gz")[["row_idx", "seller_id", "is_seen", "y_used", "p_b10"]]
    parts = {"confirm": co}
    tcf = f"{tcd}/seed{s}_fit_tc/b10_tune_confirm_seed{s}.csv.gz"
    if os.path.exists(tcf):
        t = pd.read_csv(tcf); t = t[t.split == "tune"][["row_idx", "p_b10"]]
        tu = pd.read_csv(f"{ag}/preds_A_hist_seed{s}_tune.csv.gz")[["row_idx", "seller_id", "is_seen", "y_used"]]
        parts["tune"] = tu.merge(t, on="row_idx")
    r = {}
    for split, b in parts.items():
        m = b.merge(vc[vc.split == split][["row_idx", "y_true", "p_variant_c"]], on="row_idx")
        assert len(m) == len(b) and (m.y_used == m.y_true).all()
        y = m.y_used.values; seen = m.is_seen.values == 1; pb, pv = m.p_b10.values, m.p_variant_c.values; bb, vv = pb >= .5, pv >= .5
        d = (bb == y).astype(int) - (vv == y).astype(int)
        res = {"n": len(m)}
        for seg, msk in [("all", np.ones(len(m), bool)), ("seen", seen), ("new", ~seen)]:
            v = pd.Series(d[msk]).groupby(m.seller_id.astype(str).values[msk]).sum().values; rng = np.random.default_rng(42)
            bs = np.array([v[rng.integers(0, len(v), len(v))].sum() for _ in range(2000)])
            dis = msk & (bb != vv)
            res[seg] = dict(acc_b10=round(float((bb == y)[msk].mean()), 5), acc_vc=round(float((vv == y)[msk].mean()), 5),
                            ll_b10=round(ll(y[msk], pb[msk]), 5), ll_vc=round(ll(y[msk], pv[msk]), 5),
                            net_b10_minus_vc=int(d[msk].sum()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                            n_disagree=int(dis.sum()), b10_right_when_disagree=round(float((bb == y)[dis].mean()), 3) if dis.any() else None)
        r[split] = res
    out[str(s)] = r
print(json.dumps(out, indent=1))
