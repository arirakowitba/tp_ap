"""H-AVG (predeclared in CL-0013 §3): fixed 0.5*B10 + 0.5*variant_c, threshold 0.5, no weights chosen on any split.
Paired on reused confirm (development) vs each component and vs Blend075_Bag3. usage: python h_avg.py B10_FIT_DIR AG_DIR"""
import sys, json, numpy as np, pandas as pd
fitd, ag = sys.argv[1:3]
def ll(y, p): p = np.clip(p, 1e-6, 1 - 1e-6); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
def pair(y, a, b, sel, msk):
    d = ((b == y).astype(int) - (a == y).astype(int))[msk]; v = pd.Series(d).groupby(sel[msk]).sum().values; rng = np.random.default_rng(42)
    bs = np.array([v[rng.integers(0, len(v), len(v))].sum() for _ in range(2000)]); return int(d.sum()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
out = {}
for s in [2026, 314159, 20261030]:
    b = pd.read_csv(f"{fitd}/seed{s}_fit/paired_b10_vs_A_hist_seed{s}.csv.gz")
    vc = pd.read_csv(f"{ag}/vc_tc_{s}.csv.gz"); vc = vc[vc.split == "confirm"][["row_idx", "p_variant_c"]]
    bl = pd.read_csv(f"{ag}/paired_all_{s}.csv.gz")[["row_idx", "p_Blend075_Bag3_hist"]]
    m = b.merge(vc, on="row_idx").merge(bl, on="row_idx"); assert len(m) == len(b)
    y = m.y_used.values; seen = m.is_seen.values == 1; sel = m.seller_id.astype(str).values
    pavg = .5 * m.p_b10.values + .5 * m.p_variant_c.values
    P = {"avg": pavg, "b10": m.p_b10.values, "vc": m.p_variant_c.values, "blend": m.p_Blend075_Bag3_hist.values}
    r = {k: dict(acc=round(float(((p >= .5) == y).mean()), 5), acc_seen=round(float(((p >= .5) == y)[seen].mean()), 5), acc_new=round(float(((p >= .5) == y)[~seen].mean()), 5), ll=round(ll(y, p), 5)) for k, p in P.items()}
    for k in ["b10", "vc", "blend"]:
        r[f"avg_vs_{k}"] = {seg: pair(y, P[k] >= .5, pavg >= .5, sel, msk) for seg, msk in [("all", np.ones(len(y), bool)), ("seen", seen), ("new", ~seen)]}
    out[str(s)] = r
print(json.dumps(out, indent=1))
