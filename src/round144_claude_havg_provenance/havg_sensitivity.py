"""CC-0006: H-AVG (fixed 0.5*B10 + 0.5*vc, th 0.5) under three definitions of variant_c confirm p:
 export_cb = Antigravity CB-only export (AG-0015, used in CL-0014); replica_cb / replica_blend = Claude's Linux replica of AG-0011
 (CB-only and 0.75CB+0.25LGB). Paired vs B10 and vs the same vc definition; seen/new; seller bootstrap 2000.
usage: python havg_sensitivity.py B10_FIT_DIR REPLICA_DIR EXPORT_DIR(vc{seed}.csv.gz)"""
import sys, json, numpy as np, pandas as pd
fitd, repd, expd = sys.argv[1:4]
def ll(y, p): p = np.clip(p, 1e-6, 1 - 1e-6); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
def pair(y, a, b, sel, msk):
    d = ((b == y).astype(int) - (a == y).astype(int))[msk]; v = pd.Series(d).groupby(sel[msk]).sum().values; rng = np.random.default_rng(42)
    bs = np.array([v[rng.integers(0, len(v), len(v))].sum() for _ in range(2000)]); return [int(d.sum()), round(float(np.percentile(bs, 2.5))), round(float(np.percentile(bs, 97.5)))]
out = {}
for s in [2026, 314159, 20261030]:
    b = pd.read_csv(f"{fitd}/seed{s}_fit/paired_b10_vs_A_hist_seed{s}.csv.gz")[["row_idx", "p_b10", "y_used", "is_seen", "seller_id"]]
    e = pd.read_csv(f"{expd}/vc{s}.csv.gz"); e = e[e.split == "confirm"][["row_idx", "p_variant_c"]]
    r = pd.read_csv(f"{repd}/vc_blend_replica_seed{s}.csv.gz"); r = r[r.split == "confirm"][["row_idx", "p_cb", "p_blend"]]
    m = b.merge(e, on="row_idx").merge(r, on="row_idx"); assert len(m) == len(b) == len(e) == len(r)
    y = m.y_used.values; seen = m.is_seen.values == 1; sel = m.seller_id.astype(str).values; segs = [("all", np.ones(len(y), bool)), ("seen", seen), ("new", ~seen)]
    res = {"export_cb_vs_replica_cb_max_abs_dp": round(float(np.abs(m.p_variant_c - m.p_cb).max()), 4),
           "export_cb_vs_replica_cb_decision_diffs": int(((m.p_variant_c >= .5) != (m.p_cb >= .5)).sum())}
    for name, vc in [("export_cb", m.p_variant_c.values), ("replica_cb", m.p_cb.values), ("replica_blend", m.p_blend.values)]:
        h = .5 * m.p_b10.values + .5 * vc
        res[name] = dict(vc_acc=round(float(((vc >= .5) == y).mean()), 5), vc_ll=round(ll(y, vc), 5),
                         havg_acc=round(float(((h >= .5) == y).mean()), 5), havg_ll=round(ll(y, h), 5),
                         havg_vs_b10={k: pair(y, m.p_b10.values >= .5, h >= .5, sel, q) for k, q in segs},
                         havg_vs_vc={k: pair(y, vc >= .5, h >= .5, sel, q) for k, q in segs})
    res["b10_acc"] = round(float(((m.p_b10.values >= .5) == y).mean()), 5)
    out[str(s)] = res
print(json.dumps(out, indent=1))
