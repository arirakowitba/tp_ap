"""Evaluate H-AVG-Plus recipe per cut on tune (clean: fixed iterations) and reused confirm (development):
compares H-AVG-Plus (.5 B10 + .5 vc_blend) vs B10, vs vc_blend, vs H-AVG-single (.5 B10 + .5 CB seed=cut, 600 it), seen/new, logloss,
seller bootstrap 2000. usage: python havg_plus_eval.py RECIPE_DIR B10_TC_DIR AG_BENCH_DIR(preds_A_hist_seed*_{tune,confirm}.csv.gz)"""
import sys, json, numpy as np, pandas as pd
rd, bd, ag = sys.argv[1:4]
def ll(y, p): p = np.clip(p, 1e-6, 1 - 1e-6); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
def pair(y, a, b, sel, msk):
    d = ((b == y).astype(int) - (a == y).astype(int))[msk]; v = pd.Series(d).groupby(sel[msk]).sum().values; rng = np.random.default_rng(42)
    bs = np.array([v[rng.integers(0, len(v), len(v))].sum() for _ in range(2000)]); return [int(d.sum()), round(float(np.percentile(bs, 2.5))), round(float(np.percentile(bs, 97.5)))]
out = {}
for s in [2026, 314159, 20261030]:
    r = pd.read_csv(f"{rd}/havg_plus_recipe_cut{s}.csv.gz"); b = pd.read_csv(f"{bd}/b10_tune_confirm_seed{s}.csv.gz")
    meta = pd.concat([pd.read_csv(f"{ag}/preds_A_hist_seed{s}_{sp}.csv.gz")[["row_idx", "seller_id", "is_seen", "y_used"]] for sp in ("tune", "confirm")])
    m = r.merge(b[["row_idx", "p_b10"]], on="row_idx").merge(meta, on="row_idx"); assert len(m) == len(r) and (m.y_true == m.y_used).all()
    res = {}
    for sp in ("tune", "confirm"):
        q = m[m.split == sp]; y = q.y_true.values; seen = q.is_seen.values == 1; sel = q.seller_id.astype(str).values
        segs = [("all", np.ones(len(y), bool)), ("seen", seen), ("new", ~seen)]
        P = {"b10": q.p_b10.values, "vc_blend": q.p_vc_blend.values, "cb_bag": q.p_cb_bag.values,
             "havg_single": .5 * q.p_b10.values + .5 * q[f"p_cb_s{s}"].values, "havg_plus": .5 * q.p_b10.values + .5 * q.p_vc_blend.values}
        e = {k: dict(acc=round(float(((p >= .5) == y).mean()), 5), acc_seen=round(float(((p >= .5) == y)[seen].mean()), 5),
                     acc_new=round(float(((p >= .5) == y)[~seen].mean()), 5), ll=round(ll(y, p), 5)) for k, p in P.items()}
        for k in ("b10", "vc_blend", "havg_single"): e[f"plus_vs_{k}"] = {g: pair(y, P[k] >= .5, P["havg_plus"] >= .5, sel, w) for g, w in segs}
        res[sp] = e
    out[str(s)] = res
print(json.dumps(out, indent=1))
