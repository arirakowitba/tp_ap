"""CC-0041: B10 / H-AVG / corrected causal / corrected base on IDENTICAL rows of AG's 3-way split (tune+confirm), all fit-only on AG's fit.
Mixture weights (logit, w in {0.1..0.5}) for B10+causal and H-AVG+causal chosen on TUNE only, confirmed ONCE on CONFIRM.
usage: python compare_on_ag_split.py AG3W_DIR CAUSAL_CSV OUT_JSON"""
import sys, json, numpy as np, pandas as pd
D, cf, outp = sys.argv[1:4]
lab = pd.read_csv(f'{D}/holdout_labels.csv'); y = (lab.condition == 'used').values; role = lab.role.values; reg = lab.regime.values
b10 = np.load(f'{D}/b10/full/test_p.npy'); vc = pd.read_csv(f'{D}/vc/vc_holdout.csv.gz'); assert (vc.y.values == y).all() and (vc.orig_row.values == lab.train_row.values).all()
c = pd.read_csv(cf); assert (c.row.values == lab.train_row.values).all() and (c.y.values == y).all()
te = pd.read_json(f'{D}/test_data.jsonlines', lines=True); sel = te.seller_id.astype(str).values; cat = te.category_id.astype(str).values
L = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); sig = lambda z: 1 / (1 + np.exp(-z))
P = dict(B10=b10, HAVG=.5 * b10 + .5 * vc.p_vc_s2026.values, vc=vc.p_vc_s2026.values, causal=c.p_causal.values, base=c.p_base.values)
def ll(yy, p): p = np.clip(p, 1e-6, 1 - 1e-6); return float(-np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)))
def boot(d, g):
    v = pd.Series(d).groupby(g).sum().values; r = np.random.default_rng(42); b = np.array([v[r.integers(0, len(v), len(v))].sum() for _ in range(2000)]); return [int(np.percentile(b, q)) for q in (5, 50, 95)]
def pair(m, a, b):
    ca, cb = (a[m] >= .5) == y[m], (b[m] >= .5) == y[m]; d = cb.astype(int) - ca.astype(int)
    return dict(fixes=int((cb & ~ca).sum()), breaks=int((~cb & ca).sum()), net=int(d.sum()), q_seller=boot(d, sel[m]), q_cat=boot(d, cat[m]),
                by_regime={r: int(d[reg[m] == r].sum()) for r in sorted(set(reg[m]))})
out = {}
for r in ['tune', 'confirm']:
    m = role == r; out[f'{r}_components'] = {k: dict(acc=round(float(((p[m] >= .5) == y[m]).mean()), 5), ll=round(ll(y[m], p[m]), 5)) for k, p in P.items()}
mt, mc = role == 'tune', role == 'confirm'
for ref in ['B10', 'HAVG']:
    best = max([.1, .2, .3, .4, .5], key=lambda w: (((sig((1 - w) * L(P[ref][mt]) + w * L(P['causal'][mt])) >= .5) == y[mt]).mean(), -w))
    mix = sig((1 - best) * L(P[ref]) + best * L(P['causal']))
    out[f'{ref}+causal'] = dict(w_selected_on_tune=best, tune_vs_ref=pair(mt, P[ref], mix), CONFIRM_vs_ref=pair(mc, P[ref], mix),
                                confirm_acc=round(float(((mix[mc] >= .5) == y[mc]).mean()), 5))
out['confirm_causal_vs_B10'] = pair(mc, P['B10'], P['causal']); out['confirm_causal_vs_HAVG'] = pair(mc, P['HAVG'], P['causal'])
out['confirm_HAVG_vs_B10'] = pair(mc, P['B10'], P['HAVG']); out['confirm_causal_vs_base'] = pair(mc, P['base'], P['causal'])
json.dump(out, open(outp, 'w'), indent=1); print(json.dumps(out, indent=1))
