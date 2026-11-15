"""Fixed H-AVG (0.5 logit... no: 0.5*p B10 + 0.5*p vc, as in the scored H-AVG) vs B10 and vs vc on a cut. usage: python havg_cut.py CUT_DIR VC_CSV"""
import sys, json, numpy as np, pandas as pd
cut, vcf = sys.argv[1:3]
y = (pd.read_csv(f'{cut}/holdout_labels.csv').condition == 'used').values; b10 = np.load(f'{cut}/b10/full/test_p.npy')
vc = pd.read_csv(vcf); assert (vc.y.values == y).all()
tr = pd.read_json(f'{cut}/train_data.jsonlines', lines=True); te = pd.read_json(f'{cut}/test_data.jsonlines', lines=True)
seen = te.seller_id.isin(set(tr.seller_id)).values; sel = te.seller_id.astype(str).values
def boot(d):
    v = pd.Series(d).groupby(sel).sum().values; r = np.random.default_rng(42); b = np.array([v[r.integers(0, len(v), len(v))].sum() for _ in range(2000)]); return [int(np.percentile(b, q)) for q in (5, 50, 95)]
out = {}
for s in ['p_vc_s2026', 'p_vc_s314159']:
    h = .5 * b10 + .5 * vc[s].values
    for nm, ref in [('vs_B10', b10), ('vs_vc', vc[s].values)]:
        a, c = (ref >= .5) == y, (h >= .5) == y; d = c.astype(int) - a.astype(int)
        out[f'{s}_{nm}'] = dict(acc_havg=round(float(c.mean()), 5), acc_ref=round(float(a.mean()), 5), fixes=int((c & ~a).sum()), breaks=int((~c & a).sum()), net=int(d.sum()), net_seen=int(d[seen].sum()), net_new=int(d[~seen].sum()), q_seller=boot(d))
print(json.dumps(out, indent=1))
