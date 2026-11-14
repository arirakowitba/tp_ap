"""CC-0020 audit: hashes, holdout pairs (seller/category bootstrap, five regimes), per-ID comparison vs S4, historical support.
usage: python audit.py OUT_JSON"""
import sys, json, hashlib, numpy as np, pandas as pd
R = '/home/claude/tp_ap/'; HO = '/home/claude/work/holdout/'; out = {}
def regimes(fit_df, ev_df):
    s = fit_df.groupby('seller_id').date_created.agg(['min', 'max']); d = pd.to_datetime(ev_df.date_created).dt.floor('D')
    mn = pd.to_datetime(ev_df.seller_id.map(s['min'])).dt.floor('D'); mx = pd.to_datetime(ev_df.seller_id.map(s['max'])).dt.floor('D')
    r = np.where(mn.isna(), 'cold', np.where(d < mn, 'before', np.where(d > mx, 'after', np.where((d == mn) | (d == mx), 'ties', 'bracketed'))))
    return r
def boot(d, g, n=2000):
    v = pd.Series(d).groupby(g).sum().values; rng = np.random.default_rng(42); b = np.array([v[rng.integers(0, len(v), len(v))].sum() for _ in range(n)])
    return [int(np.percentile(b, 2.5)), int(np.percentile(b, 5)), int(np.percentile(b, 97.5))]
def pair(y, a, b, sel, cat, reg):
    ca, cb = (a >= .5) == y, (b >= .5) == y; d = cb.astype(int) - ca.astype(int)
    res = dict(fixes=int((cb & ~ca).sum()), breaks=int((~cb & ca).sum()), net=int(d.sum()), ci95_q05_seller=boot(d, sel), ci95_q05_category=boot(d, cat))
    res['by_regime'] = {k: dict(n=int((reg == k).sum()), net=int(d[reg == k].sum()), acc_a=round(float(ca[reg == k].mean()), 5), acc_b=round(float(cb[reg == k].mean()), 5)) for k in ['cold', 'before', 'bracketed', 'ties', 'after']}
    return res
# ---- holdout
tr = pd.read_json(HO + 'train_data.jsonlines', lines=True); te = pd.read_json(HO + 'test_data.jsonlines', lines=True)
y = (pd.read_csv(HO + 'holdout_labels.csv').condition == 'used').values; reg = regimes(tr, te); sel = te.seller_id.astype(str).values; cat = te.category_id.astype(str).values
P = {'b10': np.load(HO + 'b10/full/test_p.npy'), 'b_cat_noso_b_sparse_all': np.load('/home/claude/work/r145_ho/pred_b_cat_noso_b.npz')['te'],
     'b_fix_noso_b': np.load('/home/claude/work/r145_ho/pred_b_fix_noso_b.npz')['te']}
P['blend_fix'] = .5 * P['b10'] + .5 * P['b_fix_noso_b']
out['holdout_regime_counts'] = {k: int((reg == k).sum()) for k in ['cold', 'before', 'bracketed', 'ties', 'after']}
out['holdout_pairs'] = {'sparse_all->sparse_fix (b_cat_noso_b -> b_fix_noso_b)': pair(y, P['b_cat_noso_b_sparse_all'], P['b_fix_noso_b'], sel, cat, reg),
                        'b10 -> b_fix_noso_b': pair(y, P['b10'], P['b_fix_noso_b'], sel, cat, reg), 'b10 -> blend_fix': pair(y, P['b10'], P['blend_fix'], sel, cat, reg)}
out['holdout_calibration'] = {k: {r: dict(mean_p=round(float(p[reg == r].mean()), 4), rate_used=round(float(y[reg == r].mean()), 4)) for r in ['cold', 'before', 'bracketed', 'ties', 'after']} for k, p in P.items()}
# ---- test, per-ID vs S4
TR = pd.read_json('/home/claude/data/train_data.jsonlines', lines=True); TE = pd.read_json('/home/claude/data/test_data.jsonlines', lines=True); treg = regimes(TR, TE)
out['test_regime_counts'] = {k: int((treg == k).sum()) for k in ['cold', 'before', 'bracketed', 'ties', 'after']}
V = np.load('/home/claude/work/r145/V.npy'); meta = json.load(open('/home/claude/work/r145/V_meta.json')); s4 = V[[m[0] for m in meta].index('TODAY_S4')]
span = ((V - s4) != 0).any(0)
def canon(v): return hashlib.sha256(''.join(f'{i+1}:{"used" if v[i] else "new"}\n' for i in range(30000)).encode()).hexdigest()
def lf(v): return hashlib.sha256(('ID,condition\n' + ''.join(f'{i+1},{"used" if v[i] else "new"}\n' for i in range(30000))).encode()).hexdigest()
b10 = np.load('/home/claude/work/r145/p_b10.npy'); bf = np.load(R + 'results/round146_claude_fixed_model/pred_b_fix_noso_b.npz')['te']
C = {'S4': s4, 'b10_pure': (b10 >= .5).astype(int), 'b_fix_pure': (bf >= .5).astype(int), 'blend_fix_pure': ((.5 * b10 + .5 * bf) >= .5).astype(int)}
sub = R + 'submissions/claude_20261114/'
for f in ['claude_blendfix_posterior_top30_sobre_S4.csv', 'claude_blendfix_posterior_top174_sobre_S4.csv', 'claude_posterior_opt_top30_sobre_S4.csv', 'claude_posterior_opt_top131_sobre_S4.csv', 'claude_blendfix_pure_model.csv']:
    d = pd.read_csv(sub + f); assert d.ID.tolist() == list(range(1, 30001)); C[f] = (d.condition.values == 'used').astype(int)
seen = TE.seller_id.isin(set(TR.seller_id)).values
out['test_vs_S4'] = {}
for k, v in C.items():
    dif = v != s4
    out['test_vs_S4'][k] = dict(lf_sha256=lf(v), canonical_id_condition_sha256=canon(v), n_diff=int(dif.sum()), used_to_new=int((dif & (s4 == 1)).sum()), new_to_used=int((dif & (s4 == 0)).sum()),
        seen=int((dif & seen).sum()), new_seller=int((dif & ~seen).sum()), in_historical_contrast_span=int((dif & span).sum()), outside_span=int((dif & ~span).sum()),
        by_regime={r: int((dif & (treg == r)).sum()) for r in ['cold', 'before', 'bracketed', 'ties', 'after']})
out['historical_contrast_span_rows'] = int(span.sum())
json.dump(out, open(sys.argv[1], 'w'), indent=1); print(json.dumps({k: v for k, v in out.items() if k != 'holdout_calibration'}, indent=1)[:6000])
