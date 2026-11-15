"""CC-0038 frozen ensemble grid. Select on inner r145 cut, confirm ONCE on faithful holdout.
Components (fit-only on each cut): B10, b_fix, vc (AG variant_c 600 it, seed 2026; s314159 for sensitivity), fincat (b_cat_noso_b
with sparse_all = fin_cat5 proxy), T (text, title+warranty+lexicon). Logit-simplex weights step 0.1 over the 5 components.
usage: python grid.py OUT_JSON"""
import sys, json, itertools, numpy as np, pandas as pd
W = '/home/claude/work/'
L = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sig = lambda z: 1 / (1 + np.exp(-z))
def load(cut):
    if cut == 'inner':
        lab = pd.read_csv(W + 'r145_inner/inner_labels.csv'); y = (lab.condition == 'used').values; d = W + 'r145_inner/'
        P = dict(B10=np.load(d + 'b10/full/test_p.npy'), b_fix=np.load(d + 'pred_c_in_base.npz')['te'], fincat=np.load(W + 'r149/inner_fc/pred_b_cat_noso_b.npz')['te'],
                 T=np.load(W + 'r148/inner_T_abl_title_warranty_lex.npy'))
        vc = pd.read_csv(W + 'r149/vc/vc_inner.csv.gz'); tr = pd.read_json(d + 'train_data.jsonlines', lines=True); te = pd.read_json(d + 'test_data.jsonlines', lines=True)
    else:
        lab = pd.read_csv(W + 'holdout/holdout_labels.csv'); y = (lab.condition == 'used').values; d = W + 'holdout/'
        P = dict(B10=np.load(d + 'b10/full/test_p.npy'), b_fix=np.load(W + 'r145_ho/pred_b_fix_noso_b.npz')['te'], fincat=np.load(W + 'r145_ho/pred_b_cat_noso_b.npz')['te'],
                 T=np.load(W + 'r148/ho_T_title_warranty_lex.npy'))
        vc = pd.read_csv(W + 'r149/vc/vc_holdout.csv.gz'); tr = pd.read_json(d + 'train_data.jsonlines', lines=True); te = pd.read_json(d + 'test_data.jsonlines', lines=True)
    assert (vc.y.values == y).all(), 'vc label alignment failed'
    P['vc'] = vc.p_vc_s2026.values; P['vc_s2'] = vc.p_vc_s314159.values
    for k, p in P.items(): assert len(p) == len(y), k
    seen = te.seller_id.isin(set(tr.seller_id)).values
    return y, P, seen, te.seller_id.astype(str).values, te.category_id.astype(str).values
def ll(y, p): p = np.clip(p, 1e-6, 1 - 1e-6); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
def boot(d, g):
    v = pd.Series(d).groupby(g).sum().values; r = np.random.default_rng(42); b = np.array([v[r.integers(0, len(v), len(v))].sum() for _ in range(2000)])
    return [int(np.percentile(b, 5)), int(np.percentile(b, 50)), int(np.percentile(b, 95))]
def pair(y, a, b, sel, cat, seen):
    ca, cb = (a >= .5) == y, (b >= .5) == y; d = cb.astype(int) - ca.astype(int)
    return dict(fixes=int((cb & ~ca).sum()), breaks=int((~cb & ca).sum()), net=int(d.sum()), net_seen=int(d[seen].sum()), net_new=int(d[~seen].sum()),
                q_seller=boot(d, sel), q_cat=boot(d, cat), disagree=int(((a >= .5) != (b >= .5)).sum()))
COMP = ['B10', 'b_fix', 'vc', 'fincat', 'T']
grid = [w for w in itertools.product(range(11), repeat=5) if sum(w) == 10]
def mix(P, w, vck='vc'): return sig(sum((wi / 10) * L(P[vck if k == 'vc' else k]) for wi, k in zip(w, COMP)))
out = {}
for cut in ['inner']:
    y, P, seen, sel, cat = load(cut)
    out['inner_components'] = {k: dict(acc=round(float(((p >= .5) == y).mean()), 5), ll=round(ll(y, p), 5)) for k, p in P.items()}
    scores = []
    for w in grid:
        p = mix(P, w); scores.append((float(((p >= .5) == y).mean()), -ll(y, p), w))
    scores.sort(reverse=True); best = scores[0][2]
    out['inner_top10'] = [dict(w=dict(zip(COMP, [x / 10 for x in s[2]])), acc=round(s[0], 5), ll=round(-s[1], 5)) for s in scores[:10]]
    refs = {'HAVG(0.5B10+0.5vc)': (5, 0, 5, 0, 0), 'blend_fix(0.5B10+0.5bfix)': (5, 5, 0, 0, 0), 'avg_all5': (2, 2, 2, 2, 2), 'SELECTED': best}
    out['inner_pairs_vs_B10'] = {k: dict(w=dict(zip(COMP, [x / 10 for x in w])), acc=round(float(((mix(P, w) >= .5) == y).mean()), 5), ll=round(ll(y, mix(P, w)), 5),
                                         **pair(y, P['B10'], mix(P, w), sel, cat, seen)) for k, w in refs.items()}
    out['inner_selected_vs_HAVG'] = pair(y, mix(P, refs['HAVG(0.5B10+0.5vc)']), mix(P, best), sel, cat, seen)
    out['inner_seed_sensitivity_selected'] = dict(acc_vc_s2026=round(float(((mix(P, best) >= .5) == y).mean()), 5), acc_vc_s314159=round(float(((mix(P, best, 'vc_s2') >= .5) == y).mean()), 5))
    out['selected_weights'] = dict(zip(COMP, [x / 10 for x in best]))
# promotion rule (frozen in CL-0038)
r = out['inner_pairs_vs_B10']['SELECTED']; promote = r['net'] > 0 and r['q_seller'][0] >= 0; out['promote_after_inner'] = promote
y, P, seen, sel, cat = load('holdout')
out['holdout_components'] = {k: dict(acc=round(float(((p >= .5) == y).mean()), 5), ll=round(ll(y, p), 5)) for k, p in P.items()}
out['holdout_confirm_ONCE'] = {k: dict(acc=round(float(((mix(P, w) >= .5) == y).mean()), 5), ll=round(ll(y, mix(P, w)), 5), **pair(y, P['B10'], mix(P, w), sel, cat, seen))
                               for k, w in [('SELECTED', best), ('HAVG(0.5B10+0.5vc)', (5, 0, 5, 0, 0))]}
out['holdout_selected_vs_HAVG'] = pair(y, mix(P, (5, 0, 5, 0, 0)), mix(P, best), sel, cat, seen)
out['holdout_seed_sensitivity_selected'] = dict(acc_vc_s2026=out['holdout_confirm_ONCE']['SELECTED']['acc'], acc_vc_s314159=round(float(((mix(P, best, 'vc_s2') >= .5) == y).mean()), 5))
json.dump(out, open(sys.argv[1], 'w'), indent=1); print(json.dumps(out, indent=1))
