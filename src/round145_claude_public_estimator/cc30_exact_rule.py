"""CC-0030: exact-rule holdout pairs + seller/category bootstrap + transfer simulation for U36, F86 and union.
Holdout analog (baseline B10, since outside span S4 == B10 == 10-10 for the strict test rows):
  U-rule: blend_fix != B10 (any margin)            <-> test U36
  F-rule: b_fix != B10 & |p_bfix-.5| >= .05 & blend_fix == B10   <-> test F86 (excludes U36 exactly)
Cells: |p-.5| band x regime; cell < 15 rows -> band-level h (both choices made AFTER seeing the holdout => DEVELOPMENT selection).
Transfer: cluster bootstrap by seller of the holdout cells -> per-flip p -> public subset ~ uniform without replacement (20998/30000)
-> Bernoulli outcomes -> Delta = sum_public (2c-1). Shift scenarios subtract 0.05/0.10 from every h.
usage: python cc30_exact_rule.py REPO HO HO_PRED DATA STATE_DIR [EVAL_DIR EVAL_B10 EVAL_BFIX EVAL_LABELS]   (optional: independent inner cut)"""
import sys, json, numpy as np, pandas as pd
repo, HO, HP, DATA, ST = sys.argv[1:6]; inner = sys.argv[6:10] if len(sys.argv) > 9 else None
def regimes(f, e):
    s = f.groupby('seller_id').date_created.agg(['min', 'max']); d = pd.to_datetime(e.date_created).dt.floor('D')
    mn = pd.to_datetime(e.seller_id.map(s['min'])).dt.floor('D'); mx = pd.to_datetime(e.seller_id.map(s['max'])).dt.floor('D')
    return np.where(mn.isna(), 'cold', np.where(d < mn, 'before', np.where(d > mx, 'after', np.where((d == mn) | (d == mx), 'ties', 'bracketed'))))
BANDS = [(0, .05), (.05, .1), (.1, .2), (.2, .51)]; REG = ['cold', 'before', 'bracketed', 'ties', 'after']
def rules(b10, bf):
    bl = .5 * b10 + .5 * bf; db = (bl >= .5) != (b10 >= .5)
    return {'U': (db, bl), 'F': ((bf >= .5) != (b10 >= .5) & (np.abs(bf - .5) >= .05) & ~db, bf)}
def load_eval(d, b10p, bfp, lab):
    tr = pd.read_json(f'{d}/train_data.jsonlines', lines=True); te = pd.read_json(f'{d}/test_data.jsonlines', lines=True)
    return dict(y=(pd.read_csv(lab).condition == 'used').values, reg=regimes(tr, te), sel=te.seller_id.astype(str).values, cat=te.category_id.astype(str).values,
                seen=te.seller_id.isin(set(tr.seller_id)).values, b10=np.load(b10p), bf=np.load(bfp)['te'])
def boot(d, g, n=2000):
    v = pd.Series(d).groupby(g).sum().values; r = np.random.default_rng(42); b = np.array([v[r.integers(0, len(v), len(v))].sum() for _ in range(n)])
    return [int(np.percentile(b, q)) for q in (2.5, 5, 50, 97.5)]
def cell_h(win, hc, hreg, mask, w=None):
    w = np.ones(len(win)) if w is None else w; H = {}
    for lo, hi in BANDS:
        mb = mask & (hc >= lo) & (hc < hi); hb = (win * w)[mb].sum() / max(w[mb].sum(), 1e-9)
        for r in REG:
            m = mb & (hreg == r); H[(lo, r)] = (win * w)[m].sum() / w[m].sum() if m.sum() >= 15 else hb
    return H
def analyse(E, name):
    out = {}
    R = rules(E['b10'], E['bf'])
    for k, (m, p) in R.items():
        win = (p >= .5) == E['y']; d = np.where(m, 2 * win - 1, 0)
        out[k] = dict(n=int(m.sum()), fixes=int((m & win).sum()), breaks=int((m & ~win).sum()), net=int(d.sum()), boot_seller=boot(d, E['sel']), boot_category=boot(d, E['cat']),
                      by_regime={r: [int((m & (E['reg'] == r)).sum()), int(d[E['reg'] == r].sum())] for r in REG},
                      seen_new=[int(d[E['seen']].sum()), int(d[~E['seen']].sum())])
    mu = R['U'][0] | R['F'][0]; du = np.where(R['U'][0], 2 * ((R['U'][1] >= .5) == E['y']) - 1, 0) + np.where(R['F'][0], 2 * ((R['F'][1] >= .5) == E['y']) - 1, 0)
    out['union'] = dict(n=int(mu.sum()), net=int(du.sum()), boot_seller=boot(du, E['sel']), boot_category=boot(du, E['cat']))
    full_bf = (E['bf'] >= .5) != (E['b10'] >= .5) & (np.abs(E['bf'] - .5) >= .05)
    out['F_overlap_omitted_by_exclusion'] = int((full_bf & R['U'][0]).sum())
    return out
H = load_eval(HO, f'{HO}/b10/full/test_p.npy', f'{HP}/pred_b_fix_noso_b.npz', f'{HO}/holdout_labels.csv')
res = {'holdout_14k': analyse(H, 'holdout')}
if inner: res['inner_cut_11k_independent'] = analyse(load_eval(inner[0], inner[1], inner[2], inner[3]), 'inner')
# ---- test blocks and transfer simulation
man = json.load(open(f'{repo}/submissions/claude_20261114/manifest.json'))['candidates']['claude_T30_mas_U36_mas_F86_estricto.csv']['blocks']
TR = pd.read_json(f'{DATA}/train_data.jsonlines', lines=True); TE = pd.read_json(f'{DATA}/test_data.jsonlines', lines=True); treg = regimes(TR, TE)
bl = np.load(f'{ST}/p_blend_fix.npy'); bf = np.load(f'{repo}/results/round146_claude_fixed_model/pred_b_fix_noso_b.npz')['te']
T = {'U': (np.array(man['U36']) - 1, bl), 'F': (np.array(man['F86']) - 1, bf)}
hR = rules(H['b10'], H['bf']); rng = np.random.default_rng(7); sellers = pd.Series(H['sel']); usel = sellers.unique()
rep = {}
for k in ['U', 'F']:
    idx, p = T[k]; tc = np.abs(p[idx] - .5); tr_ = treg[idx]
    cells = {}
    for i in range(len(idx)):
        lo = [b for b in BANDS if b[0] <= tc[i] < b[1]][0]; cells[(lo[0], tr_[i])] = cells.get((lo[0], tr_[i]), 0) + 1
    hm, hp_ = hR[k]; hwin = ((hp_ >= .5) == H['y']).astype(float); hc = np.abs(hp_ - .5)
    cnt = {f'{a}|{b}': [n, int((hm & (hc >= a) & (hc < [x[1] for x in BANDS if x[0] == a][0]) & (H['reg'] == b)).sum())] for (a, b), n in cells.items()}
    rep[k] = dict(test_cells_vs_holdout_rows=cnt, test_regimes={r: int((tr_ == r).sum()) for r in REG},
                  test_seen=int(TE.seller_id.iloc[idx].isin(set(TR.seller_id)).sum()),
                  test_categories_present_in_holdout_rule=float(np.isin(TE.category_id.iloc[idx].astype(str).values, H['cat'][hm]).mean()))
sims = {k: [] for k in ['U', 'F', 'union']}
seller_codes = pd.factorize(sellers)[0]
for b in range(1000):
    w = np.bincount(rng.integers(0, len(usel), len(usel)), minlength=len(usel))[seller_codes].astype(float)
    tot = 0; dd = {}
    for k in ['U', 'F']:
        idx, p = T[k]; hm, hp_ = hR[k]; win = ((hp_ >= .5) == H['y']).astype(float); Hc = cell_h(win, np.abs(hp_ - .5), H['reg'], hm, w)
        tc = np.abs(p[idx] - .5); ph = np.array([Hc[([bb for bb in BANDS if bb[0] <= tc[i] < bb[1]][0][0], treg[idx][i])] for i in range(len(idx))])
        pub = rng.random(len(idx)) < 20998 / 30000; c = rng.random(len(idx)) < ph; dd[k] = (ph, pub, c); sims[k].append(int(((2 * c - 1) * pub).sum()))
    sims['union'].append(sims['U'][-1] + sims['F'][-1])
    for s in (.05, .10):
        for k in ['U', 'F']:
            ph, pub, _ = dd[k]; c = rng.random(len(ph)) < np.clip(ph - s, 0, 1); sims.setdefault(f'{k}_shift{s}', []).append(int(((2 * c - 1) * pub).sum()))
for s in (.05, .10): sims[f'union_shift{s}'] = list(np.array(sims[f'U_shift{s}']) + np.array(sims[f'F_shift{s}']))
res['transfer_simulation'] = {k: dict(q05=float(np.percentile(v, 5)), q50=float(np.percentile(v, 50)), q95=float(np.percentile(v, 95)), mean=round(float(np.mean(v)), 1), P_pos=round(float(np.mean(np.array(v) > 0)), 3)) for k, v in sims.items()}
res['test_representation'] = rep
res['selection_note'] = 'threshold |p-.5|>=.05 for F and fallback<15 were chosen after viewing the 14k holdout band table: DEVELOPMENT selection; inner_cut_11k (seed 145, disjoint evaluation rows) is the replication'
print(json.dumps(res, indent=1, default=str))
