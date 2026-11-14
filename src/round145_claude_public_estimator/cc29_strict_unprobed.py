"""CC-0029: strict 'unprobed' rows and holdout support by |p-.5| band x five regimes (cell < 15 rows -> band-level h).
strict = outside scored-vector span AND S4 == original 10-10 ensemble AND S4 == B10 full70k.
usage (from a dir with V3.npy, V3_meta.json, p_b10.npy, p_blend_fix.npy):
  python cc29_strict_unprobed.py REPO HOLDOUT_DIR HOLDOUT_PRED_DIR WT_MAIN DATA_DIR"""
import sys, json, numpy as np, pandas as pd
repo, HO, HP, WT, DATA = sys.argv[1:6]
V = np.load('V3.npy'); meta = json.load(open('V3_meta.json')); n = [m[0] for m in meta]
S4 = V[n.index('TODAY_S4')]; span = ((np.delete(V, [n.index('TODAY_TOP174'), n.index('TODAY_TOP30')], 0) - S4) != 0).any(0)
b10 = (np.load('p_b10.npy') >= .5).astype(V.dtype); bl = np.load('p_blend_fix.npy'); bf = np.load(f'{repo}/results/round146_claude_fixed_model/pred_b_fix_noso_b.npz')['te']
e1010 = (pd.read_csv(f'{WT}/submissions/submission_optimized_ensemble.csv').condition.values == 'used').astype(V.dtype)
strict = ~span & (S4 == e1010) & (S4 == b10); U36 = strict & ((bl >= .5) != S4); F86 = strict & ((bf >= .5) != S4) & (np.abs(bf - .5) >= .05) & ~U36
def regimes(f, e):
    s = f.groupby('seller_id').date_created.agg(['min', 'max']); d = pd.to_datetime(e.date_created).dt.floor('D')
    mn = pd.to_datetime(e.seller_id.map(s['min'])).dt.floor('D'); mx = pd.to_datetime(e.seller_id.map(s['max'])).dt.floor('D')
    return np.where(mn.isna(), 'cold', np.where(d < mn, 'before', np.where(d > mx, 'after', np.where((d == mn) | (d == mx), 'ties', 'bracketed'))))
treg = regimes(pd.read_json(f'{DATA}/train_data.jsonlines', lines=True), pd.read_json(f'{DATA}/test_data.jsonlines', lines=True))
y = (pd.read_csv(f'{HO}/holdout_labels.csv').condition == 'used').values
hreg = regimes(pd.read_json(f'{HO}/train_data.jsonlines', lines=True), pd.read_json(f'{HO}/test_data.jsonlines', lines=True))
hb10 = np.load(f'{HO}/b10/full/test_p.npy'); hbf = np.load(f'{HP}/pred_b_fix_noso_b.npz')['te']; hbl = .5 * hb10 + .5 * hbf
out = {'strict_rows': int(strict.sum()), 'U36': int(U36.sum()), 'F86': int(F86.sum())}
for nm, tp, hp, mask in [('U36_blend_fix', bl, hbl, U36), ('F86_b_fix', bf, hbf, F86)]:
    hd = (hp >= .5) != (hb10 >= .5); win = (hp >= .5) == y; tc, hc = np.abs(tp - .5), np.abs(hp - .5); E = 0; cells = []
    for lo, hi in [(0, .05), (.05, .1), (.1, .2), (.2, .51)]:
        for r in ['cold', 'before', 'bracketed', 'ties', 'after']:
            mh = hd & (hc >= lo) & (hc < hi) & (hreg == r); mt = mask & (tc >= lo) & (tc < hi) & (treg == r)
            if not mt.sum(): continue
            hcell = win[mh].mean() if mh.sum() >= 15 else win[hd & (hc >= lo) & (hc < hi)].mean()
            E += 0.7 * mt.sum() * (2 * hcell - 1); cells.append([lo, r, int(mt.sum()), int(mh.sum()), round(float(hcell), 3)])
    out[nm] = dict(E=round(float(E), 1), cells=cells)
print(json.dumps(out, indent=1))
