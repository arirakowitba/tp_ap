"""Evaluate text model T and blends with B10 on a cut. usage: python eval_inner.py CUT_DIR B10_P LABELS PRED_DIR PREFIX"""
import sys, glob, os, json, numpy as np, pandas as pd
cut, b10p, lab, pdir, pre = sys.argv[1:6]
y = (pd.read_csv(lab).condition == 'used').values; b10 = np.load(b10p)
tr = pd.read_json(f'{cut}/train_data.jsonlines', lines=True); te = pd.read_json(f'{cut}/test_data.jsonlines', lines=True)
seen = te.seller_id.isin(set(tr.seller_id)).values; sel = te.seller_id.astype(str).values; cat = te.category_id.astype(str).values
L = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
def boot(d, g):
    v = pd.Series(d).groupby(g).sum().values; r = np.random.default_rng(42); b = np.array([v[r.integers(0, len(v), len(v))].sum() for _ in range(2000)]); return [int(np.percentile(b, 5)), int(np.percentile(b, 50))]
def pair(p):
    a, c = (b10 >= .5) == y, (p >= .5) == y; d = c.astype(int) - a.astype(int)
    return dict(acc=round(float(c.mean()), 5), acc_seen=round(float(c[seen].mean()), 5), acc_new=round(float(c[~seen].mean()), 5), fixes=int((c & ~a).sum()), breaks=int((~c & a).sum()), net=int(d.sum()),
                net_seen=int(d[seen].sum()), net_new=int(d[~seen].sum()), q05_seller=boot(d, sel)[0], q05_cat=boot(d, cat)[0])
res = {'B10': dict(acc=round(float(((b10 >= .5) == y).mean()), 5))}
for f in sorted(glob.glob(f'{pdir}/{pre}*.npy')):
    k = os.path.basename(f)[len(pre):-4]; t = np.load(f); res[k] = dict(T_alone_acc=round(float(((t >= .5) == y).mean()), 5))
    for w in [.1, .2, .3, .4]:
        res[k][f'blend_w{w}'] = pair(1 / (1 + np.exp(-((1 - w) * L(b10) + w * L(t)))))
print(json.dumps(res, indent=1))
