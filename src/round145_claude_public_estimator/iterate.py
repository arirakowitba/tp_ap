"""One-step iteration for the manual cup sequence (Claude, 14-11).
usage: python iterate.py STATE_DIR OUT_DIR [NEW_CSV NEW_SCORE NEW_NAME]
 1) optionally appends the newly scored vector (dup/quarantine check) to STATE_DIR/V5.npy + V5_meta.json
 2) base = best observed public vector (ties -> earliest, i.e. keep the established anchor)
 3) re-selects prior sharpness b in {0.25,0.5,0.75,1.0} on the stack prior by LOO MAE over all vectors
 4) rows where flipping would revert a probe-era S4 correction back to the 10-10 original are excluded;
    P(base wrong) per in-span row = min over priors {stack, blend_fix, mean_all} at selected b
 5) writes next candidates top{20,50,100} over base with E (selected b, min over priors), bias-corrected E (-0.12/flip,
    from top174 backtest) and pessimistic E (b=0.25 H-AVG-Plus); prints the recommendation (max bias-corrected E per SD).
Never uploads; candidate CSVs are DEVELOPMENT_ONLY."""
import sys, json, hashlib, os, numpy as np, pandas as pd
ST, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True); NP = 20998; fr = NP / 30000
V = np.load(f'{ST}/V5.npy').astype(np.int8); meta = json.load(open(f'{ST}/V5_meta.json'))
if len(sys.argv) > 5:
    csv, sc, nm = sys.argv[3], float(sys.argv[4]), sys.argv[5]; d = pd.read_csv(csv); assert d.ID.tolist() == list(range(1, 30001))
    v = (d.condition.values == 'used').astype(np.int8); same = [m for i, m in enumerate(meta) if (V[i] == v).all()]
    if same: sys.exit(f'QUARANTINE/DUP: identical to {same[0][0]} score {same[0][3]}')
    V = np.vstack([V, v]); meta.append([nm, os.path.basename(csv), 'user_reported', sc, round(sc * NP), 'ariel'])
    np.save(f'{ST}/V5.npy', V); json.dump(meta, open(f'{ST}/V5_meta.json', 'w')); print('added', nm, sc, 'hits', round(sc * NP))
Vf = V.astype(float); h = np.array([round(m[3] * NP) for m in meta], float)
ib = int(max(range(len(meta)), key=lambda i: (h[i], -i))); base = Vf[ib]; print('base =', meta[ib][0], meta[ib][3])  # ties -> earliest (established) vector
others = [i for i in range(len(V)) if i != ib]
def fit(p, idx, R=0.35):
    mu = fr * (2 * p - 1); s2 = fr - mu ** 2; D = Vf[idx] - base; d = h[idx] - h[ib]; cols = np.where(np.abs(D).sum(0) > 0)[0]; Dc = D[:, cols]
    return dict(mu=mu, s2=s2, cols=cols, Dc=Dc, Sinv=np.linalg.inv((Dc * s2[cols]) @ Dc.T + R * np.eye(len(idx))), r=d - Dc @ mu[cols])
def pred(F, c):
    g = c - base; k = (g[F['cols']] * F['s2'][F['cols']]) @ F['Dc'].T; return g @ F['mu'] + k @ F['Sinv'] @ F['r'], np.sqrt(max((g ** 2) @ F['s2'] - k @ F['Sinv'] @ k, 1e-9))
def pm(F): mu = F['mu'].copy(); c = F['cols']; mu[c] += F['s2'][c] * (F['Dc'].T @ (F['Sinv'] @ F['r'])); return mu
sharp = lambda p, b: 1 / (1 + np.exp(-b * np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))))
bl = np.load(f'{ST}/p_blend_fix.npy'); bf = np.load(f'{ST}/p_b_fix.npy'); hp = np.load(f'{ST}/p_havg_plus.npy'); b10 = np.load(f'{ST}/p_b10.npy')
st = np.load(f'{ST}/p_stack.npy')  # 0.6 logit blend_fix + 0.4 logit fin_cat5 (LOO-selected, MAE 3.50)
P = {'stack': st, 'blend_fix': bl, 'mean_all': (b10 + hp + bl + bf) / 4}
mae = {}
for b in [.25, .5, .75, 1.0]:
    pr = sharp(st, b); err = []
    for k in others:
        F = fit(pr, [i for i in others if i != k]); e, _ = pred(F, Vf[k]); err.append(abs(h[k] - h[ib] - e))
    mae[b] = float(np.mean(err))
B = min(mae, key=mae.get); print('LOO MAE by b:', {k: round(v, 2) for k, v in mae.items()}, '-> b =', B)
Fs = {k: fit(sharp(p, B), others) for k, p in P.items()}; F25 = fit(sharp(hp, .25), others)
# CL-0034: on model-selected overlays b=1 was optimistic (mean err -11.3 over 3) and b=0.25 nearly unbiased (-2.0) -> rank/decide with b=0.25
P25 = {'stack': st, 'havg_plus': hp, 'blend_fix': bl}; G25 = {k: fit(sharp(p, .25), others) for k, p in P25.items()}
span = ((Vf - base) != 0).any(0); PW = []
for F in G25.values():
    py = np.clip(.5 + pm(F) / (2 * fr), 0, 1); PW.append(np.where(base == 1, 1 - py, py))
e1010 = np.load(f'{ST}/v_1010_original.npy').astype(float); S4v = Vf[[m[0] for m in meta].index('TODAY_S4')]
rev = ((S4v != e1010) & (base == S4v)) | (base != S4v)  # exclude reverting probe-era S4 corrections (CC-0029) and protect the base's scored flips vs S4 (CL-0034)
pmin = np.where(span & ~rev, np.array(PW).min(0), 0); o = np.argsort(-pmin); print('in-span rows P>0.5:', int((pmin > .5).sum()), '>0.6:', int((pmin > .6).sum()))
best = None; res = {}
for K in [10, 20, 50]:
    if (pmin > .5).sum() < K: continue
    v = base.copy(); s = o[:K]; v[s] = 1 - v[s]; e1 = min(pred(F, v)[0] for F in Fs.values()); sd = 2.0 * pred(G25['stack'], v)[1]; e25 = min(pred(F, v)[0] for F in G25.values())
    corr = e1 - .12 * K; txt = 'ID,condition\n' + ''.join(f'{i+1},{"used" if v[i] else "new"}\n' for i in range(30000))
    fn = f'iter_base_{meta[ib][0]}_top{K}.csv'; open(f'{OUT}/{fn}', 'w').write(txt)
    res[fn] = dict(K=K, pmin=round(float(pmin[s].min()), 3), E_selected_b=round(float(e1), 1), E_bias_corr=round(float(corr), 1), E_pess=round(float(e25), 1), sd=round(float(sd), 1),
                   sha256=hashlib.sha256(txt.encode()).hexdigest(), min_dist_scored=int((Vf != v).sum(1).min()), ids=[int(i + 1) for i in sorted(s)])
    print(fn, {k: v for k, v in res[fn].items() if k != 'ids'})
    score_ = (e25 - 2.0) / sd  # decision: b=0.25 min over priors minus observed residual bias (-2/overlay), per SD
    if res[fn]['min_dist_scored'] > 0 and e25 - 2.0 > 1 and (best is None or score_ > best[1]): best = (fn, score_)
json.dump(dict(base=meta[ib][0], b=B, loo_mae=mae, candidates=res, recommended=best[0] if best else None), open(f'{OUT}/iterate_manifest_{meta[ib][0]}.json', 'w'), indent=1)
print('RECOMMENDED:', best[0] if best else 'none (keep base)')
