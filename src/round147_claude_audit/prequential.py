"""CC-0018: prequential check of the public estimator. Vectors ordered by ledger date (event id as tiebreak); for each vector k
after the first 10 dates, fit ONLY on vectors scored strictly before k's date, predict k's hits delta vs the earliest vector; report
z coverage (|z|<=1,2 with SD x2.7) and errors. Quarantined: identical vectors with contradictory scores (already removed)."""
import json, io, subprocess, numpy as np, pandas as pd
V = np.load('/home/claude/work/r145/V.npy').astype(float); meta = json.load(open('/home/claude/work/r145/V_meta.json')); h = np.array([m[4] for m in meta], float)
led = pd.read_csv(io.StringIO(subprocess.run(['git', 'show', 'origin/antigravity/overnight-recovery-coordination-20261030:results/exhaustive_evidence_and_aggregate_system_ag0154/canonical_reconciled_ledger_ag0154.csv'], capture_output=True, text=True).stdout))
dmap = dict(zip(led.event_id, led.date)); dmap.update({'TODAY_S4': '2026-11-10', 'TODAY_S1': '2026-11-14', 'TODAY_S2': '2026-11-14', 'TODAY_S3': '2026-11-14'})
dates = np.array([dmap[m[0]] for m in meta]); order = np.lexsort((np.arange(len(V)), dates))
q = np.clip(np.load('/home/claude/work/r145/p_havg_plus.npy'), 1e-4, 1 - 1e-4); p = 1 / (1 + np.exp(-.25 * np.log(q / (1 - q)))); fr = 20998 / 30000
mu = fr * (2 * p - 1); s2 = fr - mu ** 2
rows = []
for k in order:
    past = [i for i in order if dates[i] < dates[k]]
    if len(set(dates[past])) < 3: continue
    b = past[0]; idx = past[1:]; D = V[idx] - V[b]; d = h[idx] - h[b]; cols = np.where(np.abs(D).sum(0) > 0)[0]; Dc = D[:, cols]
    Sinv = np.linalg.inv((Dc * s2[cols]) @ Dc.T + .35 * np.eye(len(idx)))
    g = V[k] - V[b]; kk = (g[cols] * s2[cols]) @ Dc.T; e = g @ mu + kk @ Sinv @ (d - Dc @ mu[cols]); sd = 2.7 * np.sqrt(max((g ** 2) @ s2 - kk @ Sinv @ kk, 1e-9))
    rows.append(dict(event=meta[k][0], date=dates[k], n_past=len(past), true=float(h[k] - h[b]), est=round(float(e), 1), sd=round(float(sd), 1), z=round(float((h[k] - h[b] - e) / sd), 2)))
z = np.array([r['z'] for r in rows]); err = np.array([r['true'] - r['est'] for r in rows])
res = dict(n=len(rows), cover_1sd=float((abs(z) <= 1).mean()), cover_2sd=float((abs(z) <= 2).mean()), mean_err=float(err.mean()), mae=float(abs(err).mean()), rows=rows)
json.dump(res, open('/home/claude/tp_ap/results/round147_claude_audit/prequential.json', 'w'), indent=1)
print({k: v for k, v in res.items() if k != 'rows'}); [print(r) for r in rows[-8:]]
