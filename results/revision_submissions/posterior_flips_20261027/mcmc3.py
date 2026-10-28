"""Exact-constraint MCMC at atom level.
Each atom j (rows sharing the same set of submissions that differ from 48) has
u_j rows that are public with 48 wrong (+1 if flipped) and d_j rows public with
48 right (-1 if flipped). Constraints: for every scored submission s,
sum_{j in s} (u_j - d_j) = K_s. Moves preserve constraints exactly."""
import json, sys, numpy as np
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
from math import lgamma, log
from scipy.optimize import milp, LinearConstraint, Bounds

seed = int(sys.argv[1]); iters = int(sys.argv[2])
qa, qb = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (1.0, 1.0)
rng = np.random.default_rng(seed)
A = json.load(open("atoms.json")); K = {int(k): v for k, v in A["K"].items()}
subs = sorted(n for n in K if n != 48)
atoms = [(sorted(k), v) for k, v in A["atoms"]]; J = len(atoms)
n = np.array([len(v) for _, v in atoms])
V = np.load("nullbasis.npy")
nzs = [np.nonzero(v)[0] for v in V]

# feasible random start
nv = 2 * J; cons = []; lo = []; hi = []
for j in range(J):
    r = np.zeros(nv); r[j] = 1; r[J + j] = 1; cons.append(r); lo.append(0); hi.append(n[j])
for s in subs:
    r = np.zeros(nv)
    for j, (k, _) in enumerate(atoms):
        if s in k: r[j] = 1; r[J + j] = -1
    cons.append(r); lo.append(K[s]); hi.append(K[s])
res = milp(rng.normal(size=nv), constraints=LinearConstraint(np.array(cons), lo, hi),
           integrality=np.ones(nv), bounds=Bounds(0, np.r_[n, n]))
x = np.round(res.x).astype(int); u = x[:J].copy(); d = x[J:].copy()
Am = np.array(cons[J:])[:, :J]
assert np.all(Am @ (u - d) == np.array([K[s] for s in subs]))

pp = 0.7
def logp(uj, dj, nj, la, lb, lc):
    return -lgamma(uj + 1) - lgamma(dj + 1) - lgamma(nj - uj - dj + 1) + uj * la + dj * lb + (nj - uj - dj) * lc

q = 0.35
acc_u = np.zeros(J); acc_d = np.zeros(J); acc_q = []; ns = 0; acc = [0, 0]
burn = iters // 4
for it in range(iters):
    la, lb, lc = log(pp * q), log(pp * (1 - q)), log(1 - pp)
    # (a) public/private swaps inside atoms
    for j in rng.integers(J, size=J):
        dl = 1 if rng.random() < .5 else -1
        nu, nd = u[j] + dl, d[j] + dl
        if nu < 0 or nd < 0 or nu + nd > n[j]: continue
        if log(rng.random()) < logp(nu, nd, n[j], la, lb, lc) - logp(u[j], d[j], n[j], la, lb, lc):
            u[j], d[j] = nu, nd
    # (b) null-space moves across atoms
    for _ in range(3 * len(V)):
        a = rng.integers(len(V))
        if rng.random() < .3:
            b = rng.integers(len(V)); v = V[a] + (1 if rng.random() < .5 else -1) * V[b]; idx = np.nonzero(v)[0]
        else:
            v = V[a]; idx = nzs[a]
        sg = 1 if rng.random() < .5 else -1
        nu = u.copy(); nd = d.copy(); ok = True; dlp = 0.0
        for j in idx:
            delta = sg * v[j]
            if rng.random() < .5: nu[j] += delta
            else: nd[j] -= delta
            if nu[j] < 0 or nd[j] < 0 or nu[j] + nd[j] > n[j]: ok = False; break
            dlp += logp(nu[j], nd[j], n[j], la, lb, lc) - logp(u[j], d[j], n[j], la, lb, lc)
        acc[1] += 1
        if ok and log(rng.random()) < dlp:
            u, d = nu, nd; acc[0] += 1
    q = rng.beta(qa + u.sum(), qb + d.sum())
    if it >= burn:
        acc_u += u; acc_d += d; acc_q.append(q); ns += 1
    if it % 500 == 0:
        print(it, "q", round(q, 3), "acc", round(acc[0] / max(acc[1], 1), 3), file=sys.stderr, flush=True)
assert np.all(Am @ (u - d) == np.array([K[s] for s in subs]))
rows = []; up = []; dn = []
for j, (_, ids) in enumerate(atoms):
    for i in ids:
        rows.append(i); up.append(acc_u[j] / n[j]); dn.append(acc_d[j] / n[j])
np.savez(f"post3_{seed}.npz", rows=np.array(rows), up=np.array(up), dn=np.array(dn),
         w=np.zeros(len(rows)), n=1, q=np.array(acc_q), ns=ns)
print("done", ns, "q mean", np.mean(acc_q), file=sys.stderr)
