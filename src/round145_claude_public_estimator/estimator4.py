import json, numpy as np, os
V = np.load('V4.npy').astype(float); meta = json.load(open('V4_meta.json')); NP = int(os.environ.get('NPUB', '20998'))
h = np.array([round(m[3] * NP) for m in meta], float); i4 = [m[0] for m in meta].index(os.environ.get('BASE', 'TODAY_TOP30')); s4 = V[i4]
def fit(p, idx, R=0.35):
    frac = NP / 30000; mu = frac * (2 * p - 1); s2 = frac - mu ** 2; D = V[idx] - s4; d = h[idx] - h[i4]
    cols = np.where(np.abs(D).sum(0) > 0)[0]; Dc = D[:, cols]; Sinv = np.linalg.inv((Dc * s2[cols]) @ Dc.T + R * np.eye(len(idx)))
    return dict(mu=mu, s2=s2, cols=cols, Dc=Dc, Sinv=Sinv, r=d - Dc @ mu[cols])
def predict(F, c):
    g = c - s4; k = (g[F['cols']] * F['s2'][F['cols']]) @ F['Dc'].T
    return g @ F['mu'] + k @ F['Sinv'] @ F['r'], np.sqrt(max((g ** 2) @ F['s2'] - k @ F['Sinv'] @ k, 1e-9))
def postmean(F):
    mu = F['mu'].copy(); c = F['cols']; mu[c] += F['s2'][c] * (F['Dc'].T @ (F['Sinv'] @ F['r'])); return mu
