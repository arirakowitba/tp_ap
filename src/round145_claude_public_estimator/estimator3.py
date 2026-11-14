"""Linear-Gaussian public-score estimator. hits(v) = A + v.w, w_i = m_i(2y_i-1). Condition prior w ~ N(mu, diag s2) on all
scored vectors (differences vs S4). Predict any candidate's public hits delta vs S4 with SD. LOO calibration over scored vectors."""
import json, numpy as np
V = np.load('V3.npy').astype(float); meta = json.load(open('V3_meta.json'))
h = np.array([m[4] for m in meta], float); i4 = [m[0] for m in meta].index('TODAY_TOP30'); s4 = V[i4]
def build(p, frac=20998/30000):
    mu = frac * (2 * p - 1); s2 = frac - mu ** 2; return mu, s2
def fit(p, idx, R=0.35):
    mu, s2 = build(p); D = V[idx] - s4; d = h[idx] - h[i4]
    cols = np.where(np.abs(D).sum(0) > 0)[0]; Dc = D[:, cols]
    S = (Dc * s2[cols]) @ Dc.T + R * np.eye(len(idx)); Sinv = np.linalg.inv(S)
    return dict(mu=mu, s2=s2, cols=cols, Dc=Dc, Sinv=Sinv, r=d - Dc @ mu[cols])
def predict(F, c):
    g = c - s4; m0 = g @ F['mu']; v0 = (g ** 2) @ F['s2']
    k = (g[F['cols']] * F['s2'][F['cols']]) @ F['Dc'].T
    return m0 + k @ F['Sinv'] @ F['r'], np.sqrt(max(v0 - k @ F['Sinv'] @ k, 1e-9))
if __name__ == '__main__':
    import sys
    for name in sys.argv[1:]:
        p = np.load(name) if name.endswith('.npy') else np.load(name)['te']
        others = [i for i in range(len(V)) if i != i4]
        z = []
        for k in others:
            F = fit(p, [i for i in others if i != k] ); e, sd = predict(F, V[k]); z.append(((h[k] - h[i4]) - e) / sd)
        z = np.array(z); print(name, 'LOO z: mean %.2f sd %.2f |z|>2: %d/%d' % (z.mean(), z.std(), (abs(z) > 2).sum(), len(z)))
        F = fit(p, others); pv = (p >= .5).astype(float); e, sd = predict(F, pv)
        print('   pure model vs S4: flips %d  est delta %.1f +- %.1f  -> est public %.5f' % ((pv != s4).sum(), e, sd, (h[i4] + e) / 20998))
