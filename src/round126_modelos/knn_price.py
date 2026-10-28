"""Price relative to the title neighbours (same product), split by the neighbours' label.
A used item tends to be cheaper than new listings of the same product. Uses the OOF neighbour
indices saved by knn_title.py (train rows only see neighbours from other folds)."""
import os, numpy as np, pandas as pd

D = os.environ.get("AP2026Q2_DATA_DIR", "data/ap_2026_q_2").rstrip("/") + "/"
P = pd.read_pickle(D + "feats.pkl"); F = P["F"]; y = P["y"].astype(float); ntr = len(y)
nb = np.load(D + "knn_title_neighbors.npz"); NI, NS = nb["NI"], nb["NS"]
lp = np.log1p(pd.to_numeric(F["price"], errors="coerce").fillna(0).values)
L = y[NI]; NP = lp[NI]
out = {}
for k, smin in ((10, 0.0), (30, 0.0), (30, 0.5)):
    m = np.zeros_like(NS, bool); m[:, :k] = True; m &= NS >= smin
    for lab, nm in ((1, "used"), (0, "new")):
        mm = m & (L == lab)
        cnt = mm.sum(1)
        med = np.array([np.median(NP[i][mm[i]]) if cnt[i] else np.nan for i in range(len(NP))])
        out[f"kp_{nm}_med_k{k}_s{smin}"] = med
        out[f"kp_{nm}_cnt_k{k}_s{smin}"] = cnt
        out[f"kp_diff_{nm}_k{k}_s{smin}"] = lp - med
    out[f"kp_gap_k{k}_s{smin}"] = out[f"kp_new_med_k{k}_s{smin}"] - out[f"kp_used_med_k{k}_s{smin}"]
    allm = np.array([np.median(NP[i][m[i]]) if m[i].any() else np.nan for i in range(len(NP))])
    out[f"kp_diff_all_k{k}_s{smin}"] = lp - allm
pd.DataFrame(out).to_pickle(D + "knn_price.pkl")
print("knn_price features", len(out))
