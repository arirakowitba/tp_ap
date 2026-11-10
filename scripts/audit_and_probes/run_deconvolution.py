import json, csv, os, sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scratch.analyze_all_vectors import vectors, anchor

print("Running linear deconvolution on all 116 vectors...")

M = len(vectors)
N = 30000

# Anchor vector
anchor_hits = anchor['hits']
anchor_vec = np.array([1 if anchor['vec'][i] == 'new' else 0 for i in range(1, N + 1)], dtype=np.int8)

# Construct Design matrix X: M x N
# X[s, i] = +1 if s predicts new and anchor predicts used
# X[s, i] = -1 if s predicts used and anchor predicts new
# X[s, i] = 0 if same
X = np.zeros((M, N), dtype=np.float64)
y_delta = np.zeros(M, dtype=np.float64)

for s_idx, v in enumerate(vectors):
    y_delta[s_idx] = v['hits'] - anchor_hits
    for i in range(1, N + 1):
        pred = 1 if v['vec'][i] == 'new' else 0
        if pred == 1 and anchor_vec[i - 1] == 0:
            X[s_idx, i - 1] = 1.0
        elif pred == 0 and anchor_vec[i - 1] == 1:
            X[s_idx, i - 1] = -1.0

# Intervened items
intervened_mask = (X != 0).any(axis=0)
intervened_indices = np.where(intervened_mask)[0]
n_int = len(intervened_indices)
print(f"Total intervened items across all 116 vectors: {n_int}")

X_int = X[:, intervened_indices]

# Prior on w:
# Anchor accuracy is 0.94761
# For anchor=new (anchor_vec == 1), prior w_0 = +0.70 * (2 * 0.94761 - 1) = +0.62665
# For anchor=used (anchor_vec == 0), prior w_0 = -0.70 * (2 * 0.94761 - 1) = -0.62665
prior_w0 = np.where(anchor_vec == 1, 0.62665, -0.62665)
prior_w0_int = prior_w0[intervened_indices]

# Bayesian MAP / Dual Ridge solve:
# w = w_0 + Sigma_0 X^T (X Sigma_0 X^T + sigma^2 I)^(-1) (y - X w_0)
sigma0_sq = 0.25**2
sigma_sq = 0.5**2 # observation noise / rounding

K = X_int @ X_int.T * sigma0_sq + sigma_sq * np.eye(M)
y_res = y_delta - X_int @ prior_w0_int
alpha = np.linalg.solve(K, y_res)
w_bayesian_int = prior_w0_int + sigma0_sq * (X_int.T @ alpha)

# Bounded Least Squares on intervened items: bounds [-0.70, +0.70]
res_lsq = lsq_linear(X_int, y_delta, bounds=(-0.70, 0.70), lsmr_tol='auto', verbose=0)
w_lsq_int = res_lsq.x

# Map to posterior P(new | probes)
# p_new = clip(0.5 + w / 1.40, 0.001, 0.999)
p_new_bayesian = np.clip(0.5 + prior_w0 / 1.40, 0.001, 0.999) # initialize all with prior
p_new_bayesian[intervened_indices] = np.clip(0.5 + w_bayesian_int / 1.40, 0.001, 0.999)

p_new_lsq = np.clip(0.5 + prior_w0 / 1.40, 0.001, 0.999)
p_new_lsq[intervened_indices] = np.clip(0.5 + w_lsq_int / 1.40, 0.001, 0.999)

print("\nPosterior summary for intervened items:")
print(f"Intervened items with P(new | Bayesian) > 0.95: {np.sum(p_new_bayesian[intervened_indices] > 0.95)}")
print(f"Intervened items with P(new | Bayesian) < 0.05: {np.sum(p_new_bayesian[intervened_indices] < 0.05)}")
print(f"Intervened items with 0.40 <= P(new | Bayesian) <= 0.60: {np.sum((p_new_bayesian[intervened_indices] >= 0.40) & (p_new_bayesian[intervened_indices] <= 0.60))}")

# Check correlation between Bayesian and LSQ
corr = np.corrcoef(w_bayesian_int, w_lsq_int)[0, 1]
print(f"\nCorrelation between Bayesian MAP and Bounded LSQ effects: {corr:.4f}")

# Look at items with highest effect change vs anchor
diff_from_prior = np.abs(p_new_bayesian[intervened_indices] - (0.5 + prior_w0_int / 1.40))
top_changes = np.argsort(diff_from_prior)[::-1][:20]

print("\nTop 20 items where probes most strongly disagree with Anchor EVT-143:")
with open("data/ap_2026_q_2/raw/test_data.jsonlines", "r", encoding="utf-8") as f:
    test_lines = [json.loads(line) for line in f]

for rank, idx in enumerate(top_changes, 1):
    test_idx = intervened_indices[idx]
    iid = test_idx + 1
    it = test_lines[test_idx]
    anc_cond = "new" if anchor_vec[test_idx] == 1 else "used"
    p_b = p_new_bayesian[test_idx]
    p_l = p_new_lsq[test_idx]
    n_flips = np.sum(X[:, test_idx] != 0)
    print(f"{rank:2d}. ID {iid:5d} | Anchor: {anc_cond:4s} | P(new|Bayes): {p_b:.4f} | P(new|LSQ): {p_l:.4f} | Flips: {n_flips:2d} | ${it.get('price', 0):7.1f} | {it.get('title', '')[:40]}")
