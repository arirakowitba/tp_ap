"""Exact mathematical calculation of P(Public Score >= 0.95000) under Hypergeometric Sampling.

Parameters:
- N_total = 30,000
- N_public = 20,998
- Anchor hits = 19,898 (score = 0.947614)
- Target hits = ceil(0.95000 * 20998) = 19,949
- Required net public hits Delta_K = +51
- Candidate pool: Ranked flips from results/iterative_probe_crossing_20261110/crossed_confidences_30k.csv
"""

import json, os
import numpy as np
import pandas as pd
from scipy.stats import norm

# Load candidate flips
df_crossed = pd.read_csv('results/iterative_probe_crossing_20261110/crossed_confidences_30k.csv')
dis_df = df_crossed[df_crossed['disagrees_with_anchor']].sort_values(by='posterior_confidence', ascending=False).reset_index(drop=True)

N_TOTAL = 30000
N_PUB = 20998
PUB_FRAC = N_PUB / N_TOTAL # 0.6999333333333333
ANCHOR_HITS = 19898
REQ_HITS = int(np.ceil(0.95000 * N_PUB)) # 19949
REQ_NET = REQ_HITS - ANCHOR_HITS # 51

confidences = dis_df['posterior_confidence'].values
n_candidates = len(confidences)

print(f"Total candidates: {n_candidates}")
print(f"Anchor hits: {ANCHOR_HITS} (score: {ANCHOR_HITS/N_PUB:.5f})")
print(f"Target hits: {REQ_HITS} (score: {REQ_HITS/N_PUB:.5f})")
print(f"Required net gain on public: +{REQ_NET} hits")

results = []

# Sweep M from 10 to min(400, n_candidates)
for M in range(10, min(350, n_candidates) + 1, 5):
    sub_conf = confidences[:M]
    
    # Each flip i has probability p_i of being correct (fix: +1) and 1 - p_i of being incorrect (break: -1)
    # If in public, net gain is (2 * Z_i - 1), where Z_i ~ Bernoulli(p_i).
    # Since public is sampled without replacement (Hypergeometric):
    # Mean of public hits:
    # E[G] = (N_PUB / N_TOTAL) * sum_{i=1}^M (2 * p_i - 1)
    exp_net_test = np.sum(2.0 * sub_conf - 1.0)
    mu_G = PUB_FRAC * exp_net_test
    
    # Variance under Hypergeometric drawing:
    # Let X_i = (2 Z_i - 1) be net gain if drawn into public.
    # E[X_i] = 2 p_i - 1
    # Var(X_i) = 4 p_i (1 - p_i)
    # Total variance of G on finite population without replacement:
    # Using law of total variance:
    # Var(G) = E[Var(G | Z)] + Var(E[G | Z])
    # For hypergeometric sample of size n from N:
    # Var(sum_{i=1}^M I_i X_i) where sum I_i = n
    # Var(I_i) = n/N * (1 - n/N)
    # Cov(I_i, I_j) = - n/N * (1 - n/N) / (N - 1) for i != j
    # Var(G) = sum_i Var(I_i X_i) + sum_{i != j} Cov(I_i X_i, I_j X_j)
    
    # Compute exactly:
    # E[I_i X_i] = (n/N) (2 p_i - 1)
    # E[(I_i X_i)^2] = E[I_i X_i^2] = (n/N) * 1 = n/N (since X_i in {-1, 1}, X_i^2 = 1)
    # Var(I_i X_i) = n/N - (n/N)^2 (2 p_i - 1)^2
    
    # For i != j:
    # E[I_i X_i I_j X_j] = E[I_i I_j] E[X_i X_j]
    # E[I_i I_j] = (n/N) * ((n - 1) / (N - 1))
    # E[X_i X_j] = (2 p_i - 1) * (2 p_j - 1)
    # Cov(I_i X_i, I_j X_j) = (n/N) * ((n - 1)/(N - 1)) (2 p_i - 1)(2 p_j - 1) - (n/N)^2 (2 p_i - 1)(2 p_j - 1)
    #                       = (n/N) [ (n - 1)/(N - 1) - n/N ] (2 p_i - 1)(2 p_j - 1)
    #                       = - (n/N) * (1 - n/N) / (N - 1) * (2 p_i - 1)(2 p_j - 1)
    
    var_terms = PUB_FRAC - (PUB_FRAC ** 2) * ((2.0 * sub_conf - 1.0) ** 2)
    sum_var = np.sum(var_terms)
    
    cov_factor = - PUB_FRAC * (1.0 - PUB_FRAC) / (N_TOTAL - 1.0)
    mu_X = 2.0 * sub_conf - 1.0
    # sum_{i != j} mu_X[i] * mu_X[j] = (sum mu_X)^2 - sum (mu_X^2)
    sum_mu_X = np.sum(mu_X)
    sum_mu_X2 = np.sum(mu_X ** 2)
    sum_cross = (sum_mu_X ** 2) - sum_mu_X2
    sum_cov = cov_factor * sum_cross
    
    sigma2_G = sum_var + sum_cov
    sigma_G = np.sqrt(max(1e-6, sigma2_G))
    
    # Continuity correction: P(G >= 51) = P(G > 50.5)
    z = (mu_G - 50.5) / sigma_G
    prob_reach_095 = norm.sf(50.5, loc=mu_G, scale=sigma_G)
    
    # Expected public score
    exp_score = (ANCHOR_HITS + mu_G) / N_PUB
    
    results.append({
        'M': M,
        'mean_conf': float(np.mean(sub_conf)),
        'mu_G': float(mu_G),
        'sigma_G': float(sigma_G),
        'z_score': float(z),
        'prob_reach_095': float(prob_reach_095),
        'exp_score': float(exp_score)
    })

res_df = pd.DataFrame(results)

print("\n--- RESULTS TABLE (M, E[Hits], Sigma, Prob >= 0.95000) ---")
print(f"{'M':>4} | {'MeanConf':>8} | {'E[Gain]':>8} | {'Sigma':>7} | {'Z-score':>7} | {'P(Score>=0.95)':>14} | {'ExpScore':>9}")
print("-" * 75)
for idx, r in res_df.iterrows():
    if int(r['M']) in [20, 42, 75, 102, 130, 150, 165, 180, 200, 220, 240, 260, 280, 300]:
        print(f"{int(r['M']):4d} | {r['mean_conf']*100:7.2f}% | {r['mu_G']:+8.2f} | {r['sigma_G']:7.2f} | {r['z_score']:+7.2f} | {r['prob_reach_095']*100:13.2f}% | {r['exp_score']:9.5f}")

# Find maximum probability
best_row = res_df.loc[res_df['prob_reach_095'].idxmax()]
print("\n" + "=" * 75)
print(f"MAXIMUM PROBABILITY OF REACHING 0.95000:")
print(f"  Flips count M*: {int(best_row['M'])}")
print(f"  Mean Confidence: {best_row['mean_conf']*100:.2f}%")
print(f"  Expected Public Net Gain: {best_row['mu_G']:+.2f} hits (target is +51)")
print(f"  Standard Deviation Sigma: {best_row['sigma_G']:.2f} hits")
print(f"  Z-score: {best_row['z_score']:+.2f}")
print(f"  Probability P(Score >= 0.95000): {best_row['prob_reach_095']*100:.2f}%")
print(f"  Expected Score: {best_row['exp_score']:.5f}")
print("=" * 75)

# Monte Carlo verification on M*
print("\nRunning 100,000 Monte Carlo simulations at M* to verify exact distribution...")
M_star = int(best_row['M'])
sub_c = confidences[:M_star]

N_SIMS = 100000
# Sample whether each of M* items is in public (hypergeometric)
# Drawing M items from population of N_TOTAL where N_PUB are 1:
# The number of public items among M follows Hypergeometric(N_TOTAL, N_PUB, M_star)
# But since each item has distinct p_i, we can simulate by permuting indices:
np.random.seed(42)

sim_gains = []
for _ in range(N_SIMS):
    # Vector of length M: is item in public?
    # Generate random permutation or random choice
    in_pub = np.random.rand(M_star) < PUB_FRAC
    # Correctness of flip
    is_correct = np.random.rand(M_star) < sub_c
    # Net contribution: if in_pub: (+1 if correct else -1)
    g = np.sum(np.where(in_pub, np.where(is_correct, 1, -1), 0))
    sim_gains.append(g)

sim_gains = np.array(sim_gains)
mc_p = (sim_gains >= REQ_NET).mean()
print(f"Monte Carlo Probability (100k runs): {mc_p*100:.2f}%")
print(f"Monte Carlo Mean Gain: {sim_gains.mean():+.2f}")
print(f"Monte Carlo Std Dev: {sim_gains.std():.2f}")
print(f"Monte Carlo 5th percentile: {np.percentile(sim_gains, 5):.1f}")
print(f"Monte Carlo 50th percentile (median): {np.percentile(sim_gains, 50):.1f}")
print(f"Monte Carlo 95th percentile: {np.percentile(sim_gains, 95):.1f}")
