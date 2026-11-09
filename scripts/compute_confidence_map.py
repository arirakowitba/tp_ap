"""Compute exact mathematical confidence map for 95% and 99% confidence."""

import numpy as np

gamma = 2.9557  # Codex prequential stress factor
z_95 = 1.645
z_99 = 2.326
N_pub = 20998
base_hits = 19896

print("=" * 110)
print(f"MATHEMATICAL CONFIDENCE MAPPING (Prequential Stress Factor gamma = {gamma:.4f})")
print(f"Base: EVT-132 (19,896 hits = 0.94752) | Target: 0.95000 requires +53.0 net hits")
print("=" * 110)
print(f"{'Precision p':12s} | {'K_pub':6s} | {'K_total':7s} | {'E[Gain]':7s} | {'Sigma_adj':9s} | {'Gain 95%':8s} | {'Score 95%':10s} | {'Gain 99%':8s} | {'Score 99%':10s}")
print("-" * 110)

for p in [0.70, 0.75, 0.77, 0.80, 0.83, 0.85, 0.88, 0.90, 0.93, 0.95]:
    for K_pub in [50, 75, 100, 125, 150, 200, 250, 300, 350, 400, 500]:
        K_tot = int(K_pub * 30000 / 20998)
        exp_gain = K_pub * (2 * p - 1)
        sigma = np.sqrt(K_pub * 4 * p * (1 - p))
        sigma_adj = sigma * gamma
        gain_95 = exp_gain - z_95 * sigma_adj
        gain_99 = exp_gain - z_99 * sigma_adj
        score_95 = (base_hits + gain_95) / N_pub
        score_99 = (base_hits + gain_99) / N_pub

        # Flag if passes target
        flag_95 = "PASS" if gain_95 >= 53.0 else "    "
        flag_99 = "PASS" if gain_99 >= 53.0 else "    "

        if gain_95 >= 10.0 and gain_95 <= 120.0:
            print(f"{p*100:5.1f}%       | {K_pub:6d} | {K_tot:7d} | {exp_gain:+7.1f} | {sigma_adj:9.2f} | {gain_95:+7.1f} {flag_95} | {score_95:10.5f} | {gain_99:+7.1f} {flag_99} | {score_99:10.5f}")

print("=" * 110)
