#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
import numpy as np
from scripts.calculate_confidence_curves import safe_cands, p_te, get_emp_q, p_pub
from scipy.stats import norm

print("=== SEARCHING OPTIMAL K FOR SCORE >= 0.95000 AND PODIUM ===")
print(f"{'K':<5} | {'Avg q':<7} | {'E[Gain Pub]':<11} | {'Exp Score':<10} | {'Sigma':<7} | {'Conf(>=0.95000)':<16} | {'Conf(Podium)':<14}")
print("-" * 75)

for K in range(10, 320, 15):
    sel = safe_cands[:K]
    q_sel = get_emp_q(p_te[sel])
    bar_q = float(q_sel.mean())
    exp_gain = float(K * p_pub * (2.0 * bar_q - 1.0))
    exp_score = (19896.0 + exp_gain) / 20998.0
    var_gain = float(K * p_pub * (1.0 - p_pub * (2.0 * bar_q - 1.0)**2))
    sigma = np.sqrt(var_gain)
    
    z_podium = (exp_gain - 30.0) / sigma if sigma > 0 else 0.0
    z_9500 = (exp_gain - 53.0) / sigma if sigma > 0 else 0.0
    
    conf_podium = norm.cdf(z_podium) * 100
    conf_9500 = norm.cdf(z_9500) * 100
    
    print(f"{K:<5} | {bar_q:<7.4f} | +{exp_gain:<10.1f} | {exp_score:<10.5f} | {sigma:<7.2f} | {conf_9500:<15.2f}% | {conf_podium:<13.2f}%")
