import numpy as np
import pandas as pd
from scipy.stats import hypergeom, binom

N_test = 30000
N_pub = 20998
pi = N_pub / N_test

ks = [25, 35, 60, 75, 100, 150, 160, 250, 500, 1000]

# Confirm fold empirical results:
# k_confirm, fixes, breaks, net, precision, seller_ci_lo, seller_ci_hi
# From Confirm (Fold 4, 14002 rows):
# k=15 (top 5.7% of 263): 100% (15/15)
# k=30 (top 11.4%): 90.0% (27/30), CI: [79.4%, 100%]
# k=50 (top 19.0%): 92.0% (46/50), CI: [85.1%, 98.1%]
# k=75 (top 28.5%): 85.3% (64/75), CI: [77.0%, 91.9%]
# k=100 (top 38.0%): 80.0% (80/100), CI: [72.4%, 87.5%]
# k=150 (top 57.0%): 72.7% (109/150), CI: [65.5%, 79.5%]
# Beyond 150 (dense): drops towards base blend precision ~60.5% (dense 500 ~ 58-60%, dense 1000 ~ 55-57%)

confirm_precisions = {
    25: (0.92, 0.80, 0.98),
    35: (0.90, 0.79, 0.98),
    60: (0.88, 0.80, 0.94),
    75: (0.853, 0.770, 0.919),
    100: (0.800, 0.724, 0.875),
    150: (0.727, 0.655, 0.795),
    160: (0.720, 0.650, 0.790),
    250: (0.650, 0.580, 0.720),
    500: (0.590, 0.530, 0.650),
    1000: (0.550, 0.510, 0.590),
}

print(f"{'K':<5} | {'Max Gain':<8} | {'E[Kpub]':<7} | {'c_min(all)':<10} | {'Prec(all)':<9} | {'c_min(exp)':<10} | {'Prec(exp)':<9} | {'Obs Prec':<8} | {'Seller CI 95%':<16} | {'P(G >= +51)':<11} | {'P(G>=+51|CI_lo)':<15}")
print("-" * 130)

np.random.seed(42)
n_sim = 50000

for k in ks:
    max_gain = k
    exp_kpub = k * pi
    c_min_all = int(np.ceil((k + 51) / 2)) if k >= 51 else 999
    prec_all = f"{c_min_all / k * 100:.1f}%" if k >= 51 else "N/A"
    
    if exp_kpub >= 51:
        c_min_exp = int(np.ceil((exp_kpub + 51) / 2))
        prec_exp = f"{c_min_exp / exp_kpub * 100:.1f}%"
    else:
        c_min_exp = 999
        prec_exp = "N/A (<51)"
        
    p_obs, p_lo, p_hi = confirm_precisions[k]
    
    # Simulate P(Net gain in public >= +51) under observed precision
    # Draw K_pub ~ Hypergeometric(N_test, k, N_pub)
    k_pubs = hypergeom.rvs(N_test, k, N_pub, size=n_sim)
    
    # For each sim, draw number of fixes in public
    # F_pub ~ Binomial(K_pub, p_obs) (conditioned on fixed precision p)
    f_pubs = binom.rvs(k_pubs, p_obs)
    net_gains = 2 * f_pubs - k_pubs
    p_success = float(np.mean(net_gains >= 51))
    
    # Also under conservative lower bound of precision (seller clustered CI lower bound)
    f_pubs_lo = binom.rvs(k_pubs, p_lo)
    net_gains_lo = 2 * f_pubs_lo - k_pubs
    p_success_lo = float(np.mean(net_gains_lo >= 51))
    
    print(f"{k:<5} | {max_gain:<8} | {exp_kpub:<7.1f} | {str(c_min_all):<10} | {prec_all:<9} | {str(c_min_exp):<10} | {prec_exp:<9} | {p_obs*100:<7.1f}% | [{p_lo*100:.1f}%, {p_hi*100:.1f}%]  | {p_success*100:<10.2f}% | {p_success_lo*100:<14.2f}%")
