import numpy as np
from scipy.stats import hypergeom

N_test = 30000
N_pub = 20998
pi = N_pub / N_test

ks = [25, 35, 60, 75, 100, 150, 160, 250, 500, 1000]

print("=== FEASIBILITY UNDER EVT-143 (S0 = 0.94761, G_req = +51) ===")
print(f"{'K_test':<8} | {'Max Gain':<10} | {'Max Score':<10} | {'E[K_pub]':<10} | {'P(K_pub>=51)':<14} | {'F_req (all pub)':<16} | {'Prec (all pub)':<16} | {'Status'}")
print("-" * 105)
for k in ks:
    max_gain = k
    max_score = 0.94761 + k / N_pub
    exp_kpub = k * pi
    p_kpub_ge_51 = 1.0 - hypergeom.cdf(50, N_test, N_pub, k) if k >= 51 else 0.0
    
    if max_gain < 51:
        status = "INFEASIBLE (Deterministic Upper Bound < 0.95000)"
        f_req = "N/A"
        prec = "N/A"
    else:
        f_req = int(np.ceil((k + 51) / 2))
        prec = f"{f_req / k * 100:.1f}%"
        status = "FEASIBLE" if max_score >= 0.95000 else "INFEASIBLE"
    print(f"{k:<8} | {max_gain:<10} | {max_score:<10.5f} | {exp_kpub:<10.1f} | {p_kpub_ge_51:<14.4f} | {str(f_req):<16} | {prec:<16} | {status}")

print("\n=== FEASIBILITY UNDER 0.94661 (S0 = 0.94661, G_req = +71) ===")
print(f"{'K_test':<8} | {'Max Gain':<10} | {'Max Score':<10} | {'E[K_pub]':<10} | {'P(K_pub>=71)':<14} | {'F_req (all pub)':<16} | {'Prec (all pub)':<16} | {'Status'}")
print("-" * 105)
for k in ks:
    max_gain = k
    max_score = 0.94661 + k / N_pub
    exp_kpub = k * pi
    p_kpub_ge_71 = 1.0 - hypergeom.cdf(70, N_test, N_pub, k) if k >= 71 else 0.0
    
    if max_gain < 71:
        status = "INFEASIBLE (Deterministic Upper Bound < 0.95000)"
        f_req = "N/A"
        prec = "N/A"
    else:
        f_req = int(np.ceil((k + 71) / 2))
        prec = f"{f_req / k * 100:.1f}%"
        status = "FEASIBLE" if max_score >= 0.95000 else "INFEASIBLE"
    print(f"{k:<8} | {max_gain:<10} | {max_score:<10.5f} | {exp_kpub:<10.1f} | {p_kpub_ge_71:<14.4f} | {str(f_req):<16} | {prec:<16} | {status}")

# Also calculate realistic expected required precision under E[K_pub]
print("\n=== REALISTIC HYPERGEOMETRIC SAMPLING: AT E[K_pub] FOR EVT-143 (G=+51) ===")
print(f"{'K_test':<8} | {'E[K_pub]':<10} | {'F_req at E[K_pub]':<20} | {'Required Prec at E[K_pub]':<26} | {'Status'}")
print("-" * 80)
for k in ks:
    exp_kpub = k * pi
    if exp_kpub < 51:
        print(f"{k:<8} | {exp_kpub:<10.1f} | {'N/A (< 51)':<20} | {'N/A (Exp gain < +51)':<26} | INFEASIBLE on average")
    else:
        f_req_exp = int(np.ceil((exp_kpub + 51) / 2))
        prec_exp = f"{f_req_exp / exp_kpub * 100:.1f}%"
        print(f"{k:<8} | {exp_kpub:<10.1f} | {str(f_req_exp):<20} | {prec_exp:<26} | FEASIBLE")
