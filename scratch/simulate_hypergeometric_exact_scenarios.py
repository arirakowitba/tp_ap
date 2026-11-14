"""Simulation and Analytical Verification of Hypergeometric Public Score Gain (N=30,000, n=20,998)
Corrected for CX-0202:
1. Exact hypergeometric variance: Var(k_pub) = K * f * (1 - K/N) * (N - n) / (N - 1).
2. Joint multivariate hypergeometric partition for Candidate B (warm + cold) with exact covariance.
3. Law of Total Variance for delta_pub: Var(Delta) = E[Var(Delta|k)] + Var(E[Delta|k]).
4. All metrics treated strictly as conditional sensitivity scenarios, NOT calibrated public probabilities.
"""

import os
import sys
import json
import numpy as np
from scipy.stats import multivariate_hypergeom, hypergeom

def run():
    print("=" * 80)
    print("EXACT HYPERGEOMETRIC & MULTIVARIATE HYPERGEOMETRIC VERIFICATION (CX-0202)")
    print("=" * 80)

    N_TEST = 30000
    N_PUB = 20998
    f = N_PUB / N_TEST # 0.6999333333333333
    ffac = (N_TEST - N_PUB) / (N_TEST - 1) # 0.3000766692223074

    print(f"Population N: {N_TEST} | Public Sample n: {N_PUB} | Fraction f: {f:.6f} | Finite Correction: {ffac:.6f}")

    N_SIM = 100000
    rng = np.random.default_rng(20261114)

    # Scenarios for flip precision: (p_warm, p_cold)
    scenarios = {
        "S1_Optimista (Warm=75%, Cold=55%)": (0.75, 0.55),
        "S2_Central (Warm=65%, Cold=50%)": (0.65, 0.50),
        "S3_Conservador (Warm=58%, Cold=46%)": (0.58, 0.46),
        "S4_Adverso (Warm=50%, Cold=42%)": (0.50, 0.42),
        "S5_Prequential_Retroceso (EVT-155 like: Warm=45%, Cold=35%)": (0.45, 0.35)
    }

    # Candidates:
    # 1. Candidate Clean (AG-0223): 161 warm flips, 0 cold
    # 2. Candidate A: 303 warm flips, 0 cold
    # 3. Candidate B: 303 warm flips, 76 cold flips (Total 379)
    # 4. Candidate C: 483 warm flips, 0 cold
    # 5. Disjoint Block B2 (B\A): 76 cold flips
    # 6. Disjoint Block B3 (C\A): 180 warm flips

    configs = {
        "Candidate_Clean_AG0223": {"K_warm": 161, "K_cold": 0},
        "Candidate_A_Warm_Route": {"K_warm": 303, "K_cold": 0},
        "Candidate_B_Composite":  {"K_warm": 303, "K_cold": 76},
        "Candidate_C_VOI_Frontier": {"K_warm": 483, "K_cold": 0},
        "Block_2_B_minus_A_Cold": {"K_warm": 0, "K_cold": 76},
        "Block_3_C_minus_A_Warm": {"K_warm": 180, "K_cold": 0},
    }

    results = {}

    for name, cfg in configs.items():
        Kw = cfg["K_warm"]
        Kc = cfg["K_cold"]
        Ktot = Kw + Kc

        # Exact Analytical Moments for k_pub
        var_kw = Kw * f * (1.0 - Kw / N_TEST) * ffac if Kw > 0 else 0.0
        var_kc = Kc * f * (1.0 - Kc / N_TEST) * ffac if Kc > 0 else 0.0
        cov_kw_kc = - (Kw * Kc / N_TEST) * f * ffac if (Kw > 0 and Kc > 0) else 0.0
        var_ktot = Ktot * f * (1.0 - Ktot / N_TEST) * ffac

        mean_kw = Kw * f
        mean_kc = Kc * f
        mean_ktot = Ktot * f

        sd_kw = np.sqrt(var_kw)
        sd_kc = np.sqrt(var_kc)
        sd_ktot = np.sqrt(var_ktot)

        # Simulation:
        if Kw > 0 and Kc > 0:
            # Joint Multivariate Hypergeometric
            m_vec = [Kw, Kc, N_TEST - Kw - Kc]
            rvs = multivariate_hypergeom.rvs(m=m_vec, n=N_PUB, size=N_SIM, random_state=rng)
            sim_kw = rvs[:, 0]
            sim_kc = rvs[:, 1]
        elif Kw > 0:
            sim_kw = rng.hypergeometric(ngood=Kw, nbad=N_TEST - Kw, nsample=N_PUB, size=N_SIM)
            sim_kc = np.zeros(N_SIM, dtype=int)
        else:
            sim_kw = np.zeros(N_SIM, dtype=int)
            sim_kc = rng.hypergeometric(ngood=Kc, nbad=N_TEST - Kc, nsample=N_PUB, size=N_SIM)

        sim_ktot = sim_kw + sim_kc

        res_entry = {
            "K_total": Ktot,
            "K_warm": Kw,
            "K_cold": Kc,
            "analytical_moments": {
                "mean_k_pub": round(mean_ktot, 4),
                "var_k_pub": round(var_ktot, 4),
                "sd_k_pub": round(sd_ktot, 4),
                "mean_kw": round(mean_kw, 4),
                "var_kw": round(var_kw, 4),
                "sd_kw": round(sd_kw, 4),
                "mean_kc": round(mean_kc, 4),
                "var_kc": round(var_kc, 4),
                "sd_kc": round(sd_kc, 4),
                "cov_kw_kc": round(cov_kw_kc, 4)
            },
            "simulated_moments": {
                "mean_k_pub": round(float(sim_ktot.mean()), 4),
                "var_k_pub": round(float(sim_ktot.var(ddof=1)), 4),
                "sd_k_pub": round(float(sim_ktot.std(ddof=1)), 4),
                "cov_kw_kc": round(float(np.cov(sim_kw, sim_kc)[0, 1]), 4) if (Kw > 0 and Kc > 0) else 0.0
            },
            "scenarios": {}
        }

        print(f"\n=======================================================")
        print(f"{name}: K={Ktot} (Warm={Kw}, Cold={Kc})")
        print(f"  Theoretical: E[k_pub]={mean_ktot:.2f} | Var={var_ktot:.2f} | SD={sd_ktot:.2f}")
        print(f"  Simulated:   E[k_pub]={sim_ktot.mean():.2f} | Var={sim_ktot.var(ddof=1):.2f} | SD={sim_ktot.std(ddof=1):.2f}")
        if Kw > 0 and Kc > 0:
            print(f"  Joint Cov(kw, kc): Theo={cov_kw_kc:.4f} | Sim={np.cov(sim_kw, sim_kc)[0, 1]:.4f}")

        for sc_name, (pw, pc) in scenarios.items():
            # Exact Analytical Law of Total Variance for Delta:
            # E[Delta] = (2*pw - 1) * mean_kw + (2*pc - 1) * mean_kc
            # Var(Delta) = E[Var(Delta|k)] + Var(E[Delta|k])
            # E[Var(Delta|k)] = 4 * pw*(1 - pw) * mean_kw + 4 * pc*(1 - pc) * mean_kc
            # Var(E[Delta|k]) = (2*pw - 1)^2 * var_kw + (2*pc - 1)^2 * var_kc + 2*(2*pw - 1)*(2*pc - 1)*cov_kw_kc
            mean_delta_theo = (2.0 * pw - 1.0) * mean_kw + (2.0 * pc - 1.0) * mean_kc
            e_var = 4.0 * pw * (1.0 - pw) * mean_kw + 4.0 * pc * (1.0 - pc) * mean_kc
            var_e = ((2.0 * pw - 1.0) ** 2) * var_kw + ((2.0 * pc - 1.0) ** 2) * var_kc + 2.0 * (2.0 * pw - 1.0) * (2.0 * pc - 1.0) * cov_kw_kc
            var_delta_theo = e_var + var_e
            sd_delta_theo = np.sqrt(var_delta_theo)

            # Simulation of Delta:
            hits_w = rng.binomial(sim_kw, pw) if Kw > 0 else np.zeros(N_SIM, dtype=int)
            hits_c = rng.binomial(sim_kc, pc) if Kc > 0 else np.zeros(N_SIM, dtype=int)
            sim_delta = (2 * hits_w - sim_kw) + (2 * hits_c - sim_kc)

            p_ge_50 = float(np.mean(sim_delta >= 50))
            p_pos = float(np.mean(sim_delta > 0))
            q05 = float(np.percentile(sim_delta, 5))
            q50 = float(np.median(sim_delta))
            q95 = float(np.percentile(sim_delta, 95))

            res_entry["scenarios"][sc_name] = {
                "pw": pw,
                "pc": pc,
                "theoretical_mean_delta": round(mean_delta_theo, 2),
                "theoretical_sd_delta": round(sd_delta_theo, 2),
                "simulated_mean_delta": round(float(sim_delta.mean()), 2),
                "simulated_sd_delta": round(float(sim_delta.std(ddof=1)), 2),
                "q05_delta": round(q05, 1),
                "median_delta": round(q50, 1),
                "q95_delta": round(q95, 1),
                "conditional_p_ge_50": round(p_ge_50, 4),
                "conditional_p_positive": round(p_pos, 4)
            }

            print(f"  {sc_name[:32]:32}: Theo E[Delta]={mean_delta_theo:+6.1f}, SD={sd_delta_theo:4.1f} | Sim E={sim_delta.mean():+6.1f}, SD={sim_delta.std(ddof=1):4.1f} | Q05={q05:+5.1f} | Q95={q95:+5.1f} | P(>=+50)={p_ge_50:.1%}")

        results[name] = res_entry

    out_path = 'results/hypergeometric_simulation_without_replacement_report.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved verified report to {out_path}")

if __name__ == '__main__':
    run()
