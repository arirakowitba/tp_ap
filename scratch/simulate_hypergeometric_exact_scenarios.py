import json
import numpy as np
import pandas as pd

def run():
    print("=== HYPERGEOMETRIC SIMULATION WITHOUT REPLACEMENT (N=30000, n=20998) ===")
    N_TEST = 30000
    N_PUB = 20998
    p_pub = N_PUB / N_TEST # 0.6999333333333333

    rng = np.random.default_rng(20261114)
    N_SIM = 50000

    # Disjoint blocks:
    # B1 (A warm): 303 flips
    # B2 (B\A cold): 76 flips
    # B3 (C\A warm): 180 flips
    # Clean warm (AG-0223): 161 flips

    blocks = {
        "Block_1_A_Warm": {"K": 303, "type": "warm"},
        "Block_2_B_minus_A_Cold": {"K": 76, "type": "cold"},
        "Block_3_C_minus_A_Warm": {"K": 180, "type": "warm"},
        "Candidate_A": {"K": 303, "warm_K": 303, "cold_K": 0},
        "Candidate_B": {"K": 379, "warm_K": 303, "cold_K": 76},
        "Candidate_C": {"K": 483, "warm_K": 483, "cold_K": 0},
        "Candidate_Clean_AG0223": {"K": 161, "warm_K": 161, "cold_K": 0},
    }

    # Scenarios for flip precision: (p_warm, p_cold)
    scenarios = {
        "S1_Optimista (Warm=75%, Cold=55%)": (0.75, 0.55),
        "S2_Central (Warm=65%, Cold=50%)": (0.65, 0.50),
        "S3_Conservador (Warm=58%, Cold=46%)": (0.58, 0.46),
        "S4_Adverso (Warm=50%, Cold=42%)": (0.50, 0.42),
        "S5_Prequential_Retroceso (EVT-155 like: Warm=45%, Cold=35%)": (0.45, 0.35)
    }

    results = {}

    for cand_name, info in blocks.items():
        results[cand_name] = {
            "K_test": info["K"],
            "E_k_pub": round(info["K"] * p_pub, 1),
            "std_k_pub": round(math_std := np.sqrt(info["K"] * p_pub * (1 - p_pub) * (N_TEST - N_PUB) / (N_TEST - 1)), 2),
            "scenarios": {}
        }
        
        warm_k = info.get("warm_K", info["K"] if info.get("type") == "warm" else 0)
        cold_k = info.get("cold_K", info["K"] if info.get("type") == "cold" else 0)

        # Draw hypergeometric public flips for this candidate
        k_pub_warm = rng.hypergeometric(ngood=warm_k, nbad=N_TEST - warm_k, nsample=N_PUB, size=N_SIM) if warm_k > 0 else np.zeros(N_SIM, dtype=int)
        k_pub_cold = rng.hypergeometric(ngood=cold_k, nbad=N_TEST - cold_k, nsample=N_PUB, size=N_SIM) if cold_k > 0 else np.zeros(N_SIM, dtype=int)

        for sc_name, (pw, pc) in scenarios.items():
            # In each simulation draw, conditional on k_pub_warm and k_pub_cold,
            # hits = Binomial(k_pub_warm, pw) + Binomial(k_pub_cold, pc)
            hits_w = rng.binomial(k_pub_warm, pw)
            hits_c = rng.binomial(k_pub_cold, pc)
            total_hits = hits_w + hits_c
            total_flips = k_pub_warm + k_pub_cold
            delta_pub = 2 * total_hits - total_flips

            p_plus50 = float(np.mean(delta_pub >= 50))
            p_positive = float(np.mean(delta_pub > 0))

            results[cand_name]["scenarios"][sc_name] = {
                "p_warm": pw,
                "p_cold": pc,
                "mean_delta_pub": round(float(np.mean(delta_pub)), 1),
                "std_delta_pub": round(float(np.std(delta_pub)), 1),
                "q05_delta_pub": round(float(np.percentile(delta_pub, 5)), 1),
                "median_delta_pub": round(float(np.median(delta_pub)), 1),
                "q95_delta_pub": round(float(np.percentile(delta_pub, 95)), 1),
                "prob_ge_plus50_conditional": round(p_plus50, 4),
                "prob_positive_gain_conditional": round(p_positive, 4),
            }

    print("\n--- RESULTS OF 50,000 HYPERGEOMETRIC DRAWS WITHOUT REPLACEMENT ---")
    for cand_name, res in results.items():
        print(f"\n=======================================================")
        print(f"{cand_name} (K_test={res['K_test']}, E[k_pub]={res['E_k_pub']}, std={res['std_k_pub']}):")
        for sc_name, sc_data in res["scenarios"].items():
            print(f"  {sc_name:40}: E[Delta]={sc_data['mean_delta_pub']:+5.1f} | Q05={sc_data['q05_delta_pub']:+5.1f} | Q95={sc_data['q95_delta_pub']:+5.1f} | P(>=+50)={sc_data['prob_ge_plus50_conditional']:.1%} | P(>0)={sc_data['prob_positive_gain_conditional']:.1%}")

    with open('results/hypergeometric_simulation_without_replacement_report.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved report to results/hypergeometric_simulation_without_replacement_report.json")

if __name__ == '__main__':
    run()
