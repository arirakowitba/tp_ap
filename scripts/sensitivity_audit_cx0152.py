"""Compute Detailed Sensitivity Ranges for AG-0148 Report (CX-0152 Compliance)."""

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    report_path = ROOT / "results/nested_selector_cx0152/nested_selector_cx0152_report.json"
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    grid = data["grid_evaluations"]
    print("=" * 80)
    print("SENSITIVITY & QUANTILE SUMMARY FOR NESTED SELECTOR")
    print("=" * 80)

    for k_str, res in grid.items():
        k = int(k_str)
        exp_k_pub = res["expected_public_flips"]
        s_ci = res["seller_cluster_ci95_public_gain"]
        c_ci = res["category_cluster_ci95_public_gain"]
        q05 = res["conservative_q05_public_gain"]
        p_succ = res["prob_reach_095000_conservative"]

        med_gain = (2 * res["precision_oof"] - 1) * exp_k_pub
        score_med = 0.94761 + med_gain / 20998.0
        score_q05 = 0.94761 + q05 / 20998.0
        score_req51 = 0.94761 + 51.0 / 20998.0
        score_req53 = 0.94761 + 53.0 / 21000.0

        print(f"\n[K_test = {k:4d}] (Expected Public Flips: {exp_k_pub})")
        print(f"  Rate-matched OOF Precision:        {res['precision_oof']*100:.2f}% (Fixes={res['total_fixes']}, Breaks={res['total_breaks']})")
        print(f"  Fold Gains:                        {res['fold_net_gains']}")
        print(f"  Seller Cluster 95% CI (Gain):      [{s_ci[0]:+.1f}, {s_ci[1]:+.1f}]")
        print(f"  Category Cluster 95% CI (Gain):    [{c_ci[0]:+.1f}, {c_ci[1]:+.1f}]")
        print(f"  Conservative Q05 (Gain):           {q05:+.1f} hits")
        print(f"  Expected Public Net Gain:          {med_gain:+.1f} hits")
        print(f"  Projected Public Score (Median):   {score_med:.5f}")
        print(f"  Projected Public Score (Q05):      {score_q05:.5f}")
        print(f"  P(Public Gain >= +51):             {p_succ*100:.2f}%")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
