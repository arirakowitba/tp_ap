"""Inspect disagreement rows between baseline podium and EVT-143."""

import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def main():
    p_evt = ROOT / "submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv"
    p_pod = ROOT / "results/ultimate_podium_engine_23h/test_probabilities_podium_engine.csv"
    p_test_json = ROOT / "data/ap_2026_q_2/raw/test_data.jsonlines"

    df_evt = pd.read_csv(p_evt)
    df_pod = pd.read_csv(p_pod)

    with open(p_test_json, "r", encoding="utf-8") as f:
        titles = [json.loads(line).get("title") or "" for line in f]

    p_base = df_pod["p_optimal_blend"].values
    pred_base = np.where(p_base >= 0.5, "used", "new")
    y_143 = df_evt["condition"].values

    diff_mask = (pred_base != y_143)
    u2n_idx = np.where(diff_mask & (pred_base == "used") & (y_143 == "new"))[0]
    n2u_idx = np.where(diff_mask & (pred_base == "new") & (y_143 == "used"))[0]

    print(f"Total Disagreements: {len(u2n_idx) + len(n2u_idx)}")
    print(f"Base=used, EVT=new: {len(u2n_idx)}")
    print(f"Base=new, EVT=used: {len(n2u_idx)}")

    print("\nSample of Base=used, EVT=new (first 10):")
    for idx in u2n_idx[:10]:
        print(f"  ID {idx+1:5d} | p_base={p_base[idx]:.4f} | Title: {titles[idx][:60]}")

    print("\nSample of Base=new, EVT=used (first 10):")
    for idx in n2u_idx[:10]:
        print(f"  ID {idx+1:5d} | p_base={p_base[idx]:.4f} | Title: {titles[idx][:60]}")

    # Check probabilities distribution of disagreements
    print(f"\nProbability summary for Base=used, EVT=new:")
    print(f"  mean={np.mean(p_base[u2n_idx]):.4f}, median={np.median(p_base[u2n_idx]):.4f}, min={np.min(p_base[u2n_idx]):.4f}, max={np.max(p_base[u2n_idx]):.4f}")
    print(f"  Fraction with p_base < 0.65: {np.mean(p_base[u2n_idx] < 0.65)*100:.1f}%")

    print(f"\nProbability summary for Base=new, EVT=used:")
    print(f"  mean={np.mean(p_base[n2u_idx]):.4f}, median={np.median(p_base[n2u_idx]):.4f}, min={np.min(p_base[n2u_idx]):.4f}, max={np.max(p_base[n2u_idx]):.4f}")
    print(f"  Fraction with p_base > 0.35: {np.mean(p_base[n2u_idx] > 0.35)*100:.1f}%")

if __name__ == "__main__":
    main()
