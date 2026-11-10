"""Inspect 636 diffs between EVT-132 and p2."""
import numpy as np
import pandas as pd

p2 = np.load("results/meta_stacking_ensemble/nested_test_probabilities.npy")
df_evt = pd.read_csv("submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv")
y_evt = (df_evt["condition"] == "used").astype(int).values
y_p2 = (p2 >= 0.50).astype(int)

diff_mask = (y_evt != y_p2)
print(f"Total diffs: {diff_mask.sum()}")

evt_new_p2_used = (y_evt == 0) & (y_p2 == 1)
evt_used_p2_new = (y_evt == 1) & (y_p2 == 0)

print(f"EVT=new, p2=used: {evt_new_p2_used.sum()}")
print(f"EVT=used, p2=new: {evt_used_p2_new.sum()}")

# Load titles
import json
with open("data/ap_2026_q_2/raw/test_data.jsonlines", encoding="utf-8") as f:
    items = [json.loads(line) for line in f]

# Confidence of p2 on these items
confidence = np.abs(p2 - 0.50)

print("\n--- Top 15 Most Confident P2 predictions where EVT said USED and P2 says NEW ---")
used_to_new_indices = np.where(evt_used_p2_new)[0]
top_u2n = sorted(used_to_new_indices, key=lambda idx: p2[idx])[:15]
for idx in top_u2n:
    it = items[idx]
    print(f"ID {idx+1:5d} | P2_prob: {p2[idx]:.4f} | ${it.get('price', 0):7.1f} | {it.get('title')[:55]}")

print("\n--- Top 15 Most Confident P2 predictions where EVT said NEW and P2 says USED ---")
new_to_used_indices = np.where(evt_new_p2_used)[0]
top_n2u = sorted(new_to_used_indices, key=lambda idx: -p2[idx])[:15]
for idx in top_n2u:
    it = items[idx]
    print(f"ID {idx+1:5d} | P2_prob: {p2[idx]:.4f} | ${it.get('price', 0):7.1f} | {it.get('title')[:55]}")
