"""Examine candidate submissions from meta-stacking ensemble."""
import numpy as np
import pandas as pd
import hashlib

p_nested = np.load("results/meta_stacking_ensemble/nested_test_probabilities.npy")
p_plain = np.load("results/meta_stacking_ensemble/test_probabilities.npy")

df_evt = pd.read_csv("submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv")
y_evt = (df_evt["condition"] == "used").astype(int).values

for name, probs in [("Nested Meta-Stacker (12 models)", p_nested), ("NNLS Stacker (12 models)", p_plain)]:
    preds = (probs >= 0.50).astype(int)
    diff = (y_evt != preds).sum()
    n_used = preds.sum()
    print(f"\nModel: {name}")
    print(f"  Diff vs EVT-132: {diff} items")
    print(f"  Predicted used:  {n_used} ({n_used/30000*100:.2f}%)")
    print(f"  Confidence distribution:")
    print(f"    P >= 0.90: {(probs >= 0.90).sum()}")
    print(f"    P <= 0.10: {(probs <= 0.10).sum()}")
    print(f"    Uncertain (0.40 <= P <= 0.60): {((probs >= 0.40) & (probs <= 0.60)).sum()}")
