"""Inspect meta stacking predictions."""
import numpy as np
import pandas as pd

p1 = np.load("results/meta_stacking_ensemble/test_probabilities.npy")
p2 = np.load("results/meta_stacking_ensemble/nested_test_probabilities.npy")

print(f"p1 shape: {p1.shape}, range: [{p1.min():.4f}, {p1.max():.4f}]")
print(f"p2 shape: {p2.shape}, range: [{p2.min():.4f}, {p2.max():.4f}]")

df_evt = pd.read_csv("submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv")
y_evt = (df_evt["condition"] == "used").astype(int).values

y_p1 = (p1 >= 0.50).astype(int)
y_p2 = (p2 >= 0.50).astype(int)

diff1 = (y_evt != y_p1).sum()
diff2 = (y_evt != y_p2).sum()

print(f"Diff vs EVT-132: p1 diff={diff1}, p2 diff={diff2}")
print(f"p1 used count: {y_p1.sum()}, p2 used count: {y_p2.sum()}")
