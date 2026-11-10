"""Inspect disagreements between EVT-132 and C4."""
import pandas as pd
import numpy as np

# Load EVT-132
df_evt = pd.read_csv("submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv")

# Load C4
df_c4 = pd.read_csv("results/final_audit_20261109/candidate_c4_nested_meta_stacker_frozen.csv")

diff_mask = (df_evt["condition"] != df_c4["condition"])
print(f"Total Disagreements between EVT-132 and C4: {diff_mask.sum()}")

# Check direction
evt_new_c4_used = (df_evt["condition"] == "new") & (df_c4["condition"] == "used")
evt_used_c4_new = (df_evt["condition"] == "used") & (df_c4["condition"] == "new")

print(f"EVT=new & C4=used: {evt_new_c4_used.sum()}")
print(f"EVT=used & C4=new: {evt_used_c4_new.sum()}")

# Check with 16-model ensemble probabilities
df_audit = pd.read_csv("results/top_candidates_16_model_ensemble.csv")
p_dict = df_audit.set_index("ID")["mean_p"].to_dict()

# Check test probabilities from intensive training if available
npz_path = "results/intensive_training_20261109/intensive_specialist_test_predictions.npz"
import os
if os.path.exists(npz_path):
    data = np.load(npz_path)
    mean_p = data["mean_p"]
    print("Loaded 21-model predictions from intensive training")
