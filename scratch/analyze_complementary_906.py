import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

# Load Podium and AG-0168 OOF
podium_oof = pd.read_csv("results/ultimate_podium_engine_23h/oof_predictions_podium_engine.csv")
ag_oof = pd.read_csv("results/new_predictive_bases_sprint/paired_oof_predictions.csv")

y_true = podium_oof["y_true"].values
p_pod = podium_oof["oof_blend"].values
p_ag = ag_oof["p_agent_d_stack"].values

pred_pod = (p_pod >= 0.5).astype(int)
pred_ag = (p_ag >= 0.5).astype(int)

# Identify the 906 items
mask_906 = (pred_pod != y_true) & (pred_ag == y_true)
print(f"Total Complementary Items: {mask_906.sum()}")

# What is the distribution of probabilities on these items?
df_906 = pd.DataFrame({
    "y_true": y_true[mask_906],
    "p_pod": p_pod[mask_906],
    "p_ag": p_ag[mask_906],
    "margin_ag": np.abs(p_ag[mask_906] - 0.50),
    "margin_pod": np.abs(p_pod[mask_906] - 0.50)
})

print("AG-0168 confidence distribution on the 906 wins:")
print(df_906["margin_ag"].describe())

# What happens if we test a convex blend of Podium and AG-0168?
for w_ag in [0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20]:
    p_blend = (1 - w_ag) * p_pod + w_ag * p_ag
    acc_b = accuracy_score(y_true, (p_blend >= 0.5).astype(int))
    hits_b = int(round(acc_b * 70000))
    ll_b = log_loss(y_true, p_blend)
    pred_b = (p_blend >= 0.5).astype(int)
    fixes = int(((pred_pod != y_true) & (pred_b == y_true)).sum())
    breaks = int(((pred_pod == y_true) & (pred_b != y_true)).sum())
    print(f"w_ag={w_ag:0.2f} -> Hits={hits_b} ({hits_b - 65961:+d}), LL={ll_b:.5f}, Fixes={fixes}, Breaks={breaks}")
