"""Cross-reference C10 flips with Bayesian Fusion to find unanimous consensus."""

import pandas as pd
import numpy as np

evt = pd.read_csv("submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv")
c10 = pd.read_csv("results/dual_regime_20261109/candidate_c10_enriched_super_ensemble.csv")
fusion = pd.read_csv("results/cross_referenced_bayesian_fusion/joint_information_fusion_all30k.csv")

y_evt = (evt["condition"] == "used").astype(int).values
y_c10 = (c10["condition"] == "used").astype(int).values

flips_c10_n2u = (y_evt == 0) & (y_c10 == 1)
print(f"Total C10 new->used flips vs EVT-132: {np.sum(flips_c10_n2u)}")

fusion["is_candidate"] = fusion["is_safe"] & (fusion["joint_value"] > 0)
flips_both = flips_c10_n2u & fusion["is_candidate"].values
print(f"Total UNANIMOUS flips (C10 + Safe Bayesian Fusion): {np.sum(flips_both)}")

df_both = fusion[flips_both].sort_values(by="joint_value", ascending=False).copy()
print("\nTop 20 Unanimous Consensus Flips:")
for idx, (_, r) in enumerate(df_both.head(20).iterrows(), 1):
    print(f"  #{idx:2d} | ID {r['item_id']:5d} | JV: {r['joint_value']:.4f} | P_mod: {r['p_model_mean']:.3f} | {r['title'][:60]}")
