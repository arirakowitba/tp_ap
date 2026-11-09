"""Analyze unanimous consensus candidates by informational cluster."""

import pandas as pd
import numpy as np
import json
from pathlib import Path

DATA_DIR = Path("data/ap_2026_q_2/raw")
df_test = pd.read_json(DATA_DIR / "test_data.jsonlines", lines=True)

evt = pd.read_csv("submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv")
c10 = pd.read_csv("results/dual_regime_20261109/candidate_c10_enriched_super_ensemble.csv")
fusion = pd.read_csv("results/cross_referenced_bayesian_fusion/joint_information_fusion_all30k.csv")

y_evt = (evt["condition"] == "used").astype(int).values
y_c10 = (c10["condition"] == "used").astype(int).values

c10_n2u = (y_evt == 0) & (y_c10 == 1)
fusion["is_candidate"] = fusion["is_safe"] & (fusion["joint_value"] > 0)
unanimous_mask = c10_n2u & fusion["is_candidate"].values

df_pool = fusion[unanimous_mask].copy()
print(f"Total Unanimous Candidates Pool: {len(df_pool)}")

# Classify into orthogonal informational clusters:
# Cluster A: Pure Semantic & Vintage/Collectible (Books, Vinyl, Antiques, Coins, Obvious used keywords)
# Category prefixes: MLA3025 (Libros), MLA1182 (Instrumentos/Vinilos), MLA1403 (Antigüedades), MLA1430 (Monedas)
cat_vintage = {"MLA3025", "MLA1182", "MLA1403", "MLA1430", "MLA1144"}
is_vintage_cat = df_pool["category_id"].isin(cat_vintage)
has_condition_kw = df_pool["has_kw"]

df_pool["cluster_A_semantic"] = has_condition_kw | is_vintage_cat

# Cluster B: Tabular Model Consensus (High confidence tree models: p_model_confidence > 0.85 & std < 0.05)
df_pool["cluster_B_trees"] = (df_pool["p_model_mean"] >= 0.85) & (df_pool["p_model_std"] <= 0.05) & ~df_pool["cluster_A_semantic"]

# Cluster C: Structured Consumer Goods (Apparel, Electronics, Home items with high joint value)
df_pool["cluster_C_consumer"] = ~df_pool["cluster_A_semantic"] & ~df_pool["cluster_B_trees"]

print(f"Cluster A (Semantic & Vintage/Collectibles): {df_pool['cluster_A_semantic'].sum()}")
print(f"Cluster B (High-Confidence Tabular Trees):   {df_pool['cluster_B_trees'].sum()}")
print(f"Cluster C (Structured Consumer Goods):       {df_pool['cluster_C_consumer'].sum()}")

print("\nSample Cluster A:")
for _, r in df_pool[df_pool["cluster_A_semantic"]].head(5).iterrows():
    print(f"  ID {r['item_id']:5d} | Cat: {r['category_id']} | JV: {r['joint_value']:.4f} | {r['title'][:60]}")

print("\nSample Cluster B:")
for _, r in df_pool[df_pool["cluster_B_trees"]].head(5).iterrows():
    print(f"  ID {r['item_id']:5d} | Cat: {r['category_id']} | P_mod: {r['p_model_mean']:.3f} | {r['title'][:60]}")

print("\nSample Cluster C:")
for _, r in df_pool[df_pool["cluster_C_consumer"]].head(5).iterrows():
    print(f"  ID {r['item_id']:5d} | Cat: {r['category_id']} | JV: {r['joint_value']:.4f} | {r['title'][:60]}")
