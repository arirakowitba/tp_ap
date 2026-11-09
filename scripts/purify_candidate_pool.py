"""Purify candidate pool with strict domain guardrails."""

import pandas as pd
import numpy as np
import re

DATA_DIR = Path("data/ap_2026_q_2/raw") if "Path" in locals() else None
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
df_pool["test_title"] = df_test.iloc[df_pool["item_id"].values - 1]["title"].values
df_pool["title_lower"] = df_pool["test_title"].fillna("").astype(str).str.lower()

# Toxic exclusion rules:
# 1. Food / Alimentos y Bebidas (MLA1403 is Antigüedades, but MLA1429 is Alimentos y Bebidas)
is_food = df_pool["category_id"].str.startswith("MLA1429") | df_pool["title_lower"].str.contains(r"\bpor kilo\b|\bcongelad|\balimento|\bcomida|\bgolosina|\bcandy\b")

# 2. "A estrenar" / "Sin uso" / "Caja cerrada" (Potential genuine new)
is_unworn_new = df_pool["title_lower"].str.contains(r"\ba estrenar\b|\bsin uso\b|\bcaja cerrada\b|\bnuevo sellado\b|\b100% nuevo\b")

# 3. Automotive spares (High false positive rate in new parts)
is_auto_parts = df_pool["title_lower"].str.contains(r"\bparagolpe\b|\boptica\b|\bfaro\b|\bembrague\b|\bpastillas de freno\b")

# 4. Filter
toxic_mask = is_food | is_unworn_new | is_auto_parts
print(f"Total initial pool: {len(df_pool)}")
print(f"Filtered out toxic / ambiguous items: {toxic_mask.sum()}")
print(df_pool[toxic_mask][["item_id", "category_id", "test_title"]])

df_purified = df_pool[~toxic_mask].sort_values(by=["joint_value", "p_model_mean"], ascending=[False, False]).copy()
print(f"\nRemaining Purified Unanimous Pool: {len(df_purified)}")

print("\nTop 25 Purified Candidates for Slot 1:")
for idx, (_, r) in enumerate(df_purified.head(25).iterrows(), 1):
    print(f"  #{idx:2d} | ID {r['item_id']:5d} | JV: {r['joint_value']:.4f} | P_mod: {r['p_model_mean']:.3f} | {r['test_title'][:65]}")
