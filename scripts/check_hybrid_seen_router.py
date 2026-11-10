"""Calculate potential of Hybrid Seen-Seller Router: EVT-132 on unseen, Meta-Stacker on seen."""
import json
import numpy as np
import pandas as pd

# 1. Load train and test sellers
with open("data/ap_2026_q_2/raw/train_data.jsonlines", encoding="utf-8") as f:
    train_items = [json.loads(line) for line in f]
train_sellers = set(it.get("seller_id") for it in train_items if it.get("seller_id") is not None)

with open("data/ap_2026_q_2/raw/test_data.jsonlines", encoding="utf-8") as f:
    test_items = [json.loads(line) for line in f]
test_sellers = [it.get("seller_id") for it in test_items]

seen_mask = np.array([s in train_sellers for s in test_sellers])
unseen_mask = ~seen_mask

print(f"Total Test Items: {len(test_items)}")
print(f"Seen Seller Items:   {seen_mask.sum()} ({seen_mask.mean()*100:.2f}%)")
print(f"Unseen Seller Items: {unseen_mask.sum()} ({unseen_mask.mean()*100:.2f}%)")

# Load EVT-132
df_evt = pd.read_csv("submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv")
y_evt = df_evt["condition"].values

# Load 12-Model Meta Stacker
p_nested = np.load("results/meta_stacking_ensemble/nested_test_probabilities.npy")
y_nested = np.where(p_nested >= 0.50, "used", "new")

# Disagreements by regime
diff_seen = (y_evt[seen_mask] != y_nested[seen_mask]).sum()
diff_unseen = (y_evt[unseen_mask] != y_nested[unseen_mask]).sum()

print(f"\nDisagreements on Seen Sellers:   {diff_seen} items")
print(f"Disagreements on Unseen Sellers: {diff_unseen} items")

# Hybrid router:
# On unseen sellers: KEEP EVT-132!
# On seen sellers: ONLY take High-Confidence Meta-Stacker predictions!
margin_seen = np.abs(p_nested[seen_mask] - 0.50)
print(f"Seen disagreements with margin > 0.35 (P > 0.85 or P < 0.15): {((y_evt[seen_mask] != y_nested[seen_mask]) & (margin_seen > 0.35)).sum()}")
print(f"Seen disagreements with margin > 0.40 (P > 0.90 or P < 0.10): {((y_evt[seen_mask] != y_nested[seen_mask]) & (margin_seen > 0.40)).sum()}")
