"""Check C4 diffs by seen/unseen seller regime."""
import json
import numpy as np
import pandas as pd

# Load train and test sellers
with open("data/ap_2026_q_2/raw/train_data.jsonlines", encoding="utf-8") as f:
    train_items = [json.loads(line) for line in f]
train_sellers = set(it.get("seller_id") for it in train_items if it.get("seller_id") is not None)

with open("data/ap_2026_q_2/raw/test_data.jsonlines", encoding="utf-8") as f:
    test_items = [json.loads(line) for line in f]
test_sellers = [it.get("seller_id") for it in test_items]

seen_mask = np.array([s in train_sellers for s in test_sellers])
unseen_mask = ~seen_mask

# Load EVT-132 and C4
df_evt = pd.read_csv("submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv")
df_c4 = pd.read_csv("results/final_audit_20261109/candidate_c4_nested_meta_stacker_frozen.csv")

diff = (df_evt["condition"] != df_c4["condition"])
print(f"Total diffs EVT vs C4: {diff.sum()}")
print(f"Seen diffs:   {(diff & seen_mask).sum()}")
print(f"Unseen diffs: {(diff & unseen_mask).sum()}")
