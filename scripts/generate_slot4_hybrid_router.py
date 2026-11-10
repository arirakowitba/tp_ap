"""Generate Slot 4: Hybrid Seen-Seller Router (C4 on Seen, EVT-132 on Unseen).

Protects against unseen seller shift:
- Unseen Sellers (8,513 items): 100% EVT-132 anchor (0.94752)
- Seen Sellers (21,487 items): C4 Nested Meta-Stacker (96.09% accuracy in CV)
- Diff vs EVT-132: Exactly 749 items (all on seen sellers)
- Class balance: 14,156 used (47.19%), 15,844 new (52.81%)
"""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".")
out_dir = ROOT / "submissions/portfolio_20261109_podium_ladder"
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Load sellers
with open("data/ap_2026_q_2/raw/train_data.jsonlines", encoding="utf-8") as f:
    train_sellers = set(json.loads(line).get("seller_id") for line in f)

with open("data/ap_2026_q_2/raw/test_data.jsonlines", encoding="utf-8") as f:
    test_sellers = [json.loads(line).get("seller_id") for line in f]

seen_mask = np.array([s in train_sellers for s in test_sellers])
assert len(seen_mask) == 30000

# 2. Load EVT-132 and C4
df_evt = pd.read_csv("submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv")
df_c4 = pd.read_csv("results/final_audit_20261109/candidate_c4_nested_meta_stacker_frozen.csv")

# 3. Create Hybrid
df_hybrid = df_evt.copy()
df_hybrid.loc[seen_mask, "condition"] = df_c4.loc[seen_mask, "condition"]

# 4. Verifications
assert len(df_hybrid) == 30000
assert list(df_hybrid.columns) == ["ID", "condition"]
assert set(df_hybrid["condition"].unique()) == {"new", "used"}
assert df_hybrid["ID"].tolist() == list(range(1, 30001))

out_path = out_dir / "submission_slot4_hybrid_seen_router_c4.csv"
df_hybrid.to_csv(out_path, index=False, lineterminator="\n")

data = out_path.read_bytes().replace(b"\r\n", b"\n")
sha = hashlib.sha256(data).hexdigest()

diff = (df_hybrid["condition"] != df_evt["condition"]).sum()
diff_unseen = (df_hybrid.loc[~seen_mask, "condition"] != df_evt.loc[~seen_mask, "condition"]).sum()

print("=" * 80)
print(f"File Generated:         {out_path.name}")
print(f"Path:                   {out_path.resolve()}")
print(f"SHA-256 (LF):           {sha}")
print(f"Total Diffs vs EVT-132: {diff} items (ALL on Seen Sellers)")
print(f"Diffs on Unseen Sellers:{diff_unseen} items (ZERO differences, 100% protected)")
print(f"Total Predicted 'used': {(df_hybrid['condition'] == 'used').sum()} / 30,000")
print(f"Total Predicted 'new':  {(df_hybrid['condition'] == 'new').sum()} / 30,000")
print("Role:                   Hybrid Seen-Seller Router: 96.1% seen accuracy + robust anchor on unseen.")
print("=" * 80)
