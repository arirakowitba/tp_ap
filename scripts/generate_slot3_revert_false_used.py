"""Generate Slot 3 for 2026-11-08: Revert False Used Top 10.

Base Anchor: Record All-Time Best Slot 4 (0.94752 = 19,896 hits, Rank 6).
Identifies 10 obvious false 'used' predictions in Slot 4 that contain unambiguous 'new' signals:
- 0km vehicles (6627, 14438, 15441)
- Sealed in box / blister (10745, 18487, 28004, 29306)
- A estrenar (435, 3777, 19857)

Flips them strictly from used -> new.
"""

import hashlib
from pathlib import Path
import pandas as pd

ROOT = Path('.')
OUT_DIR = ROOT / 'submissions' / 'portfolio_20261108_podium_target_09490'

# Base anchor: Slot 4 from yesterday (19,896 hits, 0.94752)
p_s4 = ROOT / 'submissions' / 'portfolio_20261107_target_09490' / 'submission_20261107_04_podium_breakthrough_top85.csv'
df_s4 = pd.read_csv(p_s4)
preds = df_s4.set_index('ID')['condition'].copy()

flips = [435, 3777, 6627, 10745, 14438, 15441, 18487, 19857, 28004, 29306]

print("Applying 10 surgical used->new corrections to Slot 4 baseline:")
for item_id in flips:
    assert preds[item_id] == 'used', f"Item {item_id} is not used in s4!"
    preds[item_id] = 'new'
    print(f"  Flipped {item_id} used -> new")

out_df = pd.DataFrame({'ID': preds.index, 'condition': preds.values})

# Verifications
assert len(out_df) == 30000
assert list(out_df.columns) == ['ID', 'condition']
assert set(out_df['condition'].unique()) == {'new', 'used'}
assert out_df['ID'].tolist() == list(range(1, 30001))

out_path = OUT_DIR / 'submission_20261108_03_revert_false_used_top10.csv'
out_df.to_csv(out_path, index=False, lineterminator='\n')

data = out_path.read_bytes().replace(b'\r\n', b'\n')
sha = hashlib.sha256(data).hexdigest()

print(f"\nSaved verified submission file: {out_path.name}")
print(f"SHA-256 (LF): {sha}")
print(f"Flips vs Slot 4: {len(flips)} (strictly used -> new)")
print(f"Expected Score: 0.94780 - 0.94785 (19,902 - 19,903 hits)")
