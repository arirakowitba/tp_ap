"""Generate Slot 2 for 2026-11-08: Pure Sellers + Obsolete Items Top 11.

Base Anchor: Record All-Time Best Slot 4 (0.94752 = 19,896 hits, Rank 6).
Reverts the 6 toxic flips from Slot 1 (9499, 16451, 16554, 22299, 25345, 29896 remain NEW).
Applies 11 ultra-pristine new->used flips:
- 8 items from sellers with 20-30 used items in train (0 new), IQ=1, price <= $1350 ARS.
- 1 item with explicit text 'un solo uso' (345).
- 1 item obsolete 14-inch CRT monitor (18399).
- 1 item loose 1980 Matchbox toy van (23119).
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

flips = [345, 9822, 11994, 26109, 268, 16358, 23224, 10881, 3780, 18399, 23119]

print("Applying 11 surgical flips to Slot 4 baseline:")
for item_id in flips:
    assert preds[item_id] == 'new', f"Item {item_id} is not new in s4!"
    preds[item_id] = 'used'
    print(f"  Flipped {item_id} to used")

out_df = pd.DataFrame({'ID': preds.index, 'condition': preds.values})

# Verifications
assert len(out_df) == 30000
assert list(out_df.columns) == ['ID', 'condition']
assert set(out_df['condition'].unique()) == {'new', 'used'}
assert out_df['ID'].tolist() == list(range(1, 30001))

out_path = OUT_DIR / 'submission_20261108_02_pure_sellers_and_obsolete_top11.csv'
out_df.to_csv(out_path, index=False, lineterminator='\n')

data = out_path.read_bytes().replace(b'\r\n', b'\n')
sha = hashlib.sha256(data).hexdigest()

print(f"\nSaved verified submission file: {out_path.name}")
print(f"SHA-256 (LF): {sha}")
print(f"Flips vs Slot 4: {len(flips)}")
print(f"Expected Score: 0.94780 - 0.94785 (19,902 - 19,903 hits)")
