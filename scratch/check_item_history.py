import json
from pathlib import Path
import pandas as pd

ROOT = Path('.')
p_ledger = ROOT / 'results' / 'forensic_ledger_20261104' / 'submission_ledger_master_85vec.json'

with open(p_ledger, encoding='utf-8') as f:
    ledger = json.load(f)

items = [9499, 16451, 16554, 22299, 25345, 29896]

print("=== CHECKING INTERVENTIONS ACROSS HISTORICAL SUBMISSIONS ===")
for item_id in items:
    print(f"\n--- ITEM {item_id} ---")
    subs_containing = []
    for e in ledger['entries']:
        f_ids = e.get('flipped_ids') or e.get('flipped_ids_vs_d1d2') or []
        if item_id in f_ids:
            subs_containing.append((e.get('name') or e.get('key') or e.get('submission_file'), e.get('score'), e.get('delta_vs_d1d2')))
    print(f"Total historical submissions containing {item_id}: {len(subs_containing)}")
    for name, score, delta in subs_containing[:10]:
        print(f"  {name} | score={score} | delta_vs_d1d2={delta}")
