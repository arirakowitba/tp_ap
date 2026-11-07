import pandas as pd, json
from pathlib import Path

p_s4 = 'submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv'
p_s5 = 'submissions/portfolio_20261107_target_09490/submission_20261107_05_podium_climax_top105.csv'
p_test = 'data/ap_2026_q_2/raw/test_data.jsonlines'

s4 = pd.read_csv(p_s4).set_index('ID')['condition']
s5 = pd.read_csv(p_s5).set_index('ID')['condition']

with open(p_test, encoding='utf-8') as f:
    test_rows = [json.loads(line) for line in f]

diff_s5_s4 = s5[s5 != s4]
print(f"Flips in Slot 5 vs Slot 4: {len(diff_s5_s4)}")
print(diff_s5_s4.value_counts())

diff_items = diff_s5_s4.index
print("\n=== THE 20 ITEMS FLIPPED IN SLOT 5 VS SLOT 4 (WHICH CAUSED -3 HITS) ===")
for item_id in diff_items:
    r = test_rows[item_id - 1]
    print(f"ID={item_id:5d} Price=${float(r.get('price', 0) or 0):6.0f} Cat={r.get('category_id')} | {r.get('title')}")
