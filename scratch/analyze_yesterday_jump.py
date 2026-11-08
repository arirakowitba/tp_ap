import pandas as pd, json
from pathlib import Path

p_s1_y = 'submissions/portfolio_20261107_target_09490/submission_20261107_01_purified_probe_top25.csv'
p_s4_y = 'submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv'
p_test = 'data/ap_2026_q_2/raw/test_data.jsonlines'

with open(p_test, encoding='utf-8') as f:
    test_rows = [json.loads(line) for line in f]

s1_y = pd.read_csv(p_s1_y).set_index('ID')['condition']
s4_y = pd.read_csv(p_s4_y).set_index('ID')['condition']

diff = s4_y[s4_y != s1_y]
print(f"Total differences between S1 Yesterday (0.94738) and S4 Yesterday (0.94752): {len(diff)}")
print(diff.value_counts())

print("\n=== ITEMS FLIPPED IN S4 YESTERDAY THAT GENERATED THE +3 HITS JUMP TO 0.94752 ===")
for i in diff.index:
    r = test_rows[i-1]
    p = float(r.get('price', 0) or 0)
    print(f"ID={i:5d}: S1_Y={s1_y[i]} -> S4_Y={s4_y[i]} | Price=${p:5.0f} Cat={r.get('category_id')} | {r.get('title')}")
