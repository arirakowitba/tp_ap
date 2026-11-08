import pandas as pd, json
from pathlib import Path

p_d1d2 = 'submissions/candidates_20261104/candidate_01_d1d2_plus_upper10.csv'
p_s4_y = 'submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv'
p_test = 'data/ap_2026_q_2/raw/test_data.jsonlines'

with open(p_test, encoding='utf-8') as f:
    test_rows = [json.loads(line) for line in f]

d1 = pd.read_csv(p_d1d2).set_index('ID')['condition']
s4 = pd.read_csv(p_s4_y).set_index('ID')['condition']

diff = s4[s4 != d1]
print(f"Total flips in S4 Yesterday vs D1D2: {len(diff)}")
print(diff.value_counts())

print("\n=== FLIPS IN S4 YESTERDAY (WHICH SCORED 0.94752) ===")
for i in diff.index:
    r = test_rows[i-1]
    p = float(r.get('price', 0) or 0)
    print(f"ID={i:5d}: D1={d1[i]} -> S4={s4[i]} | Price=${p:5.0f} | {r.get('title')[:60]}")
