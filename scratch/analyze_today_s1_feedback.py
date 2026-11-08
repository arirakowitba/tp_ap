import pandas as pd, json
from pathlib import Path

p_s4_yesterday = 'submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv'
p_s1_yesterday = 'submissions/portfolio_20261107_target_09490/submission_20261107_01_purified_probe_top25.csv'
p_s1_today = 'submissions/portfolio_20261108_podium_target_09490/submission_20261108_01_surgical_rank5_capture_top6.csv'
p_test = 'data/ap_2026_q_2/raw/test_data.jsonlines'

with open(p_test, encoding='utf-8') as f:
    test_rows = [json.loads(line) for line in f]

s4_y = pd.read_csv(p_s4_yesterday).set_index('ID')['condition']
s1_y = pd.read_csv(p_s1_yesterday).set_index('ID')['condition']
s1_t = pd.read_csv(p_s1_today).set_index('ID')['condition']

diff_s1t_s4y = s1_t[s1_t != s4_y]
print(f"Flips in S1 Today vs S4 Yesterday: {len(diff_s1t_s4y)}")
for i in diff_s1t_s4y.index:
    r = test_rows[i-1]
    p = float(r.get('price', 0) or 0)
    t = r.get('title', '')
    print(f"ID={i:5d}: S4_Yesterday={s4_y[i]} -> S1_Today={s1_t[i]} | S1_Yesterday={s1_y[i]} | Price=${p:.0f} | {t}")
