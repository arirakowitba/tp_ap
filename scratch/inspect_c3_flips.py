import json
import pandas as pd

with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    test_rows = [json.loads(line) for line in f]
df_test = pd.DataFrame(test_rows)

s4 = pd.read_csv('submissions/SUBMISSION_4_TODAY_RECOMMENDED.csv')
s1 = pd.read_csv('submissions/SUBMISSION_1_TODAY_RECOMMENDED.csv')
s5 = pd.read_csv('submissions/SUBMISSION_5_TODAY_FINAL.csv')
c3 = pd.read_csv('submissions/new_bases_portfolio_20261111/SUBMISSION_CANDIDATE_3_CONSERVATIVE_OVERLAY_K15.csv')
base_probs = pd.read_csv('results/new_predictive_bases_sprint/test_probabilities_bases.csv')

s1_flips = set(s1.loc[s1['condition'] != s4['condition'], 'ID'])
s5_flips = set(s5.loc[s5['condition'] != s4['condition'], 'ID'])
c3_flips = set(c3.loc[c3['condition'] != s4['condition'], 'ID'])

print(f"Total C3 flips: {len(c3_flips)}")
for cid in sorted(c3_flips):
    row_info = df_test.iloc[cid - 1]
    p_stack = base_probs.loc[base_probs['ID'] == cid, 'p_agent_d_stack'].values[0]
    in_s1 = cid in s1_flips
    in_s5 = cid in s5_flips
    title = row_info['title']
    seller = row_info['seller_id']
    cat = row_info['category_id']
    print(f"ID {cid:5d} | p={p_stack:.4f} | S1={in_s1} | S5={in_s5} | seller={seller} | cat={cat} | title: {title}")
