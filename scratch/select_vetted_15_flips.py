import pandas as pd
import json

matrix = pd.read_csv('results/unified_multimodal_confidence_matrix.csv').set_index('ID')
s7_4 = pd.read_csv('submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv').set_index('ID')['condition']
d1d2 = pd.read_csv('submissions/candidate_d1_d2_combined_on48.csv').set_index('ID')['condition']

# Ingest test data for full details
with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    test_rows = {i+1: json.loads(line) for i, line in enumerate(f)}

# Let's inspect potential flips vs s7_4
# Direction A: used -> new (corrections of early false-used heuristic)
cands_u2n = []
for idx, row in matrix.iterrows():
    if row['curr_s4'] == 'used':
        # Condition in s7_4 is used
        p_m = row['p_multimodal']
        tests = row['bayes_tests']
        gain = row['bayes_gain']
        is_tox = row['is_toxic']
        if p_m < 0.25 and not is_tox and tests >= 5 and gain > 0.30:
            cands_u2n.append((idx, 'used->new', tests, gain, p_m, row['p_seller_used'], row['price'], row['title']))

df_u2n = pd.DataFrame(cands_u2n, columns=['ID', 'direction', 'tests', 'bayes_gain', 'p_multi', 'p_seller_used', 'price', 'title'])
df_u2n = df_u2n.sort_values(by=['bayes_gain', 'tests'], ascending=[False, False])

# Direction B: new -> used (physical secondhand/vintage items currently marked new)
cands_n2u = []
for idx, row in matrix.iterrows():
    if row['curr_s4'] == 'new':
        p_m = row['p_multimodal']
        tests = row['bayes_tests']
        gain = row['bayes_gain']
        has_bad = row['has_bad_text']
        if p_m > 0.65 and not has_bad and tests >= 5 and gain > 0.10:
            cands_n2u.append((idx, 'new->used', tests, gain, p_m, row['p_seller_used'], row['price'], row['title']))

df_n2u = pd.DataFrame(cands_n2u, columns=['ID', 'direction', 'tests', 'bayes_gain', 'p_multi', 'p_seller_used', 'price', 'title'])
df_n2u = df_n2u.sort_values(by=['bayes_gain', 'tests'], ascending=[False, False])

print("=== CANDIDATES: USED -> NEW ===")
print(df_u2n.to_string())

print("\n=== CANDIDATES: NEW -> USED ===")
print(df_n2u.to_string())
