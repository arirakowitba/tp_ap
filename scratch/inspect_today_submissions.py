import pandas as pd

s7_4 = pd.read_csv('submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv')
s8_1 = pd.read_csv('submissions/portfolio_20261108_podium_target_09490/submission_20261108_01_surgical_rank5_capture_top6.csv')
s8_2 = pd.read_csv('submissions/portfolio_20261108_podium_target_09490/submission_20261108_02_pure_sellers_and_obsolete_top11.csv')
s8_3 = pd.read_csv('submissions/portfolio_20261108_podium_target_09490/submission_20261108_03_revert_false_used_top10.csv')
s8_4 = pd.read_csv('submissions/portfolio_20261108_podium_target_09490/submission_20261108_04_vintage_parts_and_antiques_top10.csv')
s8_5 = pd.read_csv('submissions/portfolio_20261108_podium_target_09490/submission_20261108_05_balanced_consensus_top20.csv')
matrix = pd.read_csv('results/unified_multimodal_confidence_matrix.csv').set_index('ID')

def print_sub(name, sub, delta):
    diff = sub['condition'] != s7_4['condition']
    diff_ids = sub.loc[diff, 'ID'].tolist()
    print(f'=== {name} (N={len(diff_ids)}, actual delta={delta}) ===')
    for idx in diff_ids:
        row = matrix.loc[idx]
        from_c = s7_4.loc[s7_4['ID']==idx, 'condition'].values[0]
        to_c = sub.loc[sub['ID']==idx, 'condition'].values[0]
        tests = row['bayes_tests'] if pd.notnull(row['bayes_tests']) else 0
        b_map = row['bayes_map'] if pd.notnull(row['bayes_map']) else 0
        p_m = row['p_multimodal'] if pd.notnull(row['p_multimodal']) else 0
        print(f'ID {idx:5d} | {from_c}->{to_c} | tests={tests:2.0f} | p_multi={p_m:.3f} | map={b_map:+.3f} | title={str(row["title"])[:45]}')

print_sub('s8_1 (score 0.94738)', s8_1, -3)
print_sub('s8_2 (score 0.94733)', s8_2, -4)
print_sub('s8_3 (score 0.94742)', s8_3, -2)
print_sub('s8_4 (score 0.94752)', s8_4, 0)
print_sub('s8_5 (score 0.94728)', s8_5, -5)
