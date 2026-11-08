import pandas as pd

m = pd.read_csv('results/unified_multimodal_confidence_matrix.csv')
d1d2 = pd.read_csv('submissions/candidate_d1_d2_combined_on48.csv').set_index('ID')['condition']
s7_4 = pd.read_csv('submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv').set_index('ID')['condition']

# Items where s7_4 == d1d2 (unflipped in s4)
unflipped_in_s4 = [i for i in m['ID'] if s7_4.loc[i] == d1d2.loc[i]]
print(f'Total unflipped in s4: {len(unflipped_in_s4)}')

sub = m[m['ID'].isin(unflipped_in_s4)].copy()

# Sort by bayes_gain descending
top_gains = sub[(sub['bayes_tests'] >= 2) & (sub['bayes_gain'] > 0.35) & (~sub['is_toxic']) & (~sub['has_bad_text'])].copy()
top_gains = top_gains.sort_values(by=['bayes_tests', 'bayes_gain'], ascending=[False, False])

print(f'\nCandidates with tests >= 2 and bayes_gain > 0.35 (N={len(top_gains)}):')
cols = ['ID', 'curr_s4', 'bayes_tests', 'bayes_gain', 'p_multimodal', 'p_seller_used', 'price', 'title']
print(top_gains[cols].to_string())
