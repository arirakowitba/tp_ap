import pandas as pd
import numpy as np

# Load unified multimodal confidence matrix
matrix = pd.read_csv('results/unified_multimodal_confidence_matrix.csv')
print(f'Total rows in matrix: {len(matrix)}')

# Anchor s7_4 (our 0.94752 peak)
s7_4 = pd.read_csv('submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv').set_index('ID')['condition']

# Let's inspect items where curr_s4 == 'new' and p_multimodal is very high (> 0.80)
# (candidates for new -> used)
cand_n2u = matrix[(matrix['curr_s4'] == 'new') & (matrix['p_multimodal'] > 0.70) & (~matrix['has_bad_text'])].copy()
cand_n2u['abs_gain'] = cand_n2u['bayes_gain']
cand_n2u = cand_n2u.sort_values(by=['bayes_tests', 'p_multimodal'], ascending=[False, False])

print(f'\nTotal new->used candidates with p_multimodal > 0.70: {len(cand_n2u)}')
print('\nTop 15 new->used candidates with highest test count:')
cols = ['ID', 'bayes_tests', 'bayes_gain', 'bayes_map', 'p_multimodal', 'p_seller_used', 'price', 'title']
print(cand_n2u[cols].head(15).to_string())

# Let's inspect items where curr_s4 == 'used' and p_multimodal is very low (< 0.20)
# (candidates for used -> new)
cand_u2n = matrix[(matrix['curr_s4'] == 'used') & (matrix['p_multimodal'] < 0.25) & (~matrix['is_toxic'])].copy()
cand_u2n = cand_u2n.sort_values(by=['bayes_tests', 'p_multimodal'], ascending=[False, True])

print(f'\nTotal used->new candidates with p_multimodal < 0.25: {len(cand_u2n)}')
print('\nTop 15 used->new candidates with highest test count:')
print(cand_u2n[cols].head(15).to_string())
