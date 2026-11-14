import pandas as pd

t30 = pd.read_csv('submissions/claude_20261114/claude_blendfix_posterior_top30_sobre_S4.csv')
t30_u58 = pd.read_csv('submissions/claude_20261114/claude_T30_mas_U58_blendfix_noprobado.csv')
t30_u58_f86 = pd.read_csv('submissions/claude_20261114/claude_T30_mas_U58_mas_F86_bfix_noprobado.csv')
u37 = pd.read_csv('results/CANDIDATE_T30_PLUS_U37.csv')

flips_u58 = set(t30[t30['condition'] != t30_u58['condition']]['ID'])
flips_f86 = set(t30_u58[t30_u58['condition'] != t30_u58_f86['condition']]['ID'])
flips_u37 = set(t30[t30['condition'] != u37['condition']]['ID'])

print(f"Overlap U58 and F86: {len(flips_u58.intersection(flips_f86))}")
print(f"Overlap U37 and F86: {len(flips_u37.intersection(flips_f86))}")
print(f"Overlap U37 and Top30: {len(flips_u37.intersection(set(t30[t30['condition'] != pd.read_csv('submissions/SUBMISSION_4_TODAY_RECOMMENDED.csv')['condition']]['ID'])))}")
