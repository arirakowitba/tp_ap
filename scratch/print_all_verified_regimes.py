import pandas as pd
import numpy as np

reg_full70k = np.load('scratch/reg_full70k.npy')
reg_dev = np.load('scratch/reg_dev.npy')

t30 = pd.read_csv('submissions/claude_20261114/claude_blendfix_posterior_top30_sobre_S4.csv')
u36 = pd.read_csv('submissions/claude_20261114/claude_T30_mas_U36_estricto.csv')
u37 = pd.read_csv('results/CANDIDATE_T30_PLUS_U37.csv')
f86 = pd.read_csv('submissions/claude_20261114/claude_T30_mas_F86_estricto.csv')
clean = pd.read_csv('results/CANDIDATE_T30_PLUS_WARM_CLEAN.csv')
cand_a = pd.read_csv('results/CANDIDATE_T30_PLUS_WARM_A.csv')

candidates = {
    'U36 (Claude)': (u36['condition'] != t30['condition']).values,
    'U37 (AG)': (u37['condition'] != t30['condition']).values,
    'F86 (Claude)': (f86['condition'] != t30['condition']).values,
    'Clean (AG 160)': (clean['condition'] != t30['condition']).values,
    'A (AG 301)': (cand_a['condition'] != t30['condition']).values,
}

regimes_list = ['cold', 'bracketed', 'ties', 'before', 'after']

print("=== REGIME COUNTS UNDER FULL 70K FIT ===")
summary_70k = []
for name, mask in candidates.items():
    row = {'Candidate': name, 'Total': mask.sum()}
    for r in regimes_list:
        row[r] = int((mask & (reg_full70k == r)).sum())
    summary_70k.append(row)
print(pd.DataFrame(summary_70k).to_string(index=False))

print("\n=== REGIME COUNTS UNDER DEV 55.9K FIT ===")
summary_dev = []
for name, mask in candidates.items():
    row = {'Candidate': name, 'Total': mask.sum()}
    for r in regimes_list:
        row[r] = int((mask & (reg_dev == r)).sum())
    summary_dev.append(row)
print(pd.DataFrame(summary_dev).to_string(index=False))
