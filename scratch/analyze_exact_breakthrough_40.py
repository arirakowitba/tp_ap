import pandas as pd

s7_2 = pd.read_csv('submissions/portfolio_20261107_target_09490/submission_20261107_02_calibrated_expansion_top45.csv').set_index('ID')['condition']
s7_4 = pd.read_csv('submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv').set_index('ID')['condition']

diff = (s7_4 != s7_2)
print(f"Total diffs between s7_2 (0.94738) and s7_4 (0.94752): {diff.sum()}")

diff_ids = s7_4[diff].index.tolist()
print(f"IDs that created the +3 hit breakthrough to 0.94752:\n{diff_ids}")

# What directions?
u2n = ((s7_2 == 'used') & (s7_4 == 'new')).sum()
n2u = ((s7_2 == 'new') & (s7_4 == 'used')).sum()
print(f"used -> new: {u2n}")
print(f"new -> used: {n2u}")
