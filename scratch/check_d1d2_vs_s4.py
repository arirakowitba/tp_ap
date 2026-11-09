import pandas as pd

d1d2 = pd.read_csv('submissions/candidate_d1_d2_combined_on48.csv').set_index('ID')['condition']
s4 = pd.read_csv('submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv').set_index('ID')['condition']

diff = (s4 != d1d2)
print(f"Total flips between d1d2 and s4: {diff.sum()}")

# What directions?
u2n = ((d1d2 == 'used') & (s4 == 'new')).sum()
n2u = ((d1d2 == 'new') & (s4 == 'used')).sum()
print(f"used -> new: {u2n}")
print(f"new -> used: {n2u}")
