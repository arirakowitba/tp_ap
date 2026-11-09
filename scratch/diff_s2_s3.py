import pandas as pd

s2 = pd.read_csv('submissions/submission_20261109_02_pure_used_capture_top6.csv').set_index('ID')['condition']
s3 = pd.read_csv('submissions/submission_20261109_03_pure_used_expansion_top12.csv').set_index('ID')['condition']
anchor = pd.read_csv('submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv').set_index('ID')['condition']

# Diffs between s2 and s3
diff = (s2 != s3)
print(f"Total differences between s2 and s3: {diff.sum()}")

diff_ids = s2[diff].index.tolist()
for idx in diff_ids:
    print(f"ID {idx:5d} | anchor: {anchor.loc[idx]} | s2: {s2.loc[idx]} | s3: {s3.loc[idx]}")
