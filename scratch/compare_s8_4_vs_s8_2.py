import pandas as pd

s8_2 = pd.read_csv('submissions/portfolio_20261108_podium_target_09490/submission_20261108_02_pure_sellers_and_obsolete_top11.csv').set_index('ID')['condition']
s8_4 = pd.read_csv('submissions/portfolio_20261108_podium_target_09490/submission_20261108_04_vintage_parts_and_antiques_top10.csv').set_index('ID')['condition']
anchor = pd.read_csv('submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv').set_index('ID')['condition']

# s8_2 score: 0.94733 (19892 hits) -> delta vs anchor: -4
# s8_4 score: 0.94752 (19896 hits) -> delta vs anchor: 0 (and +4 over s8_2!)

diff = (s8_4 != s8_2)
print(f"Diffs between s8_4 and s8_2: {diff.sum()}")

diff_ids = s8_4[diff].index.tolist()
for idx in diff_ids:
    print(f"ID {idx:5d} | anchor: {anchor.loc[idx]} | s8_2: {s8_2.loc[idx]} | s8_4: {s8_4.loc[idx]}")
