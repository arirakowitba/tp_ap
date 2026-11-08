import pandas as pd, json
from pathlib import Path

ROOT = Path('.')

subs = [
    ('sub_94695', 'submissions/today_20261106_calibrated/submission_20261105_05b_climax_expanded_top169.csv', 19884),
    ('sub_94714', 'submissions/today_20261106_calibrated/submission_20261106_04_pure_unanimous_consumer_top7.csv', 19888),
    ('sub_94719', 'submissions/today_20261106_calibrated/submission_20261106_05_climax_consumer_unanimous_top15.csv', 19889),
    ('sub_94738', 'submissions/portfolio_20261107_target_09490/submission_20261107_01_purified_probe_top25.csv', 19893),
    ('sub_94752', 'submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv', 19896),
]

p_d1d2 = ROOT / 'submissions' / 'candidates_20261104' / 'candidate_01_d1d2_plus_upper10.csv'
d1_df = pd.read_csv(p_d1d2).set_index('ID')['condition']

data = {}
for name, p, hits in subs:
    path = ROOT / p
    if path.exists():
        df = pd.read_csv(path).set_index('ID')['condition']
        diff_used = df[(df == 'used') & (d1_df == 'new')].index
        diff_new = df[(df == 'new') & (d1_df == 'used')].index
        data[name] = {'hits': hits, 'flips_used': set(diff_used), 'flips_new': set(diff_new), 'series': df}
        print(f"{name:10s} | hits={hits} | new->used: {len(diff_used)} | used->new: {len(diff_new)}")

# Find items flipped in sub_94752 that were NOT in sub_94738
s752 = data['sub_94752']['series']
s738 = data['sub_94738']['series']
diff_752_738 = s752[s752 != s738]
print(f"\nDifferences between 0.94752 (19896) and 0.94738 (19893): {len(diff_752_738)}")
print(diff_752_738.value_counts())
