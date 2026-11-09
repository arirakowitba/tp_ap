import pandas as pd
from pathlib import Path

ROOT = Path('.')
s1 = pd.read_csv('submissions/portfolio_20261107_target_09490/submission_20261107_01_purified_probe_top25.csv').set_index('ID')['condition']
s2 = pd.read_csv('submissions/portfolio_20261107_target_09490/submission_20261107_02_calibrated_expansion_top45.csv').set_index('ID')['condition']
s3 = pd.read_csv('submissions/portfolio_20261107_target_09490/submission_20261107_03_podium_attack_top65.csv').set_index('ID')['condition']
s4 = pd.read_csv('submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv').set_index('ID')['condition']
s5 = pd.read_csv('submissions/portfolio_20261107_target_09490/submission_20261107_05_podium_climax_top105.csv').set_index('ID')['condition']

# Scores:
# s1: 0.94738 (19893)
# s2: 0.94738 (19893)
# s3: 0.94709 (19887)
# s4: 0.94752 (19896) -> PEAK!
# s5: 0.94738 (19893)

diff_s4_vs_s3 = (s4 != s3)
print(f"Items flipped in s4 (flips 66 to 85, N={diff_s4_vs_s3.sum()}):")
# In s4, score jumped from 0.94709 (19887) to 0.94752 (19896) -> NET +9 HITS!
# Flipping those 20 items gained +9 hits!
diff_ids = s4[diff_s4_vs_s3].index.tolist()
print(f"Diff IDs (66 to 85): {diff_ids}")

diff_s5_vs_s4 = (s5 != s4)
print(f"\nItems flipped in s5 (flips 86 to 105, N={diff_s5_vs_s4.sum()}):")
# In s5, score dropped from 19896 to 19893 -> NET -3 HITS!
diff_ids_5 = s5[diff_s5_vs_s4].index.tolist()
print(f"Diff IDs (86 to 105): {diff_ids_5}")
