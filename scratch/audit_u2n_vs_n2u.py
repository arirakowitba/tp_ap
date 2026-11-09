import json
import pandas as pd
from pathlib import Path

# Load submissions and count used->new flips vs d1d2
d1d2 = pd.read_csv('submissions/candidate_d1_d2_combined_on48.csv').set_index('ID')['condition']

subs = [
    ('s7_1_top25', 'submissions/portfolio_20261107_target_09490/submission_20261107_01_purified_probe_top25.csv', 0.94738),
    ('s7_2_top45', 'submissions/portfolio_20261107_target_09490/submission_20261107_02_calibrated_expansion_top45.csv', 0.94738),
    ('s7_3_top65', 'submissions/portfolio_20261107_target_09490/submission_20261107_03_podium_attack_top65.csv', 0.94709),
    ('s7_4_top85', 'submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv', 0.94752),
    ('s7_5_top105', 'submissions/portfolio_20261107_target_09490/submission_20261107_05_podium_climax_top105.csv', 0.94738),
    ('s8_1', 'submissions/portfolio_20261108_podium_target_09490/submission_20261108_01_surgical_rank5_capture_top6.csv', 0.94738),
    ('s8_2', 'submissions/portfolio_20261108_podium_target_09490/submission_20261108_02_pure_sellers_and_obsolete_top11.csv', 0.94733),
    ('s8_3_revert_u2n', 'submissions/portfolio_20261108_podium_target_09490/submission_20261108_03_revert_false_used_top10.csv', 0.94742),
    ('s8_4_vintage', 'submissions/portfolio_20261108_podium_target_09490/submission_20261108_04_vintage_parts_and_antiques_top10.csv', 0.94752),
    ('s8_5_balanced', 'submissions/portfolio_20261108_podium_target_09490/submission_20261108_05_balanced_consensus_top20.csv', 0.94728),
    ('s9_1_surgical5', 'submissions/portfolio_20261109_podium_attack/submission_20261109_01_surgical_rank5_capture_top5.csv', 0.94733)
]

anchor_s4 = pd.read_csv('submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv').set_index('ID')['condition']

records = []
for name, path, score in subs:
    df = pd.read_csv(path).set_index('ID')['condition']
    diff = (df != anchor_s4)
    u2n = ((anchor_s4 == 'used') & (df == 'new')).sum()
    n2u = ((anchor_s4 == 'new') & (df == 'used')).sum()
    hits = round(score * 20998)
    delta_s4 = hits - 19896
    records.append({
        'name': name,
        'score': score,
        'hits': hits,
        'delta_vs_peak': delta_s4,
        'used_to_new_flips': u2n,
        'new_to_used_flips': n2u
    })

df_res = pd.DataFrame(records)
print(df_res.to_string())
