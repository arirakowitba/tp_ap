import pandas as pd
from pathlib import Path

ROOT = Path('.')
p_anchor = ROOT / 'submissions' / 'portfolio_20261107_target_09490' / 'submission_20261107_04_podium_breakthrough_top85.csv'
df_anchor = pd.read_csv(p_anchor).set_index('ID')['condition']

target_dir = ROOT / 'submissions' / 'portfolio_20261109_podium_attack'
files = [
    ('submission_20261109_01_surgical_rank5_capture_top5.csv', 5),
    ('submission_20261109_02_calibrated_milestone_top10.csv', 10),
    ('submission_20261109_03_rank4_breakthrough_top15.csv', 15),
    ('submission_20261109_04_podium_attack_top25.csv', 25),
    ('submission_20261109_05_podium_climax_top38.csv', 38)
]

for fname, exp_diff in files:
    p = target_dir / fname
    assert p.exists(), f'Missing file {fname}'
    df = pd.read_csv(p)
    assert len(df) == 30000, f'Row count mismatch in {fname}: {len(df)}'
    assert list(df.columns) == ['ID', 'condition'], f'Column mismatch in {fname}: {df.columns}'
    assert df['condition'].isnull().sum() == 0, f'Nulls found in {fname}'
    assert set(df['condition'].unique()) == {'new', 'used'}, f'Invalid labels in {fname}'
    
    diff = (df.set_index('ID')['condition'] != df_anchor).sum()
    assert diff == exp_diff, f'Diff mismatch in {fname}: expected {exp_diff}, got {diff}'
    print(f'PASS: {fname} | Rows: {len(df)} | Columns: {list(df.columns)} | Diff vs Anchor: {diff}')

print('\nALL 5 SUBMISSIONS 100% VALIDATED AND VERIFIED!')
