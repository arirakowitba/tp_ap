"""Audit & Verification Script for Portfolio 20261108."""
import hashlib, json
from pathlib import Path
import pandas as pd

ROOT = Path('.')
PORTFOLIO_DIR = ROOT / 'submissions' / 'portfolio_20261108_podium_target_09490'
p_s4 = ROOT / 'submissions' / 'portfolio_20261107_target_09490' / 'submission_20261107_04_podium_breakthrough_top85.csv'
df_s4 = pd.read_csv(p_s4).set_index('ID')['condition']

with open(PORTFOLIO_DIR / 'portfolio_manifest_20261108.json', encoding='utf-8') as f:
    manifest = json.load(f)

print(f"Loaded manifest with {len(manifest)} slots.")

for entry in manifest:
    p = Path(entry['path'])
    assert p.exists(), f"File missing: {p}"
    
    # 1. Row count & format
    df = pd.read_csv(p)
    assert len(df) == 30000, f"Invalid rows in {p.name}: {len(df)}"
    assert list(df.columns) == ['ID', 'condition'], f"Invalid columns in {p.name}: {df.columns}"
    assert set(df['condition'].unique()) == {'new', 'used'}, f"Invalid conditions: {df['condition'].unique()}"
    assert df['ID'].tolist() == list(range(1, 30001)), f"Invalid ID ordering in {p.name}"

    # 2. Check LF Hash
    data = p.read_bytes().replace(b'\r\n', b'\n')
    lf_hash = hashlib.sha256(data).hexdigest()
    assert lf_hash == entry['sha256_lf'], f"Hash mismatch in {p.name}: {lf_hash} != {entry['sha256_lf']}"

    # 3. Flips vs Slot 4 anchor
    s_cond = df.set_index('ID')['condition']
    diff = s_cond[s_cond != df_s4]
    assert len(diff) == entry['flips_vs_s4'], f"Flip count mismatch: {len(diff)} != {entry['flips_vs_s4']}"
    assert (diff == 'used').all(), f"Found illegal used->new flips in {p.name}!"

    print(f"Slot {entry['slot']}: {entry['filename']} PASSED ALL AUDITS! Flips vs s4: {len(diff)} (strictly new->used), Min Tests: {entry['min_times_tested']}, SHA: {lf_hash}")

print("\n100% of Pristine Portfolio 20261108 files passed verification!")
