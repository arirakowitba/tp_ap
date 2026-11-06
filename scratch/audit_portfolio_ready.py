import hashlib
from pathlib import Path
import json
import pandas as pd

PORT_DIR = Path('submissions/tomorrow_20261106_portfolio_09490')
manifest_file = PORT_DIR / 'portfolio_manifest.json'

with open(manifest_file) as f:
    manifest = json.load(f)

print('=== PORTFOLIO INTEGRITY AUDIT ===')
for c in manifest['candidates']:
    f_name = c['file']
    slot = c['slot']
    flips = c['flips_vs_sub2']
    exp = c['expected_score']
    expected_sha = c['sha256_lf']
    
    file_path = PORT_DIR / f_name
    df = pd.read_csv(file_path)
    data = file_path.read_bytes().replace(b'\r\n', b'\n')
    sha = hashlib.sha256(data).hexdigest()
    assert sha == expected_sha, f'Hash mismatch for {f_name}'
    assert len(df) == 30000, f'Invalid rows for {f_name}'
    print(f'Slot {slot}: {f_name}')
    print(f'  Flips vs Sub 2: {flips} | Expected Score: {exp:.5f} | SHA-256 LF: {sha}')
