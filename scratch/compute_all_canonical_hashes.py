import hashlib
import pandas as pd

def compute_hashes(path):
    with open(path, 'rb') as f:
        raw = f.read().replace(b'\r\n', b'\n')
    lf_sha = hashlib.sha256(raw).hexdigest()

    df = pd.read_csv(path).sort_values('ID')
    payload = '\n'.join(f'{int(i)}:{str(t)}' for i, t in zip(df['ID'], df['condition'])) + '\n'
    gate_f = hashlib.sha256(payload.encode('utf-8')).hexdigest()

    return lf_sha, gate_f

files = [
    ('CANDIDATE_T30_PLUS_U37.csv', 'results/CANDIDATE_T30_PLUS_U37.csv'),
    ('claude_T30_mas_U36_estricto.csv', 'submissions/claude_20261114/claude_T30_mas_U36_estricto.csv'),
    ('claude_T30_mas_F86_estricto.csv', 'submissions/claude_20261114/claude_T30_mas_F86_estricto.csv'),
    ('claude_T30_mas_U36_mas_F86_estricto.csv', 'submissions/claude_20261114/claude_T30_mas_U36_mas_F86_estricto.csv'),
]

for name, path in files:
    lf, gf = compute_hashes(path)
    print(f"{name}:")
    print(f"  LF SHA-256: {lf}")
    print(f"  Gate F    : {gf}")
