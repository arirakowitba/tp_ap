"""Append a newly scored submission as an aggregate constraint and recompute candidates.
usage: python add_score.py CSV PUBLIC_SCORE EVENT_NAME   (works on /home/claude/work/r145/V2.npy + V2_meta.json; Npub=20998)
Then: python opt2.py   (recomputes posterior gains under both priors, in-span consensus ranking, prints estimates).
Rules: never add a vector whose score is not user-reported with its LF SHA256; identical vectors with different scores go to quarantine."""
import sys, json, hashlib, numpy as np, pandas as pd
csv, sc, name = sys.argv[1], float(sys.argv[2]), sys.argv[3]
b = open(csv, 'rb').read().replace(b'\r\n', b'\n'); d = pd.read_csv(csv); assert d.ID.tolist() == list(range(1, 30001))
V = np.load('V3.npy'); meta = json.load(open('V3_meta.json')); v = (d.condition.values == 'used').astype(V.dtype)
dup = [m for i, m in enumerate(meta) if (V[i] == v).all()]
if dup: sys.exit(f'QUARANTINE: identical to already-scored {dup[0][0]} ({dup[0][3]})' if abs(dup[0][3] - sc) > 1e-9 else f'already present: {dup[0][0]}')
np.save('V3.npy', np.vstack([V, v])); meta.append([name, csv.split('/')[-1], 'OK', sc, round(sc * 20998)]); json.dump(meta, open('V3_meta.json', 'w'))
print('added', name, 'hits', round(sc * 20998), 'lf_sha256', hashlib.sha256(b).hexdigest(), 'n_vectors', len(meta))
