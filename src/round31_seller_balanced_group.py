"""Group-held-out HistGB with group-OOF text logits, paired to round 29."""
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

os.environ['AP_ROUND5_GROUP'] = '1'
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root.parent / 'optimized'))
from round31_seller_balanced_source import run_stage
from src.ap2026q2_common import build_features
from src.ap2026q2_deep_features import build_deep_features

seed = int(os.environ.get('AP_GROUP_SEED', '883'))
prefix = 'round29' if seed == 883 else f'round29_seed{seed}'
with zipfile.ZipFile(root.parent / 'recovered/Analisis rpedictivo/data.zip') as z:
    with z.open('train_data.jsonlines') as f:
        raw = pd.DataFrame(json.loads(line) for line in f)
y = (raw.pop('condition') == 'used').to_numpy().astype('int8')
docs = pd.read_pickle(root / 'original_docs.pkl')
assert np.array_equal(docs.iloc[:70000]['y'].astype('int8'), y)
df = build_deep_features(raw, build_features(raw), include_post_listing=False)
df['seller_id'] = raw['seller_id'].astype(str)
df['docs'] = docs.iloc[:70000]['docs'].fillna('').astype(str).to_numpy()
df['y'] = y
hold = np.load(root / f'{prefix}_no_seller_id.npz')
va = hold['rows']
tr = np.setdiff1d(np.arange(70000), va)
assert np.array_equal(np.sort(va), np.sort(np.load(root / f'{prefix}_no_seller_id.npz')['rows']))
assert not set(df.seller_id.iloc[tr]) & set(df.seller_id.iloc[va])
p, info = run_stage(df, y, tr, va, f'group_{seed}')
out = {'seed': seed, 'n_train': len(tr), 'n_valid': len(va),
       'accuracy_05': float(accuracy_score(y[va], p >= .5)),
       'accuracy_historical_0475': float(accuracy_score(y[va], p >= .475)),
       'log_loss': float(log_loss(y[va], p)), 'info': info}
np.savez_compressed(root / f'round31_balanced_group_{seed}.npz', rows=va, y=y[va], p=p)
(root / f'round31_balanced_group_{seed}.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
