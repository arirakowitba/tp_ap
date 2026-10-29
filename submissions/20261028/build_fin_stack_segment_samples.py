"""Recreate two representative 150-row fin_stack probes from bands 60-62.
Requires data.zip with original 70k train and 30k test JSONL; never reads test labels.
"""
import hashlib, json, zipfile
from pathlib import Path
import pandas as pd
H=Path(__file__).resolve().parent
m=json.loads((H/'manifest_bandas_fin_stack.json').read_text())
p59=next(p for p in m['probes'] if p['number']==59)
base=pd.read_csv(H/p59['file'])
mask=base.ID.isin(p59['ids'])
base.loc[mask,'condition']=base.loc[mask,'condition'].map({'new':'used','used':'new'})
assert hashlib.sha256(base.to_csv(index=False).encode()).hexdigest()==m['base_sha256']
pure=pd.read_csv(H/'submission_63_fin_stack_modelo_puro.csv')
with zipfile.ZipFile('data.zip') as z:
    train=pd.DataFrame(json.loads(line) for line in z.open('train_data.jsonlines'))
    test=pd.DataFrame(json.loads(line) for line in z.open('test_data.jsonlines'))
seen=test.seller_id.isin(set(train.seller_id)).to_numpy()
pool=set().union(*(set(p['ids']) for p in m['probes'] if p['number'] in (60,61,62)))
assert len(pool)==820
for name,seg in [('seen',seen),('new',~seen)]:
    choices=[i for i in pool if seg[i-1]]
    assert all(base.condition.iloc[i-1]!=pure.condition.iloc[i-1] for i in choices)
    choices.sort(key=lambda i: hashlib.sha256(f'20261028:fin_stack:{name}:{i}'.encode()).digest())
    chosen=choices[:150]
    output=base.copy(); take=output.ID.isin(chosen)
    output.loc[take,'condition']=pure.loc[take,'condition'].values
    path=H/f'probe_fin_stack_{name}_unprobed150_sobre48.csv'
    output.to_csv(path,index=False)
    print(path.name,hashlib.sha256(path.read_bytes()).hexdigest())
