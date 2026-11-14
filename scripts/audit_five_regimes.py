import csv, hashlib, json, zipfile
from collections import defaultdict, Counter
from pathlib import Path

archive = Path('/workspace/scratch/94f9fc484995/ap_input_20261112/ap-2026-q-2 (1).zip')
base = Path('/workspace/scratch/1a8e5d0ea219/entrega_ap/S4_EVT154_respaldo.csv')
files = [Path('/workspace/scratch/44047ed27ff7') / n for n in ['refit_70k.csv','pruned_historical_bounds.csv','high_margin_contrast.csv']]
train = defaultdict(list)
with zipfile.ZipFile(archive) as z:
    with z.open('train_data.jsonlines') as f:
        for b in f:
            r = json.loads(b)
            train[str(r.get('seller_id'))].append(int(r.get('start_time') or 0))
    with z.open('test_data.jsonlines') as f:
        test = [json.loads(b) for b in f]

def vector(p):
    with p.open(newline='') as f:
        rows = list(csv.DictReader(f))
    out = {int(x['ID']):x['condition'] for x in rows}
    assert len(rows)==30000 and len(out)==30000 and set(out)==set(range(1,30001))
    return out
s4 = vector(base)
candidates = {p.name:vector(p) for p in files}
counts = Counter()
flips = {name:Counter() for name in candidates}
for key,row in enumerate(test,1):
    times = train.get(str(row.get('seller_id')))
    time = int(row.get('start_time') or 0)
    if not times:
        regime = 'new_seller'
    elif time < min(times):
        regime = 'before_train'
    elif time > max(times):
        regime = 'after_train'
    elif any(t == time for t in times):
        regime = 'same_timestamp'
    else:
        regime = 'bracketed'
    counts[regime]+=1
    for name,v in candidates.items():
        flips[name][regime] += (v[key]!=s4[key])
report = {'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'test_regimes':dict(counts),'candidate_flips_by_regime':{k:dict(v) for k,v in flips.items()}}
Path('five_regimes_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
