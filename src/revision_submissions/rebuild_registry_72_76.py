"""Rebuild submitted72–76 from verified48 JSON and frozen registry; no Kaggle API."""
import argparse,csv,hashlib,json
from pathlib import Path

def encode(labels):
    return ('ID,condition\n'+''.join(f'{i},{labels[str(i)]}\n' for i in range(1,30001))).encode()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base-json',type=Path,required=True)
    ap.add_argument('--registry',type=Path,default=Path('results/revision_submissions/registry_72_76_20261029.json'))
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    base=json.loads(args.base_json.read_text())
    registry=json.loads(args.registry.read_text())
    assert set(base)==set(map(str,range(1,30001)))
    assert set(base.values())<= {'new','used'}
    assert hashlib.sha256(encode(base)).hexdigest()==registry['baseline']['sha256']
    args.out.mkdir(parents=True,exist_ok=True)
    seen=set()
    for block in registry['submissions']:
        labels=base.copy()
        ids=block['ids']
        assert len(ids)==len(set(ids))
        if ids:
            assert not seen.intersection(ids)
            seen.update(ids)
        for i in ids:
            target=block.get('target') or block['targets'][str(i)]
            assert target in {'new','used'} and target!=base[str(i)]
            labels[str(i)]=target
        blob=encode(labels)
        assert hashlib.sha256(blob).hexdigest()==block['sha256']
        rows=list(csv.DictReader(blob.decode().splitlines()))
        assert len(rows)==30000
        assert sum(row['condition']!=base[row['ID']] for row in rows)==block['n_flips']
        dest=args.out/block['file']
        if dest.exists() and dest.read_bytes()!=blob:
            raise RuntimeError(f'Refusing to overwrite different bytes: {dest}')
        dest.write_bytes(blob)
        print(block['number'],block['score'],block['sha256'],dest)
if __name__=='__main__':
    main()
