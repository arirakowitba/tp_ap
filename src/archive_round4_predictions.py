"""Deterministic gzip checkpoints of expensive per-row probability artifacts."""
from pathlib import Path
import argparse,gzip,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
def archive(output):
    out=Path(output);records=[]
    for source in sorted(out.rglob('*_predictions.csv')):
        stem=source.name[:-len('_predictions.csv')]
        if stem.endswith('_confirmation'):completed=out/'round4_confirmation.json'
        elif stem.endswith('_full'):completed=out/'round4_refit.json'
        elif stem=='round4_selection':completed=out/'round4_selected.json'
        else:completed=source.with_name(stem+'.json')
        if not completed.exists():continue
        target=source.with_suffix('.csv.gz')
        temporary=target.with_suffix(target.suffix+'.tmp')
        with source.open('rb') as inp,temporary.open('wb') as dest:
            with gzip.GzipFile(filename='',mode='wb',fileobj=dest,mtime=0) as compressed:shutil.copyfileobj(inp,compressed)
        temporary.replace(target)
        raw=source.read_bytes()
        with gzip.open(target,'rb') as stream:assert stream.read()==raw
        records.append(dict(path=str(target.relative_to(out)),rows=raw.count(b'\n')-1,
                            uncompressed_sha256=hashlib.sha256(raw).hexdigest(),
                            compressed_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                            bytes=target.stat().st_size,uncompressed_bytes=len(raw)))
    (out/'prediction_archive.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(dict(files=len(records),compressed_bytes=sum(r['bytes'] for r in records))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'results/round_v4'));archive(p.parse_args().output)
