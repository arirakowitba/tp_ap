"""Download only public pretrained model files and verify frozen SHA256 hashes."""
from pathlib import Path
import argparse,json,hashlib,urllib.request
ROOT=Path(__file__).resolve().parents[1]
def download(folder,manifest):
    spec=json.loads(Path(manifest).read_text());root=Path(folder)
    base='https://huggingface.co/'+spec['model_id']+'/resolve/'+spec['revision']+'/'
    for name,expected in spec['sha256'].items():
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
            temp=p.with_suffix(p.suffix+'.part')
            with urllib.request.urlopen(base+name,timeout=60) as r,temp.open('wb') as f:
                while True:
                    chunk=r.read(1024*1024)
                    if not chunk:break
                    f.write(chunk)
            actual=hashlib.sha256(temp.read_bytes()).hexdigest()
            if actual!=expected:
                temp.unlink();raise ValueError('Pretrained file SHA256 mismatch: '+name)
            temp.replace(p)
        print('VERIFIED',name,expected,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--manifest',default=str(ROOT/'results/round_v3/embedding_weight_hashes.json'))
    args=p.parse_args();download(args.model_dir,args.manifest)
