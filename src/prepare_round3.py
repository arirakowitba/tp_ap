from pathlib import Path
import sys,time,json,os,argparse,signal
import pandas as pd,numpy as np
from competition_pipeline_v2 import load_data,groups_for,matrices
from optimize_leaderboard import split
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--data-dir',default=str(ROOT.parent/'data'));p.add_argument('--output',default=str(ROOT/'results/round_v3/cache_base'));p.add_argument('--deadline',type=float);args=p.parse_args()
if args.deadline and hasattr(signal, 'SIGALRM'):
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Global deadline')))
    signal.alarm(max(1,int(args.deadline-time.time())))
OUT=Path(args.output);OUT.mkdir(parents=True,exist_ok=True)
raw,test,audits=load_data(args.data_dir,OUT)
del test
y=(raw.condition=='used').to_numpy(dtype='int8');g=groups_for(raw);t,v=split(y,g,42,2)
if (OUT/'matrices.pkl').exists():
    d=pd.read_pickle(OUT/'matrices.pkl')
    if d.get('fingerprint')==audits[0]['sha256'] and np.array_equal(d.get('t'),t) and np.array_equal(d.get('v'),v):
        print('CACHE REUSED',flush=True);raise SystemExit(0)
pd.DataFrame({'row_index':v,'y_used':y[v]}).to_csv(OUT/'selection_rows.csv',index=False)
tr,va,cats=matrices(raw.iloc[t].reset_index(drop=True),y[t],g[t],raw.iloc[v].reset_index(drop=True),OUT)
pd.to_pickle(dict(tr=tr,va=va,fingerprint=audits[0]['sha256'],t=t,v=v,cats=cats),OUT/'matrices.tmp')
os.replace(OUT/'matrices.tmp',OUT/'matrices.pkl')
print('CACHE READY',tr.shape,va.shape,flush=True)
