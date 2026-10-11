"""Prepare the original best-model representation, with strictly fitted text OOF."""
from pathlib import Path
import argparse,json,signal,time,gc,os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from competition_pipeline_v2 import load_data,groups_for
from optimize_leaderboard import enrich,split,numeric_text
from optimize_stack import group_rates,text_predict
ROOT=Path(__file__).resolve().parents[1]
def run(args):
    if args.deadline<=time.time():raise TimeoutError('Deadline expired')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Deadline')));signal.alarm(max(1,int(args.deadline-time.time())))
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    raw,test,audit=load_data(args.data_dir,out)
    y=(raw.condition=='used').to_numpy(dtype='int8');g=groups_for(raw)
    if args.full:
        t,v=np.arange(len(raw)),np.arange(len(test));other_raw=test
    else:
        t,v=split(y,g,args.seed,args.holdout_folds)
        assert not set(g[t])&set(g[v])
        other_raw=raw.iloc[v].reset_index(drop=True)
    fingerprint=dict(train_sha256=audit[0]['sha256'],seed=args.seed,holdout_folds=args.holdout_folds)
    if args.full:fingerprint.update(full=True,test_sha256=audit[1]['sha256'])
    if (out/'stack.pkl').exists() and json.loads((out/'protocol.json').read_text())['fingerprint']==fingerprint:
        print('COMPATIBLE CACHE REUSED',flush=True);return
    fit,ids=enrich(raw.iloc[t].reset_index(drop=True));other,_=enrich(other_raw,ids)
    del raw,test,other_raw;gc.collect();both=pd.concat([fit,other],ignore_index=True)
    tr,va,cats,_,_=numeric_text(both,np.arange(len(t)),np.arange(len(t),len(both)));del both;gc.collect()
    payload=dict(tr=tr,va=va,cats=cats,text_tr=fit.full_text.to_numpy(),text_va=other.full_text.to_numpy(),
        y_train=y[t],y_eval=None if args.full else y[v],t=t,v=v,g_train=g[t],fingerprint=fingerprint)
    pd.to_pickle(payload,out/'native.tmp');os.replace(out/'native.tmp',out/'native.pkl')
    pd.DataFrame({'row_index':v,**({} if args.full else {'y_used':y[v]})}).to_csv(out/'evaluation_rows.csv',index=False)
    (out/'protocol.json').write_text(json.dumps(dict(fingerprint=fingerprint,train_rows=len(t),evaluation_rows=len(v),shared_families=0,attribute_ids_fit_on_train=True),indent=2))
    print('NATIVE CACHE READY',tr.shape,va.shape,flush=True)
    st,sv=group_rates(tr,va,y[t],g[t]);del tr,va,payload;gc.collect()
    oof=np.zeros(len(t));folds=StratifiedGroupKFold(3,shuffle=True,random_state=2026)
    for k,(it,iv) in enumerate(folds.split(np.zeros(len(t)),y[t],g[t])):
        oof[iv]=text_predict(fit.iloc[it],y[t[it]],fit.iloc[iv]);print('TEXT OOF',k,flush=True)
    p=text_predict(fit,y[t],other)
    st['text_oof_probability']=oof;sv['text_oof_probability']=p
    pd.to_pickle(dict(tr=st,va=sv,cats=cats,y_train=y[t],y_eval=None if args.full else y[v],t=t,v=v,g_train=g[t],fingerprint=fingerprint),out/'stack.tmp')
    os.replace(out/'stack.tmp',out/'stack.pkl');np.save(out/'text_probability.npy',p)
    print('STACK CACHE READY',st.shape,sv.shape,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--holdout-folds',type=int,default=3);p.add_argument('--deadline',type=float,required=True);p.add_argument('--full',action='store_true')
    run(p.parse_args())
