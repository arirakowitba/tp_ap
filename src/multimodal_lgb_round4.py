"""Ablate media metadata against the previously strongest LightGBM design."""
from pathlib import Path
import argparse,json,time,signal,gc,os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np,pandas as pd,optuna
from lightgbm import LGBMClassifier,early_stopping
from competition_pipeline_v2 import metrics
ROOT=Path(__file__).resolve().parents[1]
def media_pair(path,t,v):
    frame=pd.read_csv(path).set_index('row_index');assert frame.index.is_unique
    cols=[c for c in frame if c.startswith('media_')]
    return frame.loc[t,cols].reset_index(drop=True),frame.loc[v,cols].reset_index(drop=True)
def run(args):
    if args.deadline<=time.time():raise TimeoutError('Deadline expired')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Selection deadline')));signal.alarm(max(1,int(args.deadline-time.time())))
    out=Path(args.output);out.mkdir(exist_ok=True,parents=True);d=pd.read_pickle(args.cache)
    tr,va=d['tr'],d['va'];y,vy=d['y_train'],d['y_eval'];t,v=d['t'],d['v']
    for c in d['cats']:
        dtype=pd.CategoricalDtype(categories=pd.Index(tr[c].unique()));tr[c]=tr[c].astype(dtype);va[c]=va[c].astype(dtype)
    a,b=media_pair(args.media_features,t,v)
    if args.media_group=='image':columns=[c for c in a if c!='media_has_video']
    elif args.media_group=='video':columns=['media_has_video']
    elif args.media_group=='orientation':columns=[c for c in a if any(s in c for s in ('aspect','portrait','landscape','square'))]
    elif args.media_group=='dimensions':columns=[c for c in a if any(s in c for s in ('width','height','area','resolution'))]
    else:columns=list(a.columns)
    a,b=a[columns],b[columns]
    base=json.loads((ROOT/'optimization/lgb_tpe_2_params.json').read_text())
    base.update(n_estimators=2500,n_jobs=2);rows=[]
    def fit(name,params,use_media,fixed=False):
        if time.time()>=args.deadline:raise TimeoutError('Selection deadline')
        x,z=(pd.concat([tr,a],axis=1),pd.concat([va,b],axis=1)) if use_media else (tr,va)
        started=time.time();model=LGBMClassifier(**params)
        if fixed:model.fit(x,y)
        else:model.fit(x,y,eval_set=[(z,vy)],eval_metric='binary_logloss',callbacks=[early_stopping(100,verbose=False)])
        p=model.predict_proba(z)[:,1];frozen=dict(params,n_estimators=int(params['n_estimators'] if fixed else model.best_iteration_))
        row=dict(name=name,family='lgb',kind='lgb',params=frozen,use_media=use_media,uses_pixels=False,uses_video_frames=False,
            protocol='selection42_49k_21k',media_group=args.media_group,
            media_columns=columns if use_media else [],media_dtype='float64',fingerprint=d['fingerprint'],
            **metrics(vy,p,.5),seconds=time.time()-started)
        row['fixed_capacity_ablation']=fixed
        rows.append(row);np.save(out/(name+'.npy'),p);model.booster_.save_model(str(out/(name+'.txt')))
        (out/(name+'.json')).write_text(json.dumps(row,indent=2))
        pd.DataFrame({'row_index':v,'y_used':vy,'probability_used':p}).to_csv(out/(name+'_predictions.csv'),index=False)
        pd.DataFrame([{k:w for k,w in r.items() if k!='params'} for r in rows]).to_csv(out/'lgb_round4_metrics.csv',index=False)
        print('RESULT',name,row['accuracy'],row['seconds'],flush=True);del model,x,z;gc.collect();return row['accuracy']
    if args.fixed_only:
        base['n_estimators']=430
        if args.media_group=='all':fit('r4_lgb_fixed430_original',base,False,True)
        suffix='' if args.media_group=='all' else '_'+args.media_group
        fit('r4_lgb_fixed430_media'+suffix,base,True,True)
        return
    fit('r4_lgb_original',base,False);fit('r4_lgb_media',base,True)
    def objective(trial):
        params=dict(base,num_leaves=trial.suggest_int('num_leaves',40,130),min_child_samples=trial.suggest_int('min_child_samples',20,110),
            learning_rate=trial.suggest_float('learning_rate',.015,.055,log=True),reg_lambda=trial.suggest_float('reg_lambda',.3,12,log=True),
            colsample_bytree=trial.suggest_float('colsample_bytree',.65,.95),random_state=2026)
        return fit('r4_lgb_media_tpe_'+str(trial.number),params,True)
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=2026,n_startup_trials=4))
    study.optimize(objective,n_trials=args.trials,timeout=max(0,args.deadline-time.time()),callbacks=[lambda s,t:s.stop() if t.value and t.value>=.95 else None])
    selected=max(rows,key=lambda r:r['accuracy']);(out/'lgb_round4_selected.json').write_text(json.dumps(selected,indent=2))
    print('BEST',selected['name'],selected['accuracy'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--media-features',required=True);p.add_argument('--output',required=True);p.add_argument('--trials',type=int,default=12);p.add_argument('--deadline',type=float,required=True);p.add_argument('--fixed-only',action='store_true');p.add_argument('--media-group',choices=['all','image','video','orientation','dimensions'],default='all')
    run(p.parse_args())
