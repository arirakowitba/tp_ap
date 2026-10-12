"""Joint representation ablations using strictly OOF base probabilities.

All numeric blocks supplied to prepare_joint are already aligned to their split.
The caller must build normalized SVM and stacking probabilities inside training
folds; in-sample meta predictions are never valid inputs.
"""
from pathlib import Path
import argparse,gc,json,os,time,signal
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np,pandas as pd,optuna
from lightgbm import LGBMClassifier,early_stopping
from sklearn.metrics import accuracy_score
from embedding_round import title_data
from optimize_leaderboard import split
from competition_pipeline_v2 import metrics
ROOT=Path(__file__).resolve().parents[1]

def prepare_joint(tr,va,stack_oof=None,stack_holdout=None,embedding_pair=None,normalized_pair=None):
    """Reusable for seed42, frozen seed731 confirmation and final refit."""
    left,right=tr.copy().reset_index(drop=True),va.copy().reset_index(drop=True)
    if normalized_pair is not None:
        nt,nv=normalized_pair
        if len(nt)!=len(left) or len(nv)!=len(right):raise ValueError('Normalized alignment mismatch')
        for c in ['svm_oof_probability']+[c for c in nt if c.startswith('norm_')]:
            left[c]=nt[c].to_numpy();right[c]=nv[c].to_numpy()
    if stack_oof is not None:
        if stack_holdout is None or stack_oof.shape!=(len(left),3) or stack_holdout.shape!=(len(right),3):raise ValueError('Stacking alignment mismatch')
        if not np.isfinite(stack_oof).all() or not np.isfinite(stack_holdout).all():raise ValueError('Non-finite OOF block')
        for i,name in enumerate(['text_logistic','text_svm','lgb_structured_rates']):
            left['strict_oof_'+name]=stack_oof[:,i];right['strict_oof_'+name]=stack_holdout[:,i]
    if embedding_pair is not None:
        et,ev=embedding_pair
        if et.shape!=(len(left),384) or ev.shape!=(len(right),384):raise ValueError('Embedding alignment mismatch')
        names=['joint_title_embed_'+str(i) for i in range(384)]
        left=pd.concat([left,pd.DataFrame(et,columns=names)],axis=1)
        right=pd.concat([right,pd.DataFrame(ev,columns=names)],axis=1)
    return left,right

def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.deadline:
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Selection deadline reached')))
            signal.alarm(max(1,int(args.deadline-time.time())))
    titles,y,g,fingerprint=title_data(args.data_dir);del titles;gc.collect()
    t,v=split(y,g,42,2)
    d=pd.read_pickle(Path(args.cache)/'matrices.pkl')
    if d['fingerprint']!=fingerprint:raise ValueError('Dataset fingerprint mismatch')
    rows=pd.read_csv(Path(args.cache)/'selection_rows.csv')
    if not np.array_equal(rows.row_index,v) or not np.array_equal(rows.y_used,y[v]):raise ValueError('Selection row mismatch')
    stack=np.load(out/'stacking_seed42_base.npz')
    if not np.array_equal(stack['train_rows'],t) or not np.array_equal(stack['holdout_rows'],v):raise ValueError('Stack row mismatch')
    normal=pd.read_pickle(out/'normalize_matrices.pkl') if args.normalize else None
    if normal is not None and normal['fingerprint']!=fingerprint:raise ValueError('Normalized fingerprint mismatch')
    emb=np.load(out/'embedding_titles.npy',mmap_mode='r') if args.embedding else None
    tr,va=prepare_joint(d['tr'],d['va'],stack['oof'],stack['holdout'],None if emb is None else (emb[t],emb[v]),None if normal is None else (normal['tr'],normal['va']))
    del d,normal,stack,emb;gc.collect()
    blocks=dict(normalize=args.normalize,embedding=args.embedding,stacking_base=True)
    phase='joint' if args.embedding else ('normstack' if args.normalize else 'stackbase')
    records=[];started=time.time()
    base=json.loads((ROOT/'results/round_v2/lgb_context_svm_tpe_12.json').read_text())['params']
    def objective(trial):
        cfg=dict(base);cfg.update(n_jobs=2,n_estimators=2200,
         learning_rate=trial.suggest_float('learning_rate',.015,.06,log=True),
         num_leaves=trial.suggest_int('num_leaves',31,145),
         min_child_samples=trial.suggest_int('min_child_samples',30,150),
         reg_lambda=trial.suggest_float('reg_lambda',.1,20,log=True),
         colsample_bytree=trial.suggest_float('colsample_bytree',.6,1.),
         reg_alpha=trial.suggest_float('reg_alpha',.01,3,log=True))
        model=LGBMClassifier(**cfg);model.fit(tr,y[t],eval_set=[(va,y[v])],eval_metric='binary_logloss',callbacks=[early_stopping(100,verbose=False)])
        pred=model.predict_proba(va)[:,1];cfg['n_estimators']=int(model.best_iteration_)
        q=float(max(np.arange(.46,.541,.005),key=lambda z:accuracy_score(y[v],pred>=z)))
        name=f'combined_{phase}_{trial.number}';row=dict(name=name,**metrics(y[v],pred,q),params=cfg,blocks=blocks,feature_set='combined',split_seed=42,train_sha256=fingerprint,confirmation=None,seconds=time.time()-started)
        np.save(out/(name+'.npy'),pred);(out/(name+'.json')).write_text(json.dumps(row,indent=2))
        pd.DataFrame({'row_index':v,'y_used':y[v],'probability_used':pred}).to_csv(out/(name+'_predictions.csv'),index=False)
        records.append(row);pd.DataFrame([{k:z for k,z in r.items() if k not in ('params','blocks')} for r in records]).to_csv(out/f'combined_{phase}_metrics.csv',index=False)
        print('COMBINED',name,row['accuracy'],cfg['n_estimators'],flush=True)
        return row['accuracy']
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=2026,n_startup_trials=5))
    study.enqueue_trial({k:base[k] for k in ['learning_rate','num_leaves','min_child_samples','reg_lambda','colsample_bytree','reg_alpha']})
    study.optimize(objective,n_trials=args.trials,timeout=max(1,args.deadline-time.time()) if args.deadline else None)
    (out/f'combined_{phase}_selected.json').write_text(json.dumps(max(records,key=lambda r:r['accuracy']),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',default=str(ROOT/'results/round_v3'));p.add_argument('--cache',default=str(ROOT/'results/round_v3/cache_base'));p.add_argument('--deadline',type=float);p.add_argument('--trials',type=int,default=12);p.add_argument('--normalize',action='store_true');p.add_argument('--embedding',action='store_true')
    try:run(p.parse_args())
    except TimeoutError as e:print(e,flush=True);raise SystemExit(124)
