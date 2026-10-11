"""Bounded complementarity search using listing media metadata and text.

Only the seed42 selection feature cache and original selection probabilities are
read. Each LightGBM fit has a fresh subprocess and a hard wall-clock deadline.
No pixels, frames, confirmation labels or remote services enter this search.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def media_columns(columns, group):
    columns = [c for c in columns if c.startswith('media_')]
    if group == 'all':
        return columns
    if group == 'image':
        return [c for c in columns if c != 'media_has_video']
    if group == 'orientation':
        return [c for c in columns if any(word in c for word in ('aspect','portrait','landscape','square'))]
    if group == 'dimensions':
        return [c for c in columns if any(word in c for word in ('width','height','area','large','resolution','size_available'))]
    raise ValueError('Unknown media metadata group.')


def worker(args):
    import numpy as np
    import pandas as pd
    from lightgbm import LGBMClassifier, early_stopping
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
    started = time.time()
    request = json.loads(Path(args.request).read_text())
    data = pd.read_pickle(args.cache)
    fp = data['fingerprint']
    if fp.get('seed') != 42 or fp.get('holdout_folds') != 3 or len(data['t']) != 49000 or len(data['v']) != 21000:
        raise ValueError('Search requires the frozen seed42 49k/21k selection cache.')
    tr, va = data['tr'], data['va']
    for c in data['cats']:
        dtype = pd.CategoricalDtype(categories=pd.Index(tr[c].unique()))
        tr[c], va[c] = tr[c].astype(dtype), va[c].astype(dtype)
    media = pd.read_csv(args.media_features).set_index('row_index', verify_integrity=True)
    cols = media_columns(media.columns, request['media_group'])
    if not cols or any(c in tr for c in cols):
        raise ValueError('Invalid or duplicate media metadata columns.')
    for frame, rows in ((tr, data['t']), (va, data['v'])):
        values = media.loc[rows, cols]
        for c in cols:
            frame[c] = pd.to_numeric(values[c], errors='raise').to_numpy(dtype='float32')
    del media
    prior_frame = pd.read_csv(args.prior_selection).set_index('row_index', verify_integrity=True)
    if set(prior_frame.index) != set(map(int, data['v'])):
        raise ValueError('Prior selection row set differs.')
    prior_frame = prior_frame.loc[data['v']]
    y = data['y_eval']
    if not np.array_equal(prior_frame.y_used, y):
        raise ValueError('Prior selection labels differ.')
    prior = prior_frame.probability_used.to_numpy()
    params = dict(request['params'])
    model = LGBMClassifier(**params)
    model.fit(tr, data['y_train'], eval_set=[(va,y)], eval_metric='binary_logloss',
              callbacks=[early_stopping(100, verbose=False)])
    p = model.predict_proba(va)[:,1]
    if not np.isfinite(p).all():
        raise ValueError('Nonfinite model probabilities.')
    blend = .6*prior + .4*p
    name = 'r4_lgb_extended_'+str(request['trial'])
    result = dict(name=name,family='lgb',kind='lgb',params=dict(params,n_estimators=int(model.best_iteration_)),
        use_media=True,media_columns=cols,media_group=request['media_group'],
        uses_pixels=False,uses_video_frames=False,image_pixels_used=False,video_frames_used=False,
        fingerprint=fp,protocol='selection42_49k_21k',threshold=.5,
        accuracy=float(accuracy_score(y,p>=.5)),balanced_accuracy=float(balanced_accuracy_score(y,p>=.5)),
        macro_f1=float(f1_score(y,p>=.5,average='macro')),logloss=float(log_loss(y,p)),
        objective_blend_accuracy=float(accuracy_score(y,blend>=.5)),
        objective='fixed .6 prior + .4 new, threshold .5',
        prior_accuracy=float(accuracy_score(y,prior>=.5)),n_features=len(tr.columns),
        n_train=len(data['t']),n_eval=len(data['v']),seconds=time.time()-started,
        selection_only=True,confirmation_used=False,completed_at=time.time())
    out = Path(args.output)
    pd.DataFrame(dict(row_index=data['v'],y_used=y,probability_used=p)).to_csv(out/(name+'_predictions.csv'),index=False)
    np.save(out/(name+'.npy'),p)
    model.booster_.save_model(str(out/(name+'.txt')))
    atomic_json(out/(name+'.json'),result)
    print('RESULT',name,result['accuracy'],result['objective_blend_accuracy'],result['seconds'],flush=True)


def refresh(out, study=None):
    import pandas as pd
    rows = [json.loads(p.read_text()) for p in out.glob('r4_lgb_extended_*.json')]
    if rows:
        pd.DataFrame([{k:v for k,v in row.items() if k not in {'params','media_columns','fingerprint'}}
                     for row in rows]).to_csv(out/'extended_metrics.csv',index=False)
        atomic_json(out/'extended_best.json',max(rows,key=lambda r:r['objective_blend_accuracy']))
        print('BEST',len(rows),max(r['accuracy'] for r in rows),max(r['objective_blend_accuracy'] for r in rows),flush=True)
    if study is not None:
        atomic_json(out/'extended_execution.json',dict(deadline=study.user_attrs.get('deadline'),
            trials=len(study.trials),completed=sum(t.state.name=='COMPLETE' for t in study.trials),
            stopped_at=time.time(),confirmation_used=False,image_pixels_used=False,video_frames_used=False))


def run(args):
    import optuna
    if args.deadline <= time.time():
        raise TimeoutError('Selection search deadline expired.')
    out = Path(args.output); out.mkdir(parents=True,exist_ok=True)
    base = json.loads((ROOT/'optimization/lgb_tpe_2_params.json').read_text())
    base.update(n_jobs=2,n_estimators=3500,random_state=2026)
    storage='sqlite:///'+str((out/'study.sqlite').resolve())
    study=optuna.create_study(study_name='selection42_metadata_complementarity',storage=storage,load_if_exists=True,
        direction='maximize',sampler=optuna.samplers.TPESampler(seed=20261011,n_startup_trials=7))
    study.set_user_attr('deadline',args.deadline)
    if not study.trials:
        first={k:base[k] for k in ('learning_rate','num_leaves','min_child_samples','reg_lambda','reg_alpha','colsample_bytree','max_cat_threshold','cat_smooth')}
        study.enqueue_trial(dict(first,media_group='all'))
        original_best=ROOT/'results/round_v4/r4_lgb_media_tpe_9.json'
        if original_best.exists():
            pp=json.loads(original_best.read_text())['params']
            study.enqueue_trial(dict({k:pp[k] for k in first},media_group='all'))
    def objective(trial):
        if args.deadline-time.time()<35:
            raise optuna.TrialPruned('Insufficient deadline budget for next fit.')
        params=dict(base,num_leaves=trial.suggest_int('num_leaves',40,220),
            min_child_samples=trial.suggest_int('min_child_samples',20,100),
            learning_rate=trial.suggest_float('learning_rate',.012,.06,log=True),
            reg_lambda=trial.suggest_float('reg_lambda',.05,15,log=True),
            reg_alpha=trial.suggest_float('reg_alpha',1e-4,3,log=True),
            colsample_bytree=trial.suggest_float('colsample_bytree',.6,.98),
            max_cat_threshold=trial.suggest_categorical('max_cat_threshold',[8,16,32]),
            cat_smooth=trial.suggest_float('cat_smooth',5,50))
        group=trial.suggest_categorical('media_group',['all','image','orientation','dimensions'])
        request=out/('request_'+str(trial.number)+'.json')
        atomic_json(request,dict(trial=trial.number,params=params,media_group=group))
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker','--cache',args.cache,
            '--media-features',args.media_features,'--prior-selection',args.prior_selection,
            '--output',str(out),'--request',str(request),'--deadline',str(args.deadline)]
        with (out/('trial_'+str(trial.number)+'.log')).open('w') as logfile:
            process=subprocess.Popen(command,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=process.wait(timeout=max(1,args.deadline-time.time()-3))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
                raise optuna.TrialPruned('Hard selection deadline terminated native fit.')
        if code:
            raise optuna.TrialPruned('Worker failed, inspect trial log.')
        result=json.loads((out/('r4_lgb_extended_'+str(trial.number)+'.json')).read_text())
        trial.set_user_attr('standalone_accuracy',result['accuracy'])
        trial.set_user_attr('best_iteration',result['params']['n_estimators'])
        return result['objective_blend_accuracy']
    def callback(study,trial):
        refresh(out,study)
        if time.time() >= args.deadline-35 or (trial.value is not None and trial.value >= .95):
            study.stop()
    study.optimize(objective,n_trials=max(0,args.trials-len(study.trials)),
        timeout=max(0,args.deadline-time.time()-5),callbacks=[callback])
    refresh(out,study)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',default=str(ROOT/'results/round_v4/selection/stack.pkl'))
    parser.add_argument('--media-features',default=str(ROOT/'results/round_v4/media_features_train.csv'))
    parser.add_argument('--prior-selection',default=str(ROOT/'optimization/holdout_predictions.csv'))
    parser.add_argument('--output',default=str(ROOT/'results/round_v4/extended'))
    parser.add_argument('--trials',type=int,default=50)
    parser.add_argument('--deadline',type=float,required=True)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--request')
    args=parser.parse_args()
    worker(args) if args.worker else run(args)


if __name__=='__main__':main()
