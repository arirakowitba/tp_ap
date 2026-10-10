"""Time-boxed complementary LightGBM search using the audited cached split."""
from pathlib import Path
import argparse,json,time
import numpy as np,pandas as pd,optuna
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier,early_stopping
from competition_pipeline_v2 import load_data,groups_for,metrics
from optimize_leaderboard import split
ROOT=Path(__file__).resolve().parents[1]
def run(folder,out,seconds=2400):
    out=Path(out);raw,_,_=load_data(folder,out);y=(raw.condition=='used').to_numpy(dtype='int8');t,v=split(y,groups_for(raw),42,2)
    d=pd.read_pickle(out/'matrices.pkl');tr,va=d['tr'],d['va']
    rows=[]
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=7310,n_startup_trials=10),
       storage='sqlite:///'+str(out/'extended_search.sqlite3'),study_name='complementary_lgb',load_if_exists=True)
    def objective(trial):
        params=dict(n_estimators=trial.suggest_int('trees',1000,3500),learning_rate=trial.suggest_float('learning_rate',.012,.07,log=True),
          num_leaves=trial.suggest_int('num_leaves',12,160),min_child_samples=trial.suggest_int('min_child_samples',15,200),
          reg_lambda=trial.suggest_float('reg_lambda',.05,40,log=True),reg_alpha=trial.suggest_float('reg_alpha',.001,10,log=True),
          colsample_bytree=trial.suggest_float('colsample_bytree',.5,1),subsample=trial.suggest_float('subsample',.65,1),
          subsample_freq=1,extra_trees=trial.suggest_categorical('extra_trees',[True,False]),
          max_cat_threshold=trial.suggest_categorical('max_cat_threshold',[4,8,16,32,64]),
          cat_smooth=trial.suggest_float('cat_smooth',2,120,log=True),cat_l2=trial.suggest_float('cat_l2',.1,30,log=True),
          min_split_gain=trial.suggest_float('min_split_gain',0,.08),n_jobs=4,verbosity=-1,random_state=2026+trial.number)
        model=LGBMClassifier(**params);ts=time.monotonic()
        model.fit(tr,y[t],eval_set=[(va,y[v])],eval_metric='binary_logloss',callbacks=[early_stopping(100,verbose=False)])
        p=model.predict_proba(va)[:,1];q=float(max(np.arange(.45,.551,.005),key=lambda q:accuracy_score(y[v],p>=q)))
        params['n_estimators']=int(model.best_iteration_);name=f'lgb_extended_{trial.number}'
        row=dict(name=name,**metrics(y[v],p,q),params=params,seconds=time.monotonic()-ts,feature_set='context_svm')
        np.save(out/(name+'.npy'),p);(out/(name+'.json')).write_text(json.dumps(row,indent=2));rows.append(row)
        pd.DataFrame([{k:v for k,v in r.items() if k!='params'} for r in rows]).to_csv(out/'extended_metrics.csv',index=False)
        print('EXTENDED',name,row['accuracy'],flush=True);return row['accuracy']
    def stop(study,trial):
        if study.best_value>=.95:study.stop()
    study.optimize(objective,timeout=seconds,n_trials=200,callbacks=[stop])
    study.trials_dataframe().to_csv(out/'extended_trials.csv',index=False)
    (out/'extended_best.json').write_text(json.dumps({'trial':study.best_trial.number,'accuracy':study.best_value,'params':study.best_params},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',default=str(ROOT/'results/round_v2'));p.add_argument('--seconds',type=int,default=2400)
    a=p.parse_args();run(a.data_dir,a.output,a.seconds)
