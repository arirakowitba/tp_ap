import time,json,gc
import optuna,numpy as np,pandas as pd
from lightgbm import LGBMClassifier,early_stopping
from sklearn.metrics import accuracy_score
from optimize_leaderboard import OUT,record

def main():
 tr,va,y,yv=pd.read_pickle(OUT/'stack_matrix_cache.pkl')
 study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=2026,n_startup_trials=6),storage='sqlite:///'+str(OUT/'tpe_study.sqlite3'),study_name='lgb_grouped_accuracy',load_if_exists=True)
 def objective(trial):
  params=dict(n_estimators=2500,learning_rate=trial.suggest_float('learning_rate',.015,.09,log=True),num_leaves=trial.suggest_int('num_leaves',16,96),min_child_samples=trial.suggest_int('min_child_samples',15,120),reg_lambda=trial.suggest_float('reg_lambda',.5,40,log=True),reg_alpha=trial.suggest_float('reg_alpha',.001,10,log=True),colsample_bytree=trial.suggest_float('colsample_bytree',.65,1),subsample=trial.suggest_float('subsample',.7,1),subsample_freq=1,max_cat_threshold=trial.suggest_categorical('max_cat_threshold',[8,16,32,64]),cat_smooth=trial.suggest_float('cat_smooth',5,100,log=True),n_jobs=8,verbosity=-1,random_state=42)
  m=LGBMClassifier(**params);ts=time.time();m.fit(tr,y,eval_set=[(va,yv)],eval_metric='binary_logloss',callbacks=[early_stopping(100,verbose=False)])
  p=m.predict_proba(va)[:,1];thresholds=np.arange(.46,.541,.005);scores=[accuracy_score(yv,p>=t) for t in thresholds];i=int(np.argmax(scores))
  trial.set_user_attr('threshold',float(thresholds[i]));trial.set_user_attr('best_iteration',m.best_iteration_)
  name=f'lgb_tpe_{trial.number}';record(name,p,yv,seconds=time.time()-ts,best_iteration=m.best_iteration_)
  (OUT/(name+'_params.json')).write_text(json.dumps(params));del m;gc.collect()
  return float(scores[i])
 def stop_on_goal(study,trial):
  if study.best_value>=.95:study.stop()
 study.optimize(objective,n_trials=20,callbacks=[stop_on_goal])
 study.trials_dataframe().to_csv(OUT/'tpe_trials.csv',index=False)
 (OUT/'tpe_best.json').write_text(json.dumps({'trial':study.best_trial.number,'accuracy':study.best_value,'params':study.best_params,'attributes':study.best_trial.user_attrs},indent=2))
 print('TPE BEST',study.best_value,flush=True)

if __name__=='__main__':main()
