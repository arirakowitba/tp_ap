"""Six-trial TPE on fixed deep/OOF teacher features; no public labels.
Selection and threshold on tune only. Reused inherited partitions are not virgin.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,time,json,gc,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,log_loss
from catboost import CatBoostClassifier
import optuna
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
import train_deep_teacher_round2 as parent
threadpool_limits(2)
O=R/'results/training_round3/tpe';O.mkdir(parents=True,exist_ok=True);START=time.time()
def state(stage,**kw):
 d={'stage':stage,'elapsed_seconds':time.time()-START,**kw};(O/'progress.json').write_text(json.dumps(d,indent=2));print(d,flush=True)
def teacher_cache(tr,ev,name,wait=90):
 f=parent.O/(name+'_teacher.npz');deadline=time.time()+wait
 while not f.exists():
  if time.time()>deadline:raise TimeoutError(str(f))
  state('waiting_parent_teacher',name=name);time.sleep(10)
 for _ in range(30):
  try:
   z=np.load(f);assert np.array_equal(z['train_rows'],tr) and np.array_equal(z['eval_rows'],ev);a,b=z['oof'].copy(),z['p'].copy();assert np.isfinite(a).all() and np.isfinite(b).all();return a,b
  except (EOFError,ValueError,OSError):time.sleep(1)
 raise RuntimeError('Cannot read stable parent teacher '+str(f))
def metrics(y,p,seen,th):
 a=float(accuracy_score(y,p>=th));s=float(accuracy_score(y[seen],p[seen]>=th));n=float(accuracy_score(y[~seen],p[~seen]>=th));return {'accuracy':a,'seen_accuracy':s,'new_accuracy':n,'test_mix_accuracy':.716*s+.284*n,'log_loss':float(log_loss(y,p))}
def options(params,iterations,stage):
 return dict(**params,iterations=iterations,loss_function='Logloss',random_seed=20261024,thread_count=2,border_count=64,verbose=200,allow_writing_files=True,train_dir=str(O/('train_'+stage)),save_snapshot=True,snapshot_interval=30,snapshot_file=str((O/(stage+'_snapshot.bin')).resolve()))
def main():
 X,docs,y,sellers,qty=parent.load();del docs;part=np.load(R/'results/new_models_20261021/partitions.npz');fit,tune,confirm=[part[k] for k in ['fit','tune','confirm']];dev=np.r_[fit,tune]
 po,p=teacher_cache(fit,tune,'tune');A,B=parent.attach(X,fit,tune,po,p,qty);cats=A.select_dtypes(include='object').columns.tolist();seen=np.isin(sellers[tune],sellers[fit])
 manifest={'data_fingerprint':json.loads((parent.O/'data_fingerprint.json').read_text()),'features_sha256':hashlib.file_digest(open(parent.O/'features.pkl','rb'),'sha256').hexdigest(),'protocol':'6serialTPE trials,2threads, fixed teacherOOF. Tune weightedaccuracy .716seen/.284new and threshold grid. Logloss only tie-break after trials. Confirmation not selected on. Reused splits.'};(O/'protocol.json').write_text(json.dumps(manifest,indent=2))
 study=optuna.create_study(study_name='deep_tpe_round3',storage='sqlite:///'+str((O/'optimization.sqlite3').resolve()),load_if_exists=True,direction='maximize',sampler=optuna.samplers.TPESampler(seed=20261024,n_startup_trials=3))
 if not study.trials:study.enqueue_trial({'depth':7,'learning_rate':.06,'l2_leaf_reg':15.,'random_strength':.8,'rsm':.8,'bagging_temperature':1.})
 def objective(t):
  params={'depth':t.suggest_int('depth',6,8),'learning_rate':t.suggest_float('learning_rate',.04,.09,log=True),'l2_leaf_reg':t.suggest_float('l2_leaf_reg',7.,30.,log=True),'random_strength':t.suggest_float('random_strength',.3,1.3),'rsm':t.suggest_float('rsm',.65,1.),'bagging_temperature':t.suggest_float('bagging_temperature',.1,1.)}
  name='trial_'+str(t.number);state('trial_fit',number=t.number,params=params);m=CatBoostClassifier(**options(params,1000,name));m.fit(A,y[fit],cat_features=cats,eval_set=(B,y[tune]),early_stopping_rounds=80);p=m.predict_proba(B)[:,1]
  candidates=[(th,metrics(y[tune],p,seen,th)) for th in [.475,.49,.5,.51,.525]];th,met=max(candidates,key=lambda q:(q[1]['test_mix_accuracy'],-q[1]['log_loss'],-abs(q[0]-.5)));np.savez_compressed(O/(name+'_tune.npz'),rows=tune,p=p)
  result={'trial':t.number,'params':params,'trees':m.tree_count_,'threshold':th,'tune':met,'elapsed_seconds':time.time()-START};(O/(name+'.json')).write_text(json.dumps(result,indent=2));t.set_user_attr('result',result);state('trial_complete',**result);del m;gc.collect();return met['test_mix_accuracy']
 # Study persists; interrupted RUNNING trials may remain but are not selected.
 for frozen in study.get_trials(deepcopy=False,states=(optuna.trial.TrialState.RUNNING,)):
  if time.time()-START<850:
   live=optuna.trial.Trial(study,frozen._trial_id);state('resuming_running_trial',number=live.number);value=objective(live);study.tell(live,value)
 complete=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
 while len(complete)<6 and time.time()-START<850:
  study.optimize(objective,n_trials=1,catch=(RuntimeError,));complete=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
 if not complete:raise RuntimeError('No completed TPE trial')
 selected=max([t.user_attrs['result'] for t in complete],key=lambda q:(q['tune']['test_mix_accuracy'],-q['tune']['log_loss'],-q['trial']));lock={'selected':selected,'completed_trials':len(complete),'selection':'Maximum weighted tuneaccuracy, then minimumtuneLogloss. No confirm consulted.','protocol':manifest['protocol']};(O/'selection_lock.json').write_text(json.dumps(lock,indent=2));zz=np.load(O/('trial_'+str(selected['trial'])+'_tune.npz'));np.savez_compressed(O/'tune.npz',rows=zz['rows'],p=zz['p']);state('selection_locked',selected=selected);del A,B;gc.collect()
 po,p=teacher_cache(dev,confirm,'confirm');A,B=parent.attach(X,dev,confirm,po,p,qty);m=CatBoostClassifier(**options(selected['params'],selected['trees'],'confirm'));state('confirm_fit');m.fit(A,y[dev],cat_features=cats);p=m.predict_proba(B)[:,1];seen=np.isin(sellers[confirm],sellers[dev]);met=metrics(y[confirm],p,seen,selected['threshold']);np.savez_compressed(O/'confirm.npz',rows=confirm,p=p,y=y[confirm],seen=seen);m.save_model(str(O/'confirmation_model.cbm'));report={**lock,'confirmation':met,'parent_confirmation':json.loads((parent.O/'report.json').read_text()).get('confirmation'),'scope':'Exploratory search on reused partitions, no public labels and no confirm tuning.'};(O/'report.json').write_text(json.dumps(report,indent=2));state('confirmation_complete',metrics=met)
 # Full refit remains limited to a promising winner selected on tune.
 parent_lock=json.loads((parent.O/'selection_lock.json').read_text());parent_tune=np.load(parent.O/'tune.npz');pt=parent_tune['p'];st=np.isin(sellers[tune],sellers[fit]);parent_tune_metric=metrics(y[tune],pt,st,parent_lock['threshold']);report['parent_tune']=parent_tune_metric
 promising=selected['tune']['test_mix_accuracy']>=parent_tune_metric['test_mix_accuracy']-.001 and met['accuracy']>=report['parent_confirmation']['accuracy']-.002
 if promising and time.time()-START<1050:
  del A,B,m;gc.collect();tr=np.arange(70000);ev=np.arange(70000,100000)
  try:po,p=teacher_cache(tr,ev,'full',wait=120)
  except TimeoutError:report['full_refit']='Pending parent fullteacher cache';(O/'report.json').write_text(json.dumps(report,indent=2));state('done',full_refit=report['full_refit']);return
  A,B=parent.attach(X,tr,ev,po,p,qty);state('full_fit');m=CatBoostClassifier(**options(selected['params'],selected['trees'],'full'));m.fit(A,y[:70000],cat_features=cats);p=m.predict_proba(B)[:,1];np.savez_compressed(O/'test.npz',p=p);m.save_model(str(O/'model.cbm'));report['full_refit']='complete'
 else:report['full_refit']='Skipped: not near parent quality for ensemble utility, or budget'
 report['seconds']=time.time()-START;(O/'report.json').write_text(json.dumps(report,indent=2));state('done',full_refit=report['full_refit'])
if __name__=='__main__':main()
