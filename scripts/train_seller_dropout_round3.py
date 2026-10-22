"""Train-only seller dropout on deep+OOF NB teacher features.
Teacher caches are generated exclusively by the parent deep teacher pipeline.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='3'
from pathlib import Path
import sys,json,time,gc,hashlib
import numpy as np,pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score,log_loss
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
import train_deep_teacher_round2 as base
threadpool_limits(3)
O=R/'results/training_round3/dropout';O.mkdir(parents=True,exist_ok=True);D=R/'results/training_round2/deep_teacher';START=time.time();DEADLINE=START+25*60

def state(stage,**kw):
 d={'stage':stage,'elapsed_seconds':time.time()-START,**kw};(O/'progress.json').write_text(json.dumps(d,indent=2));print(d,flush=True)
def wait_teacher(name,tr,ev):
 p=D/(name+'_teacher.npz')
 while not p.exists():
  state('waiting_teacher_'+name)
  if time.time()>DEADLINE:raise TimeoutError('Teacher cache unavailable before25min budget')
  time.sleep(15)
 for attempt in range(20):
  try:
   with np.load(p) as z:
    assert np.array_equal(z['train_rows'],tr) and np.array_equal(z['eval_rows'],ev)
    return z['oof'].copy(),z['p'].copy()
  except (OSError,ValueError,EOFError):time.sleep(1)
 raise RuntimeError('Teacher cache exists but cannot be read completely: '+str(p))
def dropout(A,rows,frac):
 A=A.copy();rng=np.random.default_rng(20261024);draw=rng.random(100000);mask=draw[rows]<frac
 A.loc[mask,'seller_id']=['__drop_row_'+str(i) for i in rows[mask]]
 for c in ['seller_id_price_mean','seller_id_price_std','seller_id_price_size']:
  if c in A:A.loc[mask,c]=-1.
 if 'seller_id_relative_price' in A:A.loc[mask,'seller_id_relative_price']=A.loc[mask,'log_price']+1.
 return A,int(mask.sum())
def metrics(y,p,seen,threshold=.5):
 a=float(accuracy_score(y,p>=threshold));s=float(accuracy_score(y[seen],p[seen]>=threshold));n=float(accuracy_score(y[~seen],p[~seen]>=threshold))
 return {'accuracy':a,'log_loss':float(log_loss(y,p)),'seen_accuracy':s,'new_accuracy':n,'test_mix_accuracy':.716*s+.284*n,'n_seen':int(seen.sum()),'n_new':int((~seen).sum())}
def opts(trees=750,stage='default'):
 directory=O/('training_'+stage);directory.mkdir(parents=True,exist_ok=True)
 return dict(depth=7,iterations=trees,learning_rate=.06,l2_leaf_reg=15,random_strength=.8,rsm=.8,border_count=64,thread_count=3,random_seed=20261023,verbose=200,allow_writing_files=True,train_dir=str(directory.resolve()),save_snapshot=True,snapshot_interval=30,snapshot_file=str((directory/'snapshot.bin').resolve()),loss_function='Logloss')
def main():
 state('load_cached_features');(O/'source_integrity.json').write_text(json.dumps({'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_feature_sha256':hashlib.sha256((D/'features.pkl').read_bytes()).hexdigest(),'source_tune_teacher_sha256':hashlib.sha256((D/'tune_teacher.npz').read_bytes()).hexdigest(),'snapshots':'Per config/stage snapshot every30s; caches and deterministic masks unchanged.'},indent=2));X,docs,y,sellers,qty=base.load();del docs;gc.collect();q=np.load(R/'results/new_models_20261021/partitions.npz');fit,tune,confirm=[q[k] for k in ['fit','tune','confirm']];dev=np.r_[fit,tune]
 po,pe=wait_teacher('tune',fit,tune);A,B=base.attach(X,fit,tune,po,pe,qty);cats=A.select_dtypes(include='object').columns.tolist();seen=np.isin(sellers[tune],sellers[fit]);trials=[]
 for frac in [.2,.5]:
  pth=O/('tune_'+str(frac)+'.npz');meta=O/('tune_'+str(frac)+'.json')
  if pth.exists() and meta.exists():
   z=np.load(pth);assert np.array_equal(z['rows'],tune);p=z['p'];r=json.loads(meta.read_text())
  else:
   train,n=dropout(A,fit,frac);state('tune_fit',fraction=frac,n_masked=n,n_features=train.shape[1]);m=CatBoostClassifier(**opts(stage='tune_'+str(frac)));m.fit(train,y[fit],cat_features=cats,eval_set=(B,y[tune]),early_stopping_rounds=80);p=m.predict_proba(B)[:,1];trees=m.tree_count_;threshold=max([.475,.49,.5,.51,.525],key=lambda t:(metrics(y[tune],p,seen,t)['test_mix_accuracy'],-abs(t-.5)));r={'dropout_fraction':frac,'iterations':trees,'threshold':threshold,'tune':metrics(y[tune],p,seen,threshold)};np.savez_compressed(pth,rows=tune,p=p);meta.write_text(json.dumps(r,indent=2));del m,train;gc.collect()
  trials.append(r);state('tune_complete',result=r)
 choice=max(trials,key=lambda r:(r['tune']['test_mix_accuracy'],-r['tune']['log_loss']));lock={'choice':choice,'trials':trials,'protocol':'Dropout train-only .2/.5. Teacher OOF unchanged. Test mixture .716seen/.284new. Config/threshold selected tune only, no public labels. Reused partitions, not virgin validation.','seller_stat_mask':'Price mean/std/count -> -1; relativeprice -> logprice+1, exactly unseen fallback. Geography remains observed.'};(O/'selection_lock.json').write_text(json.dumps(lock,indent=2));del A,B;gc.collect();frac=choice['dropout_fraction'];threshold=choice['threshold'];trees=choice['iterations']
 po,pe=wait_teacher('confirm',dev,confirm);A,B=base.attach(X,dev,confirm,po,pe,qty);train,n=dropout(A,dev,frac);state('confirm_fit',choice=choice,n_masked=n);m=CatBoostClassifier(**opts(trees,stage='confirm'));m.fit(train,y[dev],cat_features=cats);p=m.predict_proba(B)[:,1];seen=np.isin(sellers[confirm],sellers[dev]);report={**lock,'confirmation':metrics(y[confirm],p,seen,threshold)};np.savez_compressed(O/'confirm.npz',rows=confirm,p=p,y=y[confirm],seen=seen);m.save_model(str(O/'confirmation_model.cbm'))
 nb=np.load(R/'results/new_models_20261021/seller_native_confirm.npz');assert np.array_equal(nb['rows'],confirm);report['native']=metrics(y[confirm],nb['p'],seen)
 deep=D/'confirm.npz'
 if deep.exists():
  b=np.load(deep);assert np.array_equal(b['rows'],confirm);th=json.loads((D/'selection_lock.json').read_text())['threshold'];report['deep_baseline']=metrics(y[confirm],b['p'],seen,th)
 improve=report['confirmation']['test_mix_accuracy']>report.get('deep_baseline',report['native'])['test_mix_accuracy'];report['test_export']='Not refit yet';(O/'report.json').write_text(json.dumps(report,indent=2));state('confirmation_complete',metrics=report['confirmation']);del A,B,train,m;gc.collect()
 # Reuse full teacher only if already available; expensive generation belongs to parent.
 full=D/'full_teacher.npz'
 if improve and full.exists() and time.time()<DEADLINE-180:
  tr=np.arange(70000);ev=np.arange(70000,100000);po,pe=wait_teacher('full',tr,ev);A,B=base.attach(X,tr,ev,po,pe,qty);train,n=dropout(A,tr,frac);state('full_fit');m=CatBoostClassifier(**opts(trees,stage='full'));m.fit(train,y[tr],cat_features=cats);p=m.predict_proba(B)[:,1];np.savez_compressed(O/'test.npz',p=p);m.save_model(str(O/'model.cbm'));report['test_export']='Complete, base full70k, teacher strictlyOOF for training'
 else:report['test_export']='Skipped: not better than deep/native, full teacher unavailable, or time budget. No teacher duplicated.'
 report['seconds']=time.time()-START;(O/'report.json').write_text(json.dumps(report,indent=2));state('done',test_export=report['test_export'])
if __name__=='__main__':main()
