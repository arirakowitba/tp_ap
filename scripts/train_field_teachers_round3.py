"""Separate title/warranty cross-fitted teachers added to deep tabular CatBoost.
Reuses the combined teacher from parent deep_teacher; no test/public labels.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,json,time,gc,hashlib,joblib
import numpy as np,pandas as pd
from scipy.special import logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,log_loss
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
threadpool_limits(2)
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
import train_deep_teacher_round2 as parent
threadpool_limits(2)
O=R/'results/training_round3/fields';O.mkdir(parents=True,exist_ok=True);START=time.time()
CFG={'title':{'ngram':[3,5],'max_features':30000},'warranty':{'ngram':[2,5],'max_features':15000},'C':4,'seed':20261023,'folds':3,'NB_ratio':True}
def state(stage,**kw):
 x={'stage':stage,'elapsed_seconds':time.time()-START,**kw};(O/'progress.json').write_text(json.dumps(x,indent=2));print(x,flush=True)
def save_npz(path,**a):
 tmp=path.with_suffix('.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**a)
 os.replace(tmp,path)
def read_fields():
 p=O/'fields_input.npz'
 if p.exists():
  z=np.load(p,allow_pickle=True);return {f:z[f] for f in ['title','warranty']}
 a={'title':[],'warranty':[]}
 for name in ['train_data.jsonlines','test_data.jsonlines']:
  for line in open(R.parent/'recovered/data'/name):
   x=json.loads(line)
   for f in a:a[f].append(str(x.get(f) or '__empty__').lower())
 a={f:np.array(v,dtype=object) for f,v in a.items()};save_npz(p,**a);return a
def fit_field(docs,y,tr,queries,field,model_path=None):
 c=CFG[field];v=TfidfVectorizer(analyzer='char',ngram_range=tuple(c['ngram']),max_features=c['max_features'],min_df=3,sublinear_tf=True,strip_accents='unicode',dtype=np.float32)
 a=v.fit_transform(docs[tr]);plus=np.asarray(a[y[tr]==1].sum(0)).ravel()+1;minus=np.asarray(a[y[tr]==0].sum(0)).ravel()+1;r=np.log((plus/plus.sum())/(minus/minus.sum())).astype(np.float32);m=LogisticRegression(C=4,solver='liblinear',max_iter=300).fit(a.multiply(r).tocsr(),y[tr]);p=[m.predict_proba(v.transform(docs[q]).multiply(r).tocsr())[:,1] for q in queries]
 if model_path:
  tmp=model_path.with_suffix('.tmp');joblib.dump({'vectorizer':v,'ratio':r,'model':m,'rows':tr,'config':CFG[field]},tmp,compress=3);joblib.load(tmp);os.replace(tmp,model_path)
 return p
def field_stage(docs,y,tr,ev,stage,field):
 p=O/(stage+'_'+field+'.npz')
 if p.exists():
  z=np.load(p);assert np.array_equal(z['tr'],tr) and np.array_equal(z['ev'],ev);return z['oof'],z['p']
 oof=np.full(len(tr),np.nan)
 for i,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261023).split(tr,y[tr])):
  fp=O/(stage+'_'+field+'_fold'+str(i)+'.npz')
  if fp.exists():
   z=np.load(fp);assert np.array_equal(z['rows'],tr[b]);pred=z['p']
  else:
   state('field_oof',stage_name=stage,field=field,fold=i);pred=fit_field(docs,y,tr[a],[tr[b]],field)[0];save_npz(fp,rows=tr[b],p=pred)
  oof[b]=pred;gc.collect()
 state('field_full',stage_name=stage,field=field);pred=fit_field(docs,y,tr,[ev],field,O/(stage+'_'+field+'_model.joblib'))[0];assert np.isfinite(oof).all();save_npz(p,tr=tr,ev=ev,oof=oof,p=pred);return oof,pred
def combined(tr,ev,stage):
 p=parent.O/(stage+'_teacher.npz');deadline=time.time()+600
 while not p.exists():
  if time.time()>deadline or time.time()-START>1400:raise TimeoutError('Combined teacher still unavailable; rerun resumes own field caches')
  state('waiting_combined_teacher',stage_name=stage);time.sleep(15)
 for _ in range(5):
  try:
   z=np.load(p);assert np.array_equal(z['train_rows'],tr) and np.array_equal(z['eval_rows'],ev);return z['oof'],z['p']
  except (ValueError,EOFError):time.sleep(2)
 raise RuntimeError('Could not read combined teacher safely')
def matrices(X,raw,y,qty,tr,ev,stage):
 fields={f:field_stage(raw[f],y,tr,ev,stage,f) for f in ['title','warranty']};po,p=combined(tr,ev,stage);a,b=parent.attach(X,tr,ev,po,p,qty)
 for z,k,ix in [(a,0,tr),(b,1,ev)]:
  for f in fields:
   pp=fields[f][k];z['field_'+f+'_p']=pp;z['field_'+f+'_logit']=logit(np.clip(pp,1e-5,1-1e-5));z['field_'+f+'_minus_combined']=pp-z['teacher_nb'].to_numpy()
  z['warranty_text_empty']=(raw['warranty'][ix]=='__empty__').astype(float)
 return a,b
def metric(y,p,seen,th):return {'accuracy':float(accuracy_score(y,p>=th)),'logloss':float(log_loss(y,p)),'seen_accuracy':float(accuracy_score(y[seen],p[seen]>=th)),'new_accuracy':float(accuracy_score(y[~seen],p[~seen]>=th))}
def cat(opts,stage):return CatBoostClassifier(**opts,train_dir=str(O/('snap_'+stage)),snapshot_file=str((O/('snapshot_'+stage+'.bin')).resolve()))
def main():
 X,_,y,sellers,qty=parent.load();raw=read_fields();finger={'config':CFG,'data':json.loads((parent.O/'data_fingerprint.json').read_text()),'parent_features_sha256':hashlib.file_digest((parent.O/'features.pkl').open('rb'),'sha256').hexdigest()};sig=hashlib.sha256(json.dumps(finger,sort_keys=True).encode()).hexdigest();fp=O/'fingerprint.json'
 if fp.exists():assert json.loads(fp.read_text())['signature']==sig,'Cache signature mismatch'
 else:fp.write_text(json.dumps({'signature':sig,**finger},indent=2))
 part=np.load(R/'results/new_models_20261021/partitions.npz');fit,tune,conf=[part[k] for k in ['fit','tune','confirm']];dev=np.r_[fit,tune];opts=dict(depth=7,iterations=750,learning_rate=.06,l2_leaf_reg=15,random_strength=.8,rsm=.8,border_count=64,thread_count=2,random_seed=20261023,verbose=200,allow_writing_files=True,save_snapshot=True,snapshot_interval=30,loss_function='Logloss')
 a,b=matrices(X,raw,y,qty,fit,tune,'tune');cats=a.select_dtypes(include='object').columns.tolist();state('catboost_tune',nfeatures=a.shape[1]);m=cat(opts,'tune');m.fit(a,y[fit],cat_features=cats,eval_set=(b,y[tune]),early_stopping_rounds=80);p=m.predict_proba(b)[:,1];th=max([.475,.49,.5,.51,.525],key=lambda t:(accuracy_score(y[tune],p>=t),-abs(t-.5)));trees=m.tree_count_;save_npz(O/'tune.npz',rows=tune,p=p);lock={'trees':trees,'threshold':th,'config':CFG,'catboost_params':opts,'selection':'Tune early-stop logloss and tune accuracy threshold, before confirmation','nfeatures':a.shape[1]};(O/'selection_lock.json').write_text(json.dumps(lock,indent=2));del a,b,m;gc.collect();opts['iterations']=trees
 a,b=matrices(X,raw,y,qty,dev,conf,'confirm');state('catboost_confirm');m=cat(opts,'confirm');m.fit(a,y[dev],cat_features=cats);p=m.predict_proba(b)[:,1];seen=np.isin(sellers[conf],sellers[dev]);base=np.load(R/'results/new_models_20261021/seller_native_confirm.npz');assert np.array_equal(base['rows'],conf);save_npz(O/'confirm.npz',rows=conf,p=p,y=y[conf],seen=seen);m.save_model(str(O/'confirmation_model.cbm'));report={'lock':lock,'confirmation':metric(y[conf],p,seen,th),'native':metric(y[conf],base['p'],seen,.5),'scope':'Inherited row holdouts reused. All field teachers strictly OOF per training stage; no test/public labels. Combined teacher reused from parent. No confirmation weight search.'}
 h=R.parent/'recovered/mercadolibre_checkpoint_20261017/mercadolibre_recovery_20261017/prior_results/warranty/results/composition/holdout_predictions.csv';z=pd.DataFrame({'row':conf,'y':y[conf],'p':p}).merge(pd.read_csv(h),on='row');old=z.refit49k_without_public.to_numpy();blend=.85*old+.15*z.p.to_numpy();report['historical_common']={'n':len(z),'historical_accuracy_at0505':float(accuracy_score(z.y,old>=.505)),'candidate_accuracy':float(accuracy_score(z.y,z.p>=th)),'fixed_blend_accuracy_at0505':float(accuracy_score(z.y,blend>=.505))};(O/'report.json').write_text(json.dumps(report,indent=2));state('confirmation_complete',metrics=report['confirmation'])
 improve=report['confirmation']['accuracy']>report['native']['accuracy'] or report['historical_common']['fixed_blend_accuracy_at0505']>report['historical_common']['historical_accuracy_at0505']
 if improve and time.time()-START<1150 and (parent.O/'full_teacher.npz').exists():
  del a,b,m;gc.collect();tr=np.arange(70000);ev=np.arange(70000,100000);a,b=matrices(X,raw,y,qty,tr,ev,'full');state('catboost_full');m=cat(opts,'full');m.fit(a,y[:70000],cat_features=cats);p=m.predict_proba(b)[:,1];save_npz(O/'test.npz',p=p);m.save_model(str(O/'model.cbm'));report['full_refit']='complete'
 else:report['full_refit']='Skipped due budget/no improvement or combined full teacher not yet available; stage caches support resume'
 report['seconds']=time.time()-START;(O/'report.json').write_text(json.dumps(report,indent=2));state('done',full_refit=report['full_refit'])
if __name__=='__main__':main()
