"""Deep tabular CatBoost + strictly OOF NB teacher. No public/test labels."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,time,gc,joblib
import numpy as np,pandas as pd
from scipy.sparse import hstack
from scipy.special import logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,log_loss
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
threadpool_limits(1)
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from ap2026q2_common import build_features
from ap2026q2_deep_features import build_deep_features
O=R/'results/training_round2/deep_teacher';O.mkdir(parents=True,exist_ok=True);START=time.time()
def say(*x):print(*x,flush=True)
def state(stage,**kw):
 x={'stage':stage,'seconds':time.time()-START,**kw};(O/'progress.json').write_text(json.dumps(x,indent=2));say(x)
def load():
 if (O/'features.pkl').exists():
  a=np.load(O/'input.npz',allow_pickle=True);return pd.read_pickle(O/'features.pkl'),a['docs'],a['y'],a['sellers'],a['qty']
 frames=[];docs=[];ys=[];sellers=[];qty=[]
 for file in ['train_data.jsonlines','test_data.jsonlines']:
  chunk=[]
  for line in open(R.parent/'recovered/data'/file):
   x=json.loads(line);ys.append(int(x.get('condition')=='used'));x.pop('condition',None)
   x['attributes']=[a for a in (x.get('attributes') or []) if isinstance(a,dict) and 'condition' not in (str(a.get('id',''))+' '+str(a.get('name',''))).lower()]
   title=str(x.get('title') or '');w=str(x.get('warranty') or '');a=' '.join(str(v.get('name',''))+' '+str(v.get('value_name','')) for v in x['attributes']);docs.append((title+' '+title+' warranty '+w+' attributes '+a+' category_'+str(x.get('category_id',''))).lower());sellers.append(str(x['seller_id']));qty.append(float(x.get('initial_quantity') or 0));chunk.append(x)
   if len(chunk)>=5000:frames.append(chunk_features(chunk));chunk=[]
  if chunk:frames.append(chunk_features(chunk))
 X=pd.concat(frames,ignore_index=True);X.to_pickle(O/'features.pkl');docs=np.array(docs,dtype=object);ys=np.array(ys,dtype=np.int8);sellers=np.array(sellers);qty=np.array(qty);np.savez_compressed(O/'input.npz',docs=docs,y=ys,sellers=sellers,qty=qty);state('features_ready',shape=list(X.shape));return X,docs,ys,sellers,qty
def chunk_features(chunk):
 raw=pd.DataFrame(chunk)
 for c in ['date_created','start_time','last_updated','stop_time']:
  if c not in raw:raw[c]=None
 X=build_deep_features(raw,build_features(raw)).drop(columns='title_text');X['seller_id']=raw.seller_id.astype(str)
 for c in X:
  if not pd.api.types.is_numeric_dtype(X[c]):X[c]=X[c].astype('string').fillna('__missing__').astype(str)
  else:X[c]=X[c].astype(float).replace([np.inf,-np.inf],np.nan).fillna(-1)
 return X

def teacher_fit(docs,y,tr,queries,model_path=None):
 vs=[TfidfVectorizer(ngram_range=(1,2),max_features=30000,min_df=2,sublinear_tf=True,dtype=np.float32,strip_accents='unicode'),TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=30000,min_df=3,sublinear_tf=True,dtype=np.float32,strip_accents='unicode')]
 a=[];b=[[] for q in queries]
 for v in vs:
  a.append(v.fit_transform(docs[tr]))
  for j,q in enumerate(queries):b[j].append(v.transform(docs[q]))
 A=hstack(a,format='csr');plus=np.asarray(A[y[tr]==1].sum(axis=0)).ravel()+1;minus=np.asarray(A[y[tr]==0].sum(axis=0)).ravel()+1;r=np.log((plus/plus.sum())/(minus/minus.sum())).astype(np.float32)
 m=LogisticRegression(C=4,solver='liblinear',max_iter=250).fit(A.multiply(r).tocsr(),y[tr]);p=[m.predict_proba(hstack(q,format='csr').multiply(r).tocsr())[:,1] for q in b]
 if model_path is not None:
  tmp=model_path.with_suffix('.tmp');joblib.dump({'vectorizers':vs,'ratio':r,'model':m,'training_rows':tr},tmp,compress=3);joblib.load(tmp);os.replace(tmp,model_path)
 return p

def teacher_stage(docs,y,tr,ev,name):
 f=O/(name+'_teacher.npz')
 if f.exists():
  z=np.load(f);assert np.array_equal(z['train_rows'],tr) and np.array_equal(z['eval_rows'],ev);return z['oof'],z['p']
 oof=np.full(len(tr),np.nan)
 for i,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261023).split(tr,y[tr])):
  path=O/(name+'_fold'+str(i)+'.npz')
  if path.exists():
   z=np.load(path);assert np.array_equal(z['rows'],tr[b]);p=z['p']
  else:
   state(name+'_teacher_fold',fold=i);p=teacher_fit(docs,y,tr[a],[tr[b]])[0];np.savez_compressed(path,rows=tr[b],p=p)
  oof[b]=p;gc.collect()
 state(name+'_teacher_full');p=teacher_fit(docs,y,tr,[ev],model_path=O/(name+'_teacher_model.joblib'))[0];assert np.isfinite(oof).all();np.savez_compressed(f,train_rows=tr,eval_rows=ev,oof=oof,p=p);return oof,p

def attach(X,tr,ev,poof,pe,qty):
 A=X.iloc[tr].copy();B=X.iloc[ev].copy()
 # Label-free context, fit scoped.
 for key in ['category_id','seller_id']:
  g=A.groupby(key)['log_price'].agg(['mean','std','size'])
  for Z in [A,B]:
   for stat in ['mean','std','size']:Z[key+'_price_'+stat]=Z[key].map(g[stat]).fillna(-1).astype(float)
   Z[key+'_relative_price']=Z['log_price']-Z[key+'_price_mean']
 for Z,p,rows in [(A,poof,tr),(B,pe,ev)]:
  Z['teacher_nb']=p;Z['teacher_nb_logit']=logit(np.clip(p,1e-5,1-1e-5));Z['teacher_nb_logit_qty1']=Z.teacher_nb_logit.to_numpy()*(qty[rows]<=1);Z['teacher_nb_logit_free']=Z.teacher_nb_logit.to_numpy()*(Z.listing_type_id.to_numpy()=='free')
 return A,B

def metrics(y,p,seen,threshold):
 return {'accuracy':float(accuracy_score(y,p>=threshold)),'log_loss':float(log_loss(y,p)),'seen_accuracy':float(accuracy_score(y[seen],p[seen]>=threshold)),'new_accuracy':float(accuracy_score(y[~seen],p[~seen]>=threshold))}
def main():
 X,docs,y,sellers,qty=load();parts=np.load(R/'results/new_models_20261021/partitions.npz');fit,tune,confirm=[parts[k] for k in ['fit','tune','confirm']];dev=np.r_[fit,tune]
 po,p=teacher_stage(docs,y,fit,tune,'tune');A,B=attach(X,fit,tune,po,p,qty);cats=A.select_dtypes(include='object').columns.tolist();opts=dict(depth=7,iterations=750,learning_rate=.06,l2_leaf_reg=15,random_strength=.8,rsm=.8,border_count=64,thread_count=2,random_seed=20261023,verbose=200,allow_writing_files=True,loss_function='Logloss',save_snapshot=True,snapshot_interval=30)
 state('catboost_tune',nfeatures=A.shape[1]);m=CatBoostClassifier(**opts,train_dir=str(O/'snap_tune'),snapshot_file=str((O/'snapshot_tune.bin').resolve()));m.fit(A,y[fit],cat_features=cats,eval_set=(B,y[tune]),early_stopping_rounds=80);p=m.predict_proba(B)[:,1];threshold=max([.475,.49,.5,.51,.525],key=lambda t:(accuracy_score(y[tune],p>=t),-abs(t-.5)));trees=m.tree_count_;np.savez_compressed(O/'tune.npz',rows=tune,p=p);lock={'trees':trees,'threshold':threshold,'teacher':'3fold OOF NB, vocab30kword+30kchar, C4fixed','catboost_params':opts,'selection':'Trees early-stop logloss tune; threshold tune accuracy; no confirmation selection.'};(O/'selection_lock.json').write_text(json.dumps(lock,indent=2));del A,B,m;gc.collect()
 po,p=teacher_stage(docs,y,dev,confirm,'confirm');A,B=attach(X,dev,confirm,po,p,qty);opts['iterations']=trees;state('catboost_confirm');m=CatBoostClassifier(**opts,train_dir=str(O/'snap_confirm'),snapshot_file=str((O/'snapshot_confirm.bin').resolve()));m.fit(A,y[dev],cat_features=cats);p=m.predict_proba(B)[:,1];seen=np.isin(sellers[confirm],sellers[dev]);base=np.load(R/'results/new_models_20261021/seller_native_confirm.npz');assert np.array_equal(base['rows'],confirm);report={'lock':lock,'confirmation':metrics(y[confirm],p,seen,threshold),'native':metrics(y[confirm],base['p'],seen,.5),'scope':'Reused original row partitions, teacher strictly OOF inside each fit; no public/test labels. Not a virgin holdout.','seconds':time.time()-START};np.savez_compressed(O/'confirm.npz',rows=confirm,p=p,y=y[confirm],seen=seen);(O/'report.json').write_text(json.dumps(report,indent=2));m.save_model(str(O/'confirmation_model.cbm'));state('confirmation_complete',metrics=report['confirmation'])
 h=R.parent/'recovered/mercadolibre_checkpoint_20261017/mercadolibre_recovery_20261017/prior_results/warranty/results/composition/holdout_predictions.csv';z=pd.DataFrame({'row':confirm,'y':y[confirm],'candidate':p}).merge(pd.read_csv(h),on='row');old=z.refit49k_without_public.to_numpy();blend=.85*old+.15*z.candidate.to_numpy();report['historical_common_diagnostic']={'n':len(z),'historical_accuracy_at0505':float(accuracy_score(z.y,old>=.505)),'candidate_accuracy':float(accuracy_score(z.y,z.candidate>=threshold)),'fixed_blend_accuracy_at0505':float(accuracy_score(z.y,blend>=.505)),'scope':'Fixed .85historical/.15candidate diagnostic; no weight selection.'};(O/'report.json').write_text(json.dumps(report,indent=2))
 improve=report['confirmation']['accuracy']>report['native']['accuracy'] or report['historical_common_diagnostic']['fixed_blend_accuracy_at0505']>report['historical_common_diagnostic']['historical_accuracy_at0505']
 if improve and time.time()-START<1200:
  del A,B,m;gc.collect();tr=np.arange(70000);ev=np.arange(70000,100000);po,p=teacher_stage(docs,y,tr,ev,'full');A,B=attach(X,tr,ev,po,p,qty);state('catboost_full');m=CatBoostClassifier(**opts,train_dir=str(O/'snap_full'),snapshot_file=str((O/'snapshot_full.bin').resolve()));m.fit(A,y[:70000],cat_features=cats);p=m.predict_proba(B)[:,1];np.savez_compressed(O/'test.npz',p=p);m.save_model(str(O/'model.cbm'));report['full_refit']='complete'
 else:report['full_refit']='Skipped: no improvement or 20min elapsed before expensive full-stage'
 report['seconds']=time.time()-START;(O/'report.json').write_text(json.dumps(report,indent=2));state('done',full_refit=report['full_refit'])
if __name__=='__main__':main()
