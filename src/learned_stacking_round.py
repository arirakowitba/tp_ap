"""Learned probability stacking; every base learner is refitted in grouped OOF folds.

Unlike optimize_stack.py (a text score inserted into tree features), this trains a
LogisticRegression meta learner over several out-of-fold model predictions.
No precomputed target-encoded matrix may be reused between the inner folds.
"""
from pathlib import Path
import argparse, gc, json, os, time
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from lightgbm import LGBMClassifier
from competition_pipeline_v2 import groups_for, metrics, audit
from optimize_leaderboard import enrich, numeric_text, split
from optimize_stack import group_rates
ROOT=Path(__file__).resolve().parents[1]
BASE_NAMES=['text_logistic','text_svm','lgb_structured_rates']
LGB_PARAMS=dict(n_estimators=650,learning_rate=.03,num_leaves=63,min_child_samples=40,
 reg_lambda=5,colsample_bytree=.85,subsample=.9,subsample_freq=1,cat_smooth=20,
 max_cat_threshold=16,n_jobs=4,verbosity=-1,random_state=2026)

def check_deadline(deadline):
 if time.time()>=deadline:raise TimeoutError('Stacking deadline reached; keep finished artifacts only.')

def base_predictions(raw_train,y,groups,raw_eval,deadline):
 """All feature selection, vocabulary, SVD and encodings see raw_train only."""
 check_deadline(deadline)
 a,ids=enrich(raw_train);b,_=enrich(raw_eval,ids)
 word=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=80000,
  sublinear_tf=True,strip_accents='unicode',dtype=np.float32)
 char=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=3,
  max_features=80000,sublinear_tf=True,dtype=np.float32)
 cols=['seller_id','category_id','listing_type_id','shipping_mode','title_first']
 enc=OneHotEncoder(handle_unknown='ignore',min_frequency=3,dtype=np.float32)
 fit=sparse.hstack([word.fit_transform(a.full_text),char.fit_transform(a.title_text+' '+a.warranty_text),enc.fit_transform(a[cols].astype(str))],format='csr')
 val=sparse.hstack([word.transform(b.full_text),char.transform(b.title_text+' '+b.warranty_text),enc.transform(b[cols].astype(str))],format='csr')
 preds=[]
 for model in [LogisticRegression(C=4,solver='liblinear',max_iter=250),LinearSVC(C=.5,dual='auto',max_iter=3000,random_state=42)]:
  check_deadline(deadline);model.fit(fit,y)
  p=model.predict_proba(val)[:,1] if hasattr(model,'predict_proba') else expit(2*model.decision_function(val))
  preds.append(p)
 del fit,val,word,char,enc;gc.collect()
 check_deadline(deadline)
 both=pd.concat([a,b],ignore_index=True);t=np.arange(len(a));v=np.arange(len(a),len(both))
 tr,va,cats,_,_=numeric_text(both,t,v)
 # The rates themselves are grouped OOF inside this fold's training data.
 tr,va=group_rates(tr,va,y,groups)
 for c in cats:
  dtype=pd.CategoricalDtype(pd.Index(tr[c].astype(str).unique()))
  tr[c]=tr[c].astype(str).astype(dtype);va[c]=va[c].astype(str).astype(dtype)
 model=LGBMClassifier(**LGB_PARAMS);model.fit(tr,y);preds.append(model.predict_proba(va)[:,1])
 del a,b,both,tr,va,model;gc.collect()
 return np.column_stack(preds)

def probability_design(p,design):
 return p if design=='probability' else logit(np.clip(p,1e-5,1-1e-5))

def partition(raw,y,g,t,v,out,seed,deadline):
 """Produce exactly one honest OOF probability per training example."""
 path=out/f'stacking_seed{seed}_base.npz'
 if path.exists():
  cache=np.load(path)
  if not np.array_equal(cache['train_rows'],t) or not np.array_equal(cache['holdout_rows'],v):raise ValueError('Incompatible cached split')
  return cache['oof'],cache['holdout']
 oof=np.full((len(t),3),np.nan);fold_ids=np.full(len(t),-1)
 folds=StratifiedGroupKFold(3,shuffle=True,random_state=2026)
 for k,(it,iv) in enumerate(folds.split(np.zeros(len(t)),y[t],g[t])):
  check_deadline(deadline)
  assert not set(g[t[it]])&set(g[t[iv]])
  print('STACK OOF',seed,k,len(it),len(iv),flush=True)
  oof[iv]=base_predictions(raw.iloc[t[it]].reset_index(drop=True),y[t[it]],g[t[it]],raw.iloc[t[iv]].reset_index(drop=True),deadline)
  fold_ids[iv]=k
  pd.DataFrame({'row_index':t,'fold':fold_ids,**{name:oof[:,i] for i,name in enumerate(BASE_NAMES)}}).to_csv(out/f'stacking_seed{seed}_oof_partial.csv',index=False)
 assert np.isfinite(oof).all() and (fold_ids>=0).all()
 holdout=base_predictions(raw.iloc[t].reset_index(drop=True),y[t],g[t],raw.iloc[v].reset_index(drop=True),deadline)
 np.savez_compressed(path,oof=oof,holdout=holdout,train_rows=t,holdout_rows=v,fold_ids=fold_ids)
 pd.DataFrame({'row_index':v,'y_used':y[v],**{name:holdout[:,i] for i,name in enumerate(BASE_NAMES)}}).to_csv(out/f'stacking_seed{seed}_base_predictions.csv',index=False)
 pd.DataFrame({'row_index':t,'fold':fold_ids,'y_used':y[t],**{name:oof[:,i] for i,name in enumerate(BASE_NAMES)}}).to_csv(out/f'stacking_seed{seed}_oof.csv',index=False)
 (out/f'stacking_seed{seed}_oof_partial.csv').unlink(missing_ok=True)
 return oof,holdout

def meta_fit(oof,y,holdout,design,C):
 scale=StandardScaler();fit=scale.fit_transform(probability_design(oof,design));val=scale.transform(probability_design(holdout,design))
 model=LogisticRegression(C=C,solver='lbfgs',max_iter=500,random_state=2026)
 model.fit(fit,y)
 return model.predict_proba(val)[:,1],model,scale

def main(data_dir,output,deadline,confirm=True):
 out=Path(output);out.mkdir(parents=True,exist_ok=True);start=time.time()
 integrity=audit(Path(data_dir)/'train_data.jsonlines',70000)
 if not integrity['complete']:raise ValueError('Stacking requires all70000 valid labeled rows')
 manifest_path=out/'stacking_cache_manifest.json'
 manifest=dict(data_sha256=integrity['sha256'],base_names=BASE_NAMES,base_params=LGB_PARAMS,folds=3)
 if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:raise ValueError('Stacking cache belongs to different data or base configuration; use a new output folder')
 if not manifest_path.exists() and list(out.glob('stacking_seed*_base.npz')):raise ValueError('Unverifiable legacy stacking cache; use a new output folder')
 manifest_path.write_text(json.dumps(manifest,indent=2))
 raw=pd.read_json(Path(data_dir)/'train_data.jsonlines',lines=True)
 y=(raw.condition=='used').to_numpy(dtype=np.int8);g=groups_for(raw);t,v=split(y,g,42,2)
 with threadpool_limits(limits=4):
  oof,p=partition(raw,y,g,t,v,out,42,deadline)
  rows=[];best=None
  for i,name in enumerate(BASE_NAMES):
   q=float(max(np.arange(.46,.541,.005),key=lambda q:metrics(y[v],p[:,i],q)['accuracy']))
   rows.append(dict(name='stacking_base_'+name,protocol='selection_seed42_20pct',**metrics(y[v],p[:,i],q)))
  for design in ['probability','logit']:
   for C in [.01,.1,1,10]:
    pp,m,s=meta_fit(oof,y[t],p,design,C)
    q=float(max(np.arange(.46,.541,.005),key=lambda q:metrics(y[v],pp,q)['accuracy']))
    row=dict(name=f'stacking_{design}_C{C:g}',design=design,C=C,protocol='selection_seed42_20pct',**metrics(y[v],pp,q))
    rows.append(row)
    pd.DataFrame({'row_index':v,'y_used':y[v],'probability_used':pp}).to_csv(out/(row['name']+'_predictions.csv'),index=False)
    if best is None or row['accuracy']>best['accuracy']:best=row.copy()
  pd.DataFrame(rows).to_csv(out/'stacking_metrics.csv',index=False)
  best.update(base_names=BASE_NAMES,base_params=LGB_PARAMS,folds=3,training_rows=len(raw),data_sha256=integrity['sha256'],confirmation=None,status='frozen_pending_confirmation',runtime_seconds=time.time()-start,
   prior_overlap='Earlier optimize_stack.py inserted an OOF Logistic text score into tree features; it did not train a Logistic meta learner over multiple base models.')
  (out/'stacking_selected.json').write_text(json.dumps(best,indent=2))
  _,meta,scale=meta_fit(oof,y[t],p,best['design'],best['C'])
  (out/'stacking_coefficients.json').write_text(json.dumps(dict(base_names=BASE_NAMES,intercept=meta.intercept_.tolist(),standardized_coefficients=meta.coef_.tolist(),mean=scale.mean_.tolist(),scale=scale.scale_.tolist()),indent=2))
  print('STACK SELECTED',json.dumps(best),flush=True)
  if confirm:
   tt,vv=split(y,g,731,2);oo,pp=partition(raw,y,g,tt,vv,out,731,deadline)
   pred,_,_=meta_fit(oo,y[tt],pp,best['design'],best['C'])
   best['confirmation']=metrics(y[vv],pred,best['threshold']);best['status']='confirmed';best['runtime_seconds']=time.time()-start
   pd.DataFrame({'row_index':vv,'y_used':y[vv],'probability_used':pred}).to_csv(out/'stacking_confirmation_predictions.csv',index=False)
   (out/'stacking_selected.json').write_text(json.dumps(best,indent=2))
   print('STACK CONFIRMED',json.dumps(best),flush=True)
 return best

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--data-dir',default=os.getenv('AP2026Q2_DATA_DIR'));parser.add_argument('--output',default=str(ROOT/'results/round_v3'))
 parser.add_argument('--deadline',type=float,default=time.time()+90*60);parser.add_argument('--no-confirm',action='store_true');args=parser.parse_args()
 if not args.data_dir:parser.error('Set --data-dir')
 main(args.data_dir,args.output,args.deadline,not args.no_confirm)
