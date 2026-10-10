"""Add an independent multilingual-embedding Logistic base to honest four-base OOF.

Embedding model is frozen and unsupervised. StandardScaler and LogisticRegression
are fitted separately within each existing grouped fold, not across holdouts.
"""
from pathlib import Path
import argparse,json,os,time
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import learned_stacking_round as base3
import learned_stacking4_round as base4
from competition_pipeline_v2 import groups_for,metrics,audit
from optimize_leaderboard import split
ROOT=Path(__file__).resolve().parents[1]
BASE_NAMES=base4.BASE_NAMES+['embedding_logistic']
EMBEDDING_PARAMS=dict(C=1,solver='lbfgs',max_iter=500,random_state=2026)

def embedding_prediction(a,y,b):
 scaler=StandardScaler();fit=scaler.fit_transform(a);val=scaler.transform(b)
 model=LogisticRegression(**EMBEDDING_PARAMS);model.fit(fit,y)
 return model.predict_proba(val)[:,1]

def base_predictions(raw_train,y,groups,raw_eval,deadline,train_embeddings=None,eval_embeddings=None):
 if train_embeddings is None or eval_embeddings is None:raise ValueError('Provide frozen multilingual embeddings in exact raw row order')
 return np.column_stack([base4.base_predictions(raw_train,y,groups,raw_eval,deadline),embedding_prediction(train_embeddings,y,eval_embeddings)])

def partition(raw,y,g,t,v,out,seed,deadline):
 out=Path(out);path=out/f'stacking5_seed{seed}_base.npz'
 if path.exists():
  cache=np.load(path)
  if not np.array_equal(cache['train_rows'],t) or not np.array_equal(cache['holdout_rows'],v):raise ValueError('Incompatible cached split')
  return cache['oof'],cache['holdout']
 source=out/f'stacking4_seed{seed}_base.npz'
 if not source.exists():raise ValueError('Four-base grouped OOF cache must be completed first')
 original=np.load(source)
 assert np.array_equal(original['train_rows'],t) and np.array_equal(original['holdout_rows'],v)
 embeddings=np.load(out/'embedding_titles.npy',mmap_mode='r')
 assert embeddings.shape==(len(y),384)
 oof_emb=np.full(len(t),np.nan)
 for k in range(3):
  base3.check_deadline(deadline);it=np.flatnonzero(original['fold_ids']!=k);iv=np.flatnonzero(original['fold_ids']==k)
  assert not set(g[t[it]])&set(g[t[iv]])
  print('STACK5 EMBEDDING OOF',seed,k,flush=True)
  oof_emb[iv]=embedding_prediction(embeddings[t[it]],y[t[it]],embeddings[t[iv]])
 assert np.isfinite(oof_emb).all()
 hold_emb=embedding_prediction(embeddings[t],y[t],embeddings[v])
 oof=np.column_stack([original['oof'],oof_emb]);holdout=np.column_stack([original['holdout'],hold_emb])
 np.savez_compressed(path,oof=oof,holdout=holdout,train_rows=t,holdout_rows=v,fold_ids=original['fold_ids'])
 pd.DataFrame({'row_index':t,'fold':original['fold_ids'],'y_used':y[t],**{name:oof[:,i] for i,name in enumerate(BASE_NAMES)}}).to_csv(out/f'stacking5_seed{seed}_oof.csv',index=False)
 pd.DataFrame({'row_index':v,'y_used':y[v],**{name:holdout[:,i] for i,name in enumerate(BASE_NAMES)}}).to_csv(out/f'stacking5_seed{seed}_base_predictions.csv',index=False)
 return oof,holdout

meta_fit=base3.meta_fit

def main(data_dir,output,deadline,confirm=True):
 out=Path(output);start=time.time();integrity=audit(Path(data_dir)/'train_data.jsonlines',70000)
 if not integrity['complete']:raise ValueError('Requires70000 valid rows')
 embedding_metadata=json.loads((out/'embedding_metadata.json').read_text())
 if embedding_metadata['train_sha256']!=integrity['sha256']:raise ValueError('Embedding row cache is from another dataset')
 if json.loads((out/'stacking4_cache_manifest.json').read_text())['data_sha256']!=integrity['sha256']:raise ValueError('Four-base source cache is from another dataset')
 manifest_path=out/'stacking5_cache_manifest.json'
 manifest=dict(data_sha256=integrity['sha256'],base_names=BASE_NAMES,base_params=base3.LGB_PARAMS,cat_params=base4.CAT_PARAMS,embedding_params=EMBEDDING_PARAMS,folds=3,
  embedding_weight_hashes=json.loads((out/'embedding_weight_hashes.json').read_text()))
 if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:raise ValueError('Incompatible five-base cache')
 if not manifest_path.exists() and list(out.glob('stacking5_seed*_base.npz')):raise ValueError('Unverifiable old five-base cache')
 manifest_path.write_text(json.dumps(manifest,indent=2))
 # Only retain the labels/family IDs: no need to duplicate all nested raw data.
 rows=[]
 with (Path(data_dir)/'train_data.jsonlines').open() as f:
  for line in f:
   r=json.loads(line);rows.append({c:r.get(c) for c in ['id','parent_item_id','condition']})
 raw=pd.DataFrame(rows);y=(raw.condition=='used').to_numpy(dtype=np.int8);g=groups_for(raw);t,v=split(y,g,42,2)
 with threadpool_limits(limits=2):
  oof,p=partition(raw,y,g,t,v,out,42,deadline);rows=[];best=None
  q=float(max(np.arange(.46,.541,.005),key=lambda q:metrics(y[v],p[:,-1],q)['accuracy']))
  rows.append(dict(name='stacking5_base_embedding_logistic',protocol='selection_seed42_20pct',**metrics(y[v],p[:,-1],q)))
  for design in ['probability','logit']:
   for C in [.01,.1,1,10]:
    pp,m,s=meta_fit(oof,y[t],p,design,C)
    q=float(max(np.arange(.46,.541,.005),key=lambda q:metrics(y[v],pp,q)['accuracy']))
    row=dict(name=f'stacking5_{design}_C{C:g}',design=design,C=C,protocol='selection_seed42_20pct',**metrics(y[v],pp,q))
    rows.append(row)
    pd.DataFrame({'row_index':v,'y_used':y[v],'probability_used':pp}).to_csv(out/(row['name']+'_predictions.csv'),index=False)
    if best is None or row['accuracy']>best['accuracy']:best=row.copy()
  pd.DataFrame(rows).to_csv(out/'stacking5_metrics.csv',index=False)
  best.update(base_names=BASE_NAMES,base_params=base3.LGB_PARAMS,cat_params=base4.CAT_PARAMS,embedding_params=EMBEDDING_PARAMS,folds=3,training_rows=len(raw),data_sha256=integrity['sha256'],confirmation=None,status='frozen_pending_confirmation',runtime_seconds=time.time()-start)
  (out/'stacking5_selected.json').write_text(json.dumps(best,indent=2))
  _,meta,scale=meta_fit(oof,y[t],p,best['design'],best['C'])
  (out/'stacking5_coefficients.json').write_text(json.dumps(dict(base_names=BASE_NAMES,intercept=meta.intercept_.tolist(),standardized_coefficients=meta.coef_.tolist(),mean=scale.mean_.tolist(),scale=scale.scale_.tolist()),indent=2))
  print('STACK5 SELECTED',json.dumps(best),flush=True)
  if confirm:
   tt,vv=split(y,g,731,2);oo,pp=partition(raw,y,g,tt,vv,out,731,deadline)
   pred,_,_=meta_fit(oo,y[tt],pp,best['design'],best['C']);best['confirmation']=metrics(y[vv],pred,best['threshold']);best['status']='confirmed';best['runtime_seconds']=time.time()-start
   pd.DataFrame({'row_index':vv,'y_used':y[vv],'probability_used':pred}).to_csv(out/'stacking5_confirmation_predictions.csv',index=False)
   (out/'stacking5_selected.json').write_text(json.dumps(best,indent=2));print('STACK5 CONFIRMED',json.dumps(best),flush=True)
 return best

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data-dir',default=os.getenv('AP2026Q2_DATA_DIR'));p.add_argument('--output',default=str(ROOT/'results/round_v3'));p.add_argument('--deadline',type=float,default=time.time()+90*60);p.add_argument('--no-confirm',action='store_true');a=p.parse_args()
 if not a.data_dir:p.error('Set --data-dir')
 main(a.data_dir,a.output,a.deadline,not a.no_confirm)
