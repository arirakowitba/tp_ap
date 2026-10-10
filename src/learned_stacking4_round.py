"""Four-base Logistic OOF stacking: extend the independent three-base experiment.

Reuses only its honest OOF probabilities and adds a separately refitted CatBoost
inside exactly the same grouped folds. Does not overwrite three-base artifacts.
"""
from pathlib import Path
import argparse,gc,json,os,time
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
import learned_stacking_round as base3
from competition_pipeline_v2 import groups_for,metrics,audit
from optimize_leaderboard import enrich,numeric_text,split
from optimize_stack import group_rates
ROOT=Path(__file__).resolve().parents[1]
BASE_NAMES=base3.BASE_NAMES+['cat_structured_rates']
CAT_PARAMS=dict(iterations=800,depth=7,learning_rate=.035,l2_leaf_reg=10,
 random_seed=51,loss_function='Logloss',thread_count=2,allow_writing_files=False,
 one_hot_max_size=10,random_strength=.5,verbose=False)

def cat_predictions(raw_train,y,groups,raw_eval,deadline):
 base3.check_deadline(deadline)
 a,ids=enrich(raw_train);b,_=enrich(raw_eval,ids)
 both=pd.concat([a,b],ignore_index=True);t=np.arange(len(a));v=np.arange(len(a),len(both))
 tr,va,cats,_,_=numeric_text(both,t,v)
 tr,va=group_rates(tr,va,y,groups)
 for c in cats:tr[c]=tr[c].astype(str);va[c]=va[c].astype(str)
 for c in tr.columns.difference(cats):tr[c]=tr[c].fillna(-999);va[c]=va[c].fillna(-999)
 model=CatBoostClassifier(**CAT_PARAMS);model.fit(tr,y,cat_features=cats)
 pred=model.predict_proba(va)[:,1]
 del a,b,both,tr,va,model;gc.collect()
 return pred

def base_predictions(raw_train,y,groups,raw_eval,deadline):
 return np.column_stack([base3.base_predictions(raw_train,y,groups,raw_eval,deadline),cat_predictions(raw_train,y,groups,raw_eval,deadline)])

def partition(raw,y,g,t,v,out,seed,deadline):
 out=Path(out);path=out/f'stacking4_seed{seed}_base.npz'
 if path.exists():
  cache=np.load(path)
  if not np.array_equal(cache['train_rows'],t) or not np.array_equal(cache['holdout_rows'],v):raise ValueError('Incompatible cached split')
  return cache['oof'],cache['holdout']
 original=np.load(out/f'stacking_seed{seed}_base.npz')
 assert np.array_equal(original['train_rows'],t) and np.array_equal(original['holdout_rows'],v)
 oof_cat=np.full(len(t),np.nan)
 folds=StratifiedGroupKFold(3,shuffle=True,random_state=2026)
 for k,(it,iv) in enumerate(folds.split(np.zeros(len(t)),y[t],g[t])):
  base3.check_deadline(deadline);assert not set(g[t[it]])&set(g[t[iv]])
  assert np.all(original['fold_ids'][iv]==k)
  print('STACK4 CAT OOF',seed,k,flush=True)
  oof_cat[iv]=cat_predictions(raw.iloc[t[it]].reset_index(drop=True),y[t[it]],g[t[it]],raw.iloc[t[iv]].reset_index(drop=True),deadline)
 assert np.isfinite(oof_cat).all()
 hold_cat=cat_predictions(raw.iloc[t].reset_index(drop=True),y[t],g[t],raw.iloc[v].reset_index(drop=True),deadline)
 oof=np.column_stack([original['oof'],oof_cat]);holdout=np.column_stack([original['holdout'],hold_cat])
 np.savez_compressed(path,oof=oof,holdout=holdout,train_rows=t,holdout_rows=v,fold_ids=original['fold_ids'])
 pd.DataFrame({'row_index':t,'fold':original['fold_ids'],'y_used':y[t],**{name:oof[:,i] for i,name in enumerate(BASE_NAMES)}}).to_csv(out/f'stacking4_seed{seed}_oof.csv',index=False)
 pd.DataFrame({'row_index':v,'y_used':y[v],**{name:holdout[:,i] for i,name in enumerate(BASE_NAMES)}}).to_csv(out/f'stacking4_seed{seed}_base_predictions.csv',index=False)
 return oof,holdout

meta_fit=base3.meta_fit

def main(data_dir,output,deadline,confirm=True):
 out=Path(output);out.mkdir(parents=True,exist_ok=True);start=time.time()
 integrity=audit(Path(data_dir)/'train_data.jsonlines',70000)
 if not integrity['complete']:raise ValueError('Requires all70000 valid rows')
 if json.loads((out/'stacking_cache_manifest.json').read_text())['data_sha256']!=integrity['sha256']:raise ValueError('Three-base source cache is from another dataset')
 manifest_path=out/'stacking4_cache_manifest.json'
 manifest=dict(data_sha256=integrity['sha256'],base_names=BASE_NAMES,base_params=base3.LGB_PARAMS,cat_params=CAT_PARAMS,folds=3)
 if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:raise ValueError('Incompatible four-base cache')
 if not manifest_path.exists() and list(out.glob('stacking4_seed*_base.npz')):raise ValueError('Unverifiable old four-base cache')
 manifest_path.write_text(json.dumps(manifest,indent=2))
 raw=pd.read_json(Path(data_dir)/'train_data.jsonlines',lines=True)
 y=(raw.condition=='used').to_numpy(dtype=np.int8);g=groups_for(raw);t,v=split(y,g,42,2)
 with threadpool_limits(limits=2):
  oof,p=partition(raw,y,g,t,v,out,42,deadline);rows=[];best=None
  q=float(max(np.arange(.46,.541,.005),key=lambda q:metrics(y[v],p[:,-1],q)['accuracy']))
  rows.append(dict(name='stacking4_base_cat',protocol='selection_seed42_20pct',**metrics(y[v],p[:,-1],q)))
  for design in ['probability','logit']:
   for C in [.01,.1,1,10]:
    pp,m,s=meta_fit(oof,y[t],p,design,C)
    q=float(max(np.arange(.46,.541,.005),key=lambda q:metrics(y[v],pp,q)['accuracy']))
    row=dict(name=f'stacking4_{design}_C{C:g}',design=design,C=C,protocol='selection_seed42_20pct',**metrics(y[v],pp,q))
    rows.append(row)
    pd.DataFrame({'row_index':v,'y_used':y[v],'probability_used':pp}).to_csv(out/(row['name']+'_predictions.csv'),index=False)
    if best is None or row['accuracy']>best['accuracy']:best=row.copy()
  pd.DataFrame(rows).to_csv(out/'stacking4_metrics.csv',index=False)
  best.update(base_names=BASE_NAMES,base_params=base3.LGB_PARAMS,cat_params=CAT_PARAMS,folds=3,training_rows=len(raw),data_sha256=integrity['sha256'],confirmation=None,status='frozen_pending_confirmation',runtime_seconds=time.time()-start)
  (out/'stacking4_selected.json').write_text(json.dumps(best,indent=2))
  _,meta,scale=meta_fit(oof,y[t],p,best['design'],best['C'])
  (out/'stacking4_coefficients.json').write_text(json.dumps(dict(base_names=BASE_NAMES,intercept=meta.intercept_.tolist(),standardized_coefficients=meta.coef_.tolist(),mean=scale.mean_.tolist(),scale=scale.scale_.tolist()),indent=2))
  print('STACK4 SELECTED',json.dumps(best),flush=True)
  if confirm:
   tt,vv=split(y,g,731,2);oo,pp=partition(raw,y,g,tt,vv,out,731,deadline)
   pred,_,_=meta_fit(oo,y[tt],pp,best['design'],best['C'])
   best['confirmation']=metrics(y[vv],pred,best['threshold']);best['status']='confirmed';best['runtime_seconds']=time.time()-start
   pd.DataFrame({'row_index':vv,'y_used':y[vv],'probability_used':pred}).to_csv(out/'stacking4_confirmation_predictions.csv',index=False)
   (out/'stacking4_selected.json').write_text(json.dumps(best,indent=2));print('STACK4 CONFIRMED',json.dumps(best),flush=True)
 return best

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data-dir',default=os.getenv('AP2026Q2_DATA_DIR'));p.add_argument('--output',default=str(ROOT/'results/round_v3'));p.add_argument('--deadline',type=float,default=time.time()+90*60);p.add_argument('--no-confirm',action='store_true');a=p.parse_args()
 if not a.data_dir:p.error('Set --data-dir')
 main(a.data_dir,a.output,a.deadline,not a.no_confirm)
