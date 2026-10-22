"""Seller context LightGBM. Train-only nested OOF target features; no public labels."""
from pathlib import Path
import sys,json,time,gc
import numpy as np,pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,log_loss
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from ap2026q2_common import build_features
O=R/'results/training_next/context';O.mkdir(parents=True,exist_ok=True)
def log(*x):print(*x,flush=True)
def metrics(y,p,seen):
 return dict(accuracy=float(accuracy_score(y,p>=.5)),log_loss=float(log_loss(y,p)),seen_accuracy=float(accuracy_score(y[seen],p[seen]>=.5)),new_accuracy=float(accuracy_score(y[~seen],p[~seen]>=.5)))
def keyed(seller,cat):return [seller.astype(str),seller.astype(str)+'|'+cat.astype(str)]
def target_features(keys,y,tr,va):
 prior=float(y[tr].mean()); blocks=[]
 for key in keys:
  st=pd.DataFrame({'k':key.iloc[tr].values,'y':y[tr]}).groupby('k').y.agg(['sum','count'])
  n=key.iloc[va].map(st['count']).fillna(0).to_numpy();summ=key.iloc[va].map(st['sum']).fillna(0).to_numpy()
  block=[np.log1p(n)]
  for a in [5,20]:
   p=(summ+a*prior)/(n+a);block.extend([p,n/(n+a),-(p*np.log(np.maximum(p,1e-9))+(1-p)*np.log(np.maximum(1-p,1e-9)))])
  blocks.append(np.array(block).T)
 return np.hstack(blocks).astype('float32')
def assemble(base,keys,y,tr,va):
 # OOF target encoding for every fit row; held-out map from fit only.
 z=np.empty((len(tr),14),dtype='float32')
 for inn,out in StratifiedKFold(5,shuffle=True,random_state=4601).split(tr,y[tr]):z[out]=target_features(keys,y,tr[inn],tr[out])
 zva=target_features(keys,y,tr,va)
 a=base.iloc[tr].copy();b=base.iloc[va].copy()
 for j in range(z.shape[1]):a['te_'+str(j)]=z[:,j];b['te_'+str(j)]=zva[:,j]
 for f in [a,b]:
  unit=f['qty_unit'].to_numpy()
  for j in [1,4,8,11]:f['te_unit_'+str(j)]=f['te_'+str(j)]*unit
 # Price context maps fitted on fit rows only, no targets used.
 for gcols in [['category_id','currency_id'],['seller_id']]:
  key=base[gcols].astype(str).agg('|'.join,axis=1);p=base['log_price']
  st=pd.DataFrame({'k':key.iloc[tr].values,'p':p.iloc[tr].values}).groupby('k').p.agg(['median','mean','std','count'])
  suffix='_'.join(gcols)
  for f,idx in [(a,tr),(b,va)]:
   med=key.iloc[idx].map(st['median']).fillna(p.iloc[tr].median()).to_numpy();sd=key.iloc[idx].map(st['std']).fillna(1).clip(lower=.25).to_numpy()
   f['relative_price_'+suffix]=p.iloc[idx].values-med
   f['price_z_'+suffix]=(p.iloc[idx].values-med)/sd
   f['price_context_n_'+suffix]=np.log1p(key.iloc[idx].map(st['count']).fillna(0).to_numpy())
 for c in a.select_dtypes(include=['string','object','category']).columns:
  cats=pd.Index(a[c].fillna('__NA__').astype(str).unique());a[c]=pd.Categorical(a[c].fillna('__NA__').astype(str),categories=cats);b[c]=pd.Categorical(b[c].fillna('__NA__').astype(str),categories=cats)
 return a,b

def main():
 t0=time.time();raw=pd.read_json(R.parent/'recovered/data/train_data.jsonlines',lines=True);test=pd.read_json(R.parent/'recovered/data/test_data.jsonlines',lines=True);y=raw.condition.eq('used').to_numpy(dtype='int8');allraw=pd.concat([raw,test],ignore_index=True);del raw,test;gc.collect()
 base=build_features(allraw).drop(columns='title_text');base['seller_id']=allraw.seller_id.astype(str)
 for c in ['initial_quantity','sold_quantity','available_quantity']:
  v=pd.to_numeric(allraw[c],errors='coerce').fillna(0);base['log_'+c]=np.log1p(v.clip(lower=0)).astype('float32')
 base['qty_unit']=allraw.initial_quantity.eq(1).astype('int8');base['sold_initial_ratio']=(allraw.sold_quantity/(allraw.initial_quantity.clip(lower=1))).astype('float32');base['available_initial_ratio']=(allraw.available_quantity/allraw.initial_quantity.clip(lower=1)).astype('float32')
 for c in ['official_store_id','status','international_delivery_mode']:base[c]=allraw[c].fillna('__NA__').astype(str)
 keys=keyed(allraw.seller_id,allraw.category_id);seller=allraw.seller_id.to_numpy();del allraw;gc.collect()
 parts=np.load(R/'results/new_models_20261021/partitions.npz');fit=parts['fit'];tune=parts['tune'];conf=parts['confirm'];dev=np.concatenate([fit,tune]);nt=np.load(R/'results/new_models_20261021/seller_native_tune.npz') if (R/'results/new_models_20261021/seller_native_tune.npz').exists() else None
 # tune vector might be absent; native confirmation has never trained on confirm.
 native_t=nt['p'] if nt is not None else None
 seen_t=np.isin(seller[tune],seller[fit]);seen_c=np.isin(seller[conf],seller[dev]);a,b=assemble(base,keys,y,fit,tune);log('FEATURES',a.shape,b.shape)
 specs=[dict(num_leaves=31,min_child_samples=60,colsample_bytree=.85,reg_lambda=10),dict(num_leaves=63,min_child_samples=100,colsample_bytree=.8,reg_lambda=25)]
 reports=[];best=None
 for i,s in enumerate(specs):
  m=lgb.LGBMClassifier(n_estimators=1600,learning_rate=.035,n_jobs=2,verbosity=-1,random_state=4601,**s);m.fit(a,y[fit],eval_set=[(b,y[tune])],callbacks=[lgb.early_stopping(100,verbose=False)])
  p=m.predict_proba(b)[:,1];np.savez_compressed(O/f'tune_{i}.npz',rows=tune,p=p,y=y[tune],seen=seen_t);m.booster_.save_model(str(O/f'tune_{i}.txt'));rec={'config':i,'params':s,'iterations':m.best_iteration_,'standalone':metrics(y[tune],p,seen_t)}
  if native_t is not None:rec['blend85native']=metrics(y[tune],.85*native_t+.15*p,seen_t)
  score=max(rec['standalone']['accuracy'],rec.get('blend85native',{}).get('accuracy',0));reports.append(rec);(O/'tune_reports.json').write_text(json.dumps(reports,indent=2));log('TUNE',rec)
  if best is None or score>best[0]:best=(score,i,m.best_iteration_)
 del a,b,m;gc.collect();_,i,it=best
 policy='standalone' if native_t is None or reports[i]['standalone']['accuracy']>=reports[i]['blend85native']['accuracy'] else 'blend85native'
 locked={'config':i,'params':specs[i],'iterations':it,'policy':policy,'selection':'Maximum of standalone/context and fixed .85native+.15context on tune only; ties choose standalone. Confirmation inspected once.'};(O/'locked_before_confirmation.json').write_text(json.dumps(locked,indent=2))
 a,b=assemble(base,keys,y,dev,conf);m=lgb.LGBMClassifier(n_estimators=it,learning_rate=.035,n_jobs=2,verbosity=-1,random_state=4601,**specs[i]);m.fit(a,y[dev]);p=m.predict_proba(b)[:,1];m.booster_.save_model(str(O/'confirm_model.txt'));np.savez_compressed(O/'confirm.npz',rows=conf,p=p,y=y[conf],seen=seen_c)
 nc=np.load(R/'results/new_models_20261021/seller_native_confirm.npz')['p'];pc=p if policy=='standalone' else .85*nc+.15*p
 result={'locked':locked,'confirmation_context':metrics(y[conf],p,seen_c),'confirmation_blend85native':metrics(y[conf],.85*nc+.15*p,seen_c),'confirmation_selected':metrics(y[conf],pc,seen_c),'native_reference':metrics(y[conf],nc,seen_c),'context_fixes_native':int((((nc>=.5)!=y[conf])&((p>=.5)==y[conf])).sum()),'context_breaks_native':int((((nc>=.5)==y[conf])&((p>=.5)!=y[conf])).sum()),'elapsed_seconds':time.time()-t0,'public_labels_used':False}
 (O/'confirmation_report.json').write_text(json.dumps(result,indent=2));log('CONFIRM',result)
 del a,b,m;gc.collect()
 # Diagnostic gate predeclared by task: refit if context or fixed blend beats native.
 if max(result['confirmation_context']['accuracy'],result['confirmation_blend85native']['accuracy'])>result['native_reference']['accuracy']:
  a,b=assemble(base,keys,y,np.arange(70000),np.arange(70000,100000));m=lgb.LGBMClassifier(n_estimators=it,learning_rate=.035,n_jobs=2,verbosity=-1,random_state=4601,**specs[i]);m.fit(a,y);pt=m.predict_proba(b)[:,1];m.booster_.save_model(str(O/'full_model.txt'));np.savez_compressed(O/'test.npz',p=pt);result['full_refit']='complete'
 else:result['full_refit']='skipped: no standalone or fixed blend gain against native'
 result['elapsed_seconds']=time.time()-t0;(O/'confirmation_report.json').write_text(json.dumps(result,indent=2));log('DONE',result)
if __name__=='__main__':main()
