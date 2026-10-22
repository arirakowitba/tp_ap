"""Sparse text+structured LightGBM; train-only transforms, no public labels."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,time,gc,sys
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix,hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score,log_loss
import lightgbm as lgb
from threadpoolctl import threadpool_limits
threadpool_limits(2)
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'));sys.path.insert(0,str(R/'scripts'))
from ap2026q2_common import build_features
from train_seller_context_next import assemble,keyed
O=R/'results/training_round2/sparse';O.mkdir(parents=True,exist_ok=True)
def log(*x):print(*x,flush=True)
def metrics(y,p,seen):return dict(accuracy=float(accuracy_score(y,p>=.5)),log_loss=float(log_loss(y,p)),seen_accuracy=float(accuracy_score(y[seen],p[seen]>=.5)),new_accuracy=float(accuracy_score(y[~seen],p[~seen]>=.5)))
def read():
 records=[];docs=[];target=[]
 fields=['title','category_id','listing_type_id','buying_mode','currency_id','accepts_mercadopago','automatic_relist','price','base_price','initial_quantity','available_quantity','sold_quantity','seller_id','official_store_id','status','international_delivery_mode','warranty','parent_item_id','original_price','video_id','shipping','seller_address','date_created']
 for file in ['train_data.jsonlines','test_data.jsonlines']:
  for line in open(R.parent/'recovered/data'/file):
   x=json.loads(line);a=[v for v in x.get('attributes',[]) if isinstance(v,dict) and 'CONDITION' not in str(v.get('id','')).upper() and 'CONDITION' not in str(v.get('name','')).upper()];t=str(x.get('title') or '');w=str(x.get('warranty') or '');at=' '.join(str(v.get('name',''))+' '+str(v.get('value_name','')) for v in a);docs.append((t+' '+t+' warranty '+w+' attributes '+at).lower());row={k:x.get(k) for k in fields}
   for k in ['pictures','attributes','variations','tags','descriptions','non_mercado_pago_payment_methods']:row[k]=[None]*len(x.get(k) or [])
   records.append(row);target.append(x.get('condition')=='used')
 raw=pd.DataFrame(records);del records;gc.collect();base=build_features(raw).drop(columns='title_text');base['seller_id']=raw.seller_id.astype(str)
 for c in ['initial_quantity','sold_quantity','available_quantity']:
  v=pd.to_numeric(raw[c],errors='coerce').fillna(0);base['log_'+c]=np.log1p(v.clip(lower=0)).astype('float32')
 base['qty_unit']=raw.initial_quantity.eq(1).astype('int8');base['sold_initial_ratio']=(raw.sold_quantity/raw.initial_quantity.clip(lower=1)).astype('float32');base['available_initial_ratio']=(raw.available_quantity/raw.initial_quantity.clip(lower=1)).astype('float32')
 for c in ['official_store_id','status','international_delivery_mode']:base[c]=raw[c].fillna('__NA__').astype(str)
 keys=keyed(raw.seller_id,raw.category_id);seller=raw.seller_id.to_numpy();del raw;gc.collect();return base,keys,np.array(docs,dtype=object),np.array(target[:70000],dtype='int8'),seller

def matrix(base,keys,docs,y,tr,va):
 a,b=assemble(base,keys,y,tr,va);cats=a.select_dtypes(include=['object','string','category']).columns.tolist();nums=[c for c in a if c not in cats]
 encoder=OneHotEncoder(handle_unknown='ignore',min_frequency=5,dtype=np.float32);ac=encoder.fit_transform(a[cats].astype(str));bc=encoder.transform(b[cats].astype(str));an=csr_matrix(np.nan_to_num(a[nums].to_numpy(dtype='float32'),nan=-1,posinf=1e6,neginf=-1e6));bn=csr_matrix(np.nan_to_num(b[nums].to_numpy(dtype='float32'),nan=-1,posinf=1e6,neginf=-1e6));xs=[an,ac];vs=[bn,bc];vec=[]
 for spec in [dict(ngram_range=(1,2),min_df=3,max_features=30000),dict(analyzer='char',ngram_range=(3,5),min_df=4,max_features=30000)]:
  v=TfidfVectorizer(sublinear_tf=True,strip_accents='unicode',dtype=np.float32,**spec);xs.append(v.fit_transform(docs[tr]));vs.append(v.transform(docs[va]));vec.append(v)
 x=hstack(xs,format='csr');z=hstack(vs,format='csr');log('MATRIX',x.shape,z.shape,'nnz',x.nnz);return x,z,{'onehot':encoder,'vectors':vec,'numeric_cols':nums,'categorical_cols':cats}

def main():
 start=time.time();base,keys,docs,y,seller=read();part=np.load(R/'results/new_models_20261021/partitions.npz');fit,tune,conf=[part[k] for k in ['fit','tune','confirm']];dev=np.r_[fit,tune];x,z,_=matrix(base,keys,docs,y,fit,tune);seen=np.isin(seller[tune],seller[fit]);specs=[dict(num_leaves=15,min_child_samples=80,reg_lambda=20)];trials=[];m=None
 # Fixed .85 native+.15 sparse mixture is declared here before confirmation access.
 report={'search_budget':'Closed after config0 before confirmation because sparse training cost consumed budget. Config1 interrupted, never inspected.', 'protocol':'All transforms fit-only; target stats OOF; select standalone config on tune only. Fixed native weight .85 preregistered, no blend selection on confirm. Inherited/reused partitions.','fixed_native_weight':.85}
 (O/'protocol.json').write_text(json.dumps(report,indent=2))
 for i,s in enumerate(specs):
  if (O/f'tune_{i}.npz').exists() and (O/'progress.json').exists():
   saved=json.loads((O/'progress.json').read_text());match=[q for q in saved if q['config']==i]
   if match:trials.append(match[0]);log('RESUME TUNE',match[0]);continue
  m=lgb.LGBMClassifier(n_estimators=600,learning_rate=.05,n_jobs=2,verbosity=-1,random_state=20261022,max_bin=31,subsample=.9,colsample_bytree=.8,**s);m.fit(x,y[fit],eval_set=[(z,y[tune])],callbacks=[lgb.early_stopping(80,verbose=False),lgb.log_evaluation(100)]);p=m.predict_proba(z)[:,1];np.savez_compressed(O/f'tune_{i}.npz',rows=tune,p=p);rec={'config':i,'params':s,'trees':m.best_iteration_,'metrics':metrics(y[tune],p,seen)};trials.append(rec);(O/'progress.json').write_text(json.dumps(trials,indent=2));log('TUNE',rec)
 best=max(trials,key=lambda q:(q['metrics']['accuracy'],-q['metrics']['log_loss']));(O/'selection_lock.json').write_text(json.dumps(best,indent=2));del x,z,m;gc.collect();x,z,_=matrix(base,keys,docs,y,dev,conf);m=lgb.LGBMClassifier(n_estimators=best['trees'],learning_rate=.05,n_jobs=2,verbosity=-1,random_state=20261022,max_bin=31,subsample=.9,colsample_bytree=.8,**best['params']);m.fit(x,y[dev]);p=m.predict_proba(z)[:,1];seen=np.isin(seller[conf],seller[dev]);n=np.load(R/'results/new_models_20261021/seller_native_confirm.npz');assert np.array_equal(n['rows'],conf);native=n['p'];blend=.85*native+.15*p;np.savez_compressed(O/'confirm.npz',rows=conf,p=p,y=y[conf],seen=seen);m.booster_.save_model(str(O/'confirm_model.txt'));report.update(selected=best,confirmation=metrics(y[conf],p,seen),native=metrics(y[conf],native,seen),fixed_blend=metrics(y[conf],blend,seen));log('CONFIRM',report)
 del x,z,m;gc.collect()
 if report['fixed_blend']['accuracy']>report['native']['accuracy'] and os.environ.get('SPARSE_SKIP_FULL')!='1':
  x,z,transforms=matrix(base,keys,docs,y,np.arange(70000),np.arange(70000,100000));m=lgb.LGBMClassifier(n_estimators=best['trees'],learning_rate=.05,n_jobs=2,verbosity=-1,random_state=20261022,max_bin=31,subsample=.9,colsample_bytree=.8,**best['params']);m.fit(x,y);p=m.predict_proba(z)[:,1];np.savez_compressed(O/'test.npz',p=p);m.booster_.save_model(str(O/'full_model.txt'));report['full_refit']='complete'
 else:report['full_refit']='skipped budget cap' if os.environ.get('SPARSE_SKIP_FULL')=='1' else 'skipped no fixed blend gain'
 report['seconds']=time.time()-start;(O/'report.json').write_text(json.dumps(report,indent=2));log('DONE',report)
if __name__=='__main__':main()
