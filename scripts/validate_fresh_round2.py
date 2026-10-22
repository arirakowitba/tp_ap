"""Second row split robustness: all base fits recreated, fixed parameters, no public labels.
Not historically virgin: earlier experimentation used the same 70k training data.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,json,time,gc
import numpy as np,pandas as pd
from scipy.special import logit
from scipy.stats import binomtest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,log_loss
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
threadpool_limits(2)
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'));sys.path.insert(0,str(R/'scripts'))
from ap2026q2_common import build_features
from train_nbsvm_next import vectorize,ratio,model as nb_model
D=R/'results/training_round2/fresh';D.mkdir(parents=True,exist_ok=True);START=time.time()
def status(stage,**kw):
 a={'stage':stage,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw};(D/'status.json').write_text(json.dumps(a,indent=2));print(a,flush=True)
def data():
 cache=D/'features.pkl';aux=D/'input_arrays.npz'
 if cache.exists() and aux.exists():
  a=np.load(aux,allow_pickle=True);return pd.read_pickle(cache),a['docs'],a['y'],a['seller'],a['qty'],a['listing']
 parts=[];docs=[];ys=[];seller=[];qty=[];listing=[];chunk=[]
 def consume(rows):
  raw=pd.DataFrame(rows);x=build_features(raw).drop(columns='title_text')
  for c in ['seller_id','official_store_id','status','international_delivery_mode']:x[c]=raw[c].fillna('__missing__').astype(str)
  for c in ['warranty','title','attributes']:
   x['text_'+c]=raw[c].map(lambda v:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else str(v or '__empty__')).str.lower().replace('','__empty__')
  for c in ['initial_quantity','available_quantity','sold_quantity']:x['log_'+c]=np.log1p(pd.to_numeric(raw[c],errors='coerce').clip(lower=0)).astype(float)
  parts.append(x)
 for line in open(R.parent/'recovered/data/train_data.jsonlines'):
  x=json.loads(line);chunk.append(x);title=str(x.get('title') or '');w=str(x.get('warranty') or '');a=' '.join(str(v.get('name',''))+' '+str(v.get('value_name','')) for v in x.get('attributes',[]) if isinstance(v,dict));docs.append((title+' '+title+' warranty '+w+' attributes '+a+' category_'+str(x.get('category_id',''))).lower());ys.append(x.get('condition')=='used');seller.append(str(x['seller_id']));qty.append(float(x.get('initial_quantity') or 0));listing.append(str(x.get('listing_type_id')))
  if len(chunk)==5000:consume(chunk);chunk=[]
 if chunk:consume(chunk)
 X=pd.concat(parts,ignore_index=True);del parts
 for c in X:
  if c.startswith('text_'):continue
  if not pd.api.types.is_numeric_dtype(X[c]):X[c]=X[c].fillna('__missing__').astype(str)
  else:X[c]=X[c].astype(float).fillna(-1)
 docs=np.array(docs,dtype=object);y=np.array(ys,dtype=np.int8);seller=np.array(seller);qty=np.array(qty);listing=np.array(listing);X.to_pickle(cache);np.savez_compressed(aux,docs=docs,y=y,seller=seller,qty=qty,listing=listing);return X,docs,y,seller,qty,listing
def metrics(y,p,seen):
 return {'accuracy':float(accuracy_score(y,p>=.5)),'log_loss':float(log_loss(y,p)),'seen_accuracy':float(accuracy_score(y[seen],p[seen]>=.5)),'unseen_accuracy':float(accuracy_score(y[~seen],p[~seen]>=.5)),'n_seen':int(seen.sum()),'n_unseen':int((~seen).sum()),'brier':float(np.mean((p-y)**2)),'uncertain_40_60':int(((p>.4)&(p<.6)).sum()),'confident_errors_90':int((((p>=.5)!=y)&(np.maximum(p,1-p)>=.9)).sum())}
def meta_x(pc,pn,rows,seller,qty,listing,reference):
 l=logit(np.clip(np.column_stack([pc,pn]),1e-5,1-1e-5));seen=np.isin(seller[rows],seller[reference]);q1=qty[rows]<=1;free=listing[rows]=='free';delta=l[:,1]-l[:,0]
 return np.column_stack([l,seen,q1,free,delta*seen,delta*q1,delta*free]),seen
def main():
 status('loading_stream');X,docs,y,seller,qty,listing=data();assert len(y)==70000
 dev,confirm=train_test_split(np.arange(70000),test_size=.2,stratify=y,random_state=20261023);fit,tune=train_test_split(dev,test_size=.125,stratify=y[dev],random_state=20261024);np.savez_compressed(D/'partitions.npz',fit=fit,tune=tune,confirm=confirm)
 texts=['text_title','text_warranty','text_attributes'];cats=[c for c in X if X[c].dtype=='object' and c not in texts]
 opts=dict(depth=7,iterations=650,learning_rate=.065,l2_leaf_reg=20,loss_function='Logloss',thread_count=2,random_seed=20261021,verbose=200,allow_writing_files=False,one_hot_max_size=20,tokenizers=[{'tokenizer_id':'Space','separator_type':'ByDelimiter','delimiter':' '}],dictionaries=[{'dictionary_id':'Word','max_dictionary_size':'2500','occurrence_lower_bound':'5'}],feature_calcers=['BoW','NaiveBayes'])
 # Each stage is an independent base fit. Never reuse old predictions for this split.
 pp={}
 for stage,tr,va in [('tune',fit,tune),('confirm',dev,confirm)]:
  path=D/(stage+'_native.npz')
  if path.exists():
   a=np.load(path);assert np.array_equal(a['rows'],va);pc=a['p']
  else:
   status(stage+'_native_fit',n_fit=len(tr));m=CatBoostClassifier(**opts);m.fit(X.iloc[tr],y[tr],cat_features=cats,text_features=texts);pc=m.predict_proba(X.iloc[va])[:,1];np.savez_compressed(path,rows=va,p=pc);del m;gc.collect()
  path=D/(stage+'_nb.npz')
  if path.exists():
   a=np.load(path);assert np.array_equal(a['rows'],va);pn=a['p']
  else:
   status(stage+'_nb_fit',n_fit=len(tr));a,[b],v=vectorize(docs,tr,[va]);rr=ratio(a,y[tr]);m=nb_model(a.multiply(rr).tocsr(),y[tr],4);pn=m.predict_proba(b.multiply(rr).tocsr())[:,1];np.savez_compressed(path,rows=va,p=pn);del a,b,v,m,rr;gc.collect()
  pp[stage]=(pc,pn);status(stage+'_base_complete')
  if stage=='tune':
   xx,_=meta_x(pc,pn,tune,seller,qty,listing,fit);meta=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1000)).fit(xx,y[tune]);import joblib;joblib.dump(meta,D/'meta_model.joblib');(D/'selection_lock.json').write_text(json.dumps({'seed':20261023,'baseline_params':opts,'nb_C':4,'meta_C':.1,'blend_weight_native':.85,'selection':'All hyperparameters fixed before this second split confirmation; meta fitted only7k tune predictions.','feature_names':['native_logit','nb_logit','seen','quantity_le1','listing_free','delta_x_seen','delta_x_quantity_le1','delta_x_listing_free']},indent=2))
 pc,pn=pp['confirm'];xx,seen=meta_x(pc,pn,confirm,seller,qty,listing,dev);pm=meta.predict_proba(xx)[:,1];pb=.85*pc+.15*pn;np.savez_compressed(D/'confirmation.npz',rows=confirm,y=y[confirm],native=pc,nb=pn,blend=pb,meta=pm,seen=seen)
 report={'protocol':'Second deterministic random row split; baseline and NB fully refit. No historical predictions, test/public labels, or old meta weights. All hyperparameters fixed. Same70k data historically used, so not a virgin holdout.','seed':20261023,'native':metrics(y[confirm],pc,seen),'nb':metrics(y[confirm],pn,seen),'fixed_blend85native':metrics(y[confirm],pb,seen),'meta_fixed_C01':metrics(y[confirm],pm,seen),'paired':{}}
 old=(pc>=.5)==y[confirm]
 for name,p in [('fixed_blend',pb),('meta',pm)]:
  new=(p>=.5)==y[confirm];g=int((new&~old).sum());l=int((old&~new).sum());report['paired'][name]={'gains':g,'losses':l,'net':g-l,'p_unadjusted':float(binomtest(g,g+l,.5).pvalue) if g+l else 1}
 report['seconds']=time.time()-START;(D/'report.json').write_text(json.dumps(report,indent=2));status('done',report=report)
if __name__=='__main__':main()
