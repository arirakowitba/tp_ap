"""Quantity<=1 specialist. Tune-only model selection; fixed gated policies."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='3'
from pathlib import Path
import json,time,gc
import numpy as np,pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score,log_loss
R=Path(__file__).resolve().parents[1];O=R/'results/training_round2/specialist';O.mkdir(parents=True,exist_ok=True)
def say(*x):print(*x,flush=True)
def metrics(y,p,seen):
 return {'accuracy':float(accuracy_score(y,p>=.5)),'log_loss':float(log_loss(y,p)),'seen_accuracy':float(accuracy_score(y[seen],p[seen]>=.5)),'new_accuracy':float(accuracy_score(y[~seen],p[~seen]>=.5))}
def read():
 out=[];ys=[]
 for f in ['train_data.jsonlines','test_data.jsonlines']:
  for line in open(R.parent/'recovered/data'/f):
   d=json.loads(line);ys.append(int(d.get('condition')=='used'));z={}
   for c in ['seller_id','category_id','listing_type_id','currency_id','buying_mode','status','official_store_id','international_delivery_mode','automatic_relist','listing_source']:
    z[c]=str(d.get(c) or '__missing__')
   for c in ['title','warranty']:z['text_'+c]=str(d.get(c) or '__empty__').lower()
   for c in ['price','base_price','initial_quantity','available_quantity','sold_quantity']:
    v=d.get(c);v=float(v) if isinstance(v,(int,float)) else 0.;z['log_'+c]=np.log1p(max(v,0));z[c+'_raw']=v
   z['qty1']=int(z['initial_quantity_raw']<=1)
   for c in ['pictures','attributes','variations','tags','descriptions','non_mercado_pago_payment_methods']:z['n_'+c]=len(d.get(c) or [])
   sh=d.get('shipping') or {}
   for c in ['mode','free_shipping','local_pick_up']:z['shipping_'+c]=str(sh.get(c,'__missing__'))
   addr=d.get('seller_address') or {}
   for c in ['state','city']:z['seller_'+c]=str((addr.get(c) or {}).get('id') or '__missing__')
   z['title_chars']=len(z['text_title']);z['title_words']=len(z['text_title'].split());z['warranty_chars']=len(z['text_warranty'])
   z['sold_fraction']=z['sold_quantity_raw']/max(z['sold_quantity_raw']+z['available_quantity_raw'],1)
   z['quantity_gap']=z['initial_quantity_raw']-z['sold_quantity_raw']-z['available_quantity_raw']
   z['price_base_ratio']=z['price_raw']/max(z['base_price_raw'],1)
   z['warranty_missing']=int(not d.get('warranty'));z['video_missing']=int(not d.get('video_id'));z['parent_missing']=int(not d.get('parent_item_id'))
   out.append(z)
 return pd.DataFrame(out),np.array(ys[:70000],dtype=np.int8)
def main():
 start=time.time();X,y=read();seller=X.seller_id.to_numpy();q=X.qty1.to_numpy(bool);part=np.load(R/'results/new_models_20261021/partitions.npz');fit,tune,confirm=[part[c] for c in ['fit','tune','confirm']];dev=np.r_[fit,tune]
 texts=['text_title','text_warranty'];cats=[c for c in X if X[c].dtype=='object' and c not in texts]
 def enrich(rows):
  Z=X.copy()
  for key in ['category_id','seller_id']:
   s=X.iloc[rows].groupby(key).log_price
   for stat in ['median','std','count']:
    v=X[key].map(getattr(s,stat)()).fillna(0);Z[key+'_price_'+stat]=v
    if stat=='median':Z[key+'_relative_price']=X.log_price-v
  return Z
 base_t=np.load(R/'results/training_next/baseline_reproduction/seller_native_tune.npz');assert np.array_equal(base_t['rows'],tune)
 opts=dict(loss_function='Logloss',thread_count=3,random_seed=20261023,verbose=200,allow_writing_files=False,one_hot_max_size=20,tokenizers=[{'tokenizer_id':'Space','separator_type':'ByDelimiter','delimiter':' '}],dictionaries=[{'dictionary_id':'Word','max_dictionary_size':'5000','occurrence_lower_bound':'3'}],feature_calcers=['BoW','NaiveBayes'])
 specs=[dict(depth=6,iterations=1000,learning_rate=.06,l2_leaf_reg=10),dict(depth=8,iterations=1200,learning_rate=.06,l2_leaf_reg=30)]
 report={'protocol':'Specialist qty<=1, model config selected subset tune accuracy then logloss. For quantity>1 native is always retained. Fixed .5 specialist/native blend predeclared before confirm. Historical row split reused; no public/test labels.','specs':specs,'n_fit_specialist':int(q[fit].sum()),'n_tune_specialist':int(q[tune].sum())};Z=enrich(fit);best=None
 for i,spec in enumerate(specs):
  m=CatBoostClassifier(**opts,**spec);m.fit(Z.iloc[fit[q[fit]]],y[fit[q[fit]]],cat_features=cats,text_features=texts,eval_set=(Z.iloc[tune[q[tune]]],y[tune[q[tune]]]),early_stopping_rounds=100)
  pure=m.predict_proba(Z.iloc[tune])[:,1];gate=np.where(q[tune],pure,base_t['p']);mix=np.where(q[tune],.5*pure+.5*base_t['p'],base_t['p']);seen=np.isin(seller[tune],seller[fit]);sub={'accuracy':float(accuracy_score(y[tune[q[tune]]],pure[q[tune]]>=.5)),'log_loss':float(log_loss(y[tune[q[tune]]],pure[q[tune]]))};report[str(i)]={'trees':m.tree_count_,'subset_tune':sub,'gate_tune':metrics(y[tune],gate,seen),'fixed_blend_tune':metrics(y[tune],mix,seen)};np.savez_compressed(O/f'tune_{i}.npz',rows=tune,p=pure,gate=gate,blend=mix);(O/'progress.json').write_text(json.dumps(report,indent=2));say('TUNE',i,report[str(i)])
  score=(sub['accuracy'],-sub['log_loss'])
  if best is None or score>best[0]:best=(score,i,m.tree_count_)
  del m;gc.collect()
 i,trees=best[1:];spec=specs[i].copy();spec['iterations']=trees;report['selected']=i;(O/'selection_lock.json').write_text(json.dumps({'config':i,'trees':trees,'selection':'Subset tune accuracy, tie logloss','gate':'qty<=1 pure specialist else native','fixed_blend_weight':.5},indent=2));del Z;gc.collect();Z=enrich(dev);m=CatBoostClassifier(**opts,**spec);m.fit(Z.iloc[dev[q[dev]]],y[dev[q[dev]]],cat_features=cats,text_features=texts);pure=m.predict_proba(Z.iloc[confirm])[:,1];b=np.load(R/'results/new_models_20261021/seller_native_confirm.npz');assert np.array_equal(b['rows'],confirm);gate=np.where(q[confirm],pure,b['p']);mix=np.where(q[confirm],.5*pure+.5*b['p'],b['p']);seen=np.isin(seller[confirm],seller[dev]);report['native_confirm']=metrics(y[confirm],b['p'],seen);report['gate_confirm']=metrics(y[confirm],gate,seen);report['fixed_blend_confirm']=metrics(y[confirm],mix,seen);np.savez_compressed(O/'confirm.npz',rows=confirm,p=pure,gate=gate,blend=mix,y=y[confirm],seen=seen,qty1=q[confirm]);say('CONFIRM',report);(O/'report.json').write_text(json.dumps(report,indent=2))
 if max(report['gate_confirm']['accuracy'],report['fixed_blend_confirm']['accuracy'])>report['native_confirm']['accuracy']:
  del m,Z;gc.collect();rows=np.arange(70000);Z=enrich(rows);m=CatBoostClassifier(**opts,**spec);m.fit(Z.iloc[rows[q[:70000]]],y[q[:70000]],cat_features=cats,text_features=texts);pure=m.predict_proba(Z.iloc[70000:])[:,1];b=np.load(R/'results/new_models_20261021/seller_native_test.npz')['test'];gate=np.where(q[70000:],pure,b);mix=np.where(q[70000:],.5*pure+.5*b,b);np.savez_compressed(O/'test.npz',p=pure,gate=gate,blend=mix);m.save_model(str(O/'model.cbm'));report['full_refit']='complete'
 else:report['full_refit']='Skipped: neither predeclared policy improves native confirm'
 report['seconds']=time.time()-start;(O/'report.json').write_text(json.dumps(report,indent=2));say('DONE',report)
if __name__=='__main__':main()
