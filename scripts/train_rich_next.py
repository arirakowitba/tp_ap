"""Label-free relational features and native text. Existing row partition, tune-only selection."""
from pathlib import Path
import sys,json,time
import numpy as np,pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score,log_loss
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from ap2026q2_common import build_features
OUT=ROOT/'results/training_next/rich';OUT.mkdir(parents=True,exist_ok=True)
def say(*a):print(*a,flush=True)
def metric(y,p,seen):return dict(accuracy=float(accuracy_score(y,p>=.5)),logloss=float(log_loss(y,p)),seen_accuracy=float(accuracy_score(y[seen],p[seen]>=.5)),new_accuracy=float(accuracy_score(y[~seen],p[~seen]>=.5)))
def main():
 t=time.time();raw=pd.concat([pd.read_json(ROOT.parent/'recovered/data/train_data.jsonlines',lines=True),pd.read_json(ROOT.parent/'recovered/data/test_data.jsonlines',lines=True)],ignore_index=True);y=raw.condition.iloc[:70000].eq('used').to_numpy(int)
 part=np.load(ROOT/'results/new_models_20261021/partitions.npz');fit,tune,confirm=[part[k] for k in ['fit','tune','confirm']];dev=np.r_[fit,tune]
 X=build_features(raw).drop(columns='title_text');sellers=raw.seller_id.astype(str)
 for c in ['seller_id','official_store_id','status','international_delivery_mode','catalog_product_id']:X[c]=raw[c].fillna('__missing__').astype(str)
 for c in ['title','warranty']:X['text_'+c]=raw[c].fillna('__empty__').astype(str).str.lower().replace('','__empty__')
 banned={'ITEM_CONDITION','CONDITION'};audit={}
 def attrs(v):
  a=[]
  for d in v if isinstance(v,list) else []:
   k=str(d.get('id',''))
   if 'CONDITION' in k.upper():audit[k]=audit.get(k,0)+1;continue
   a.append(k+'_'+str(d.get('value_name') or d.get('value_id') or 'missing').replace(' ','_'))
  return ' '.join(a) or '__empty__'
 X['text_attributes']=raw.attributes.map(attrs)
 for field in ['BRAND','MODEL','MATERIAL','GENDER']:
  X['attr_'+field]=raw.attributes.map(lambda a:next((str(d.get('value_name') or d.get('value_id') or '__missing__') for d in a if str(d.get('id'))==field),'__missing__'))
 for c in ['initial_quantity','available_quantity','sold_quantity']:
  X['log_'+c]=np.log1p(pd.to_numeric(raw[c],errors='coerce').clip(lower=0)).astype(float)
 X['sold_fraction']=(raw.sold_quantity/(raw.sold_quantity+raw.available_quantity).replace(0,np.nan)).fillna(0)
 X['quantity_gap']=raw.initial_quantity-raw.available_quantity-raw.sold_quantity
 for c in ['available_quantity','sold_quantity','price']:
  X['variation_'+c+'_sum']=raw.variations.map(lambda a:sum(float(v.get(c,0) or 0) for v in a))
 X['shipping_tags']=raw.shipping.map(lambda a:' '.join(sorted(a.get('tags',[]))) or '__empty__')
 X['tags_set']=raw.tags.map(lambda a:' '.join(sorted(a)) or '__empty__')
 for c in ['automatic_relist','listing_source']:X[c]=raw[c].fillna('__missing__').astype(str)
 texts=['text_title','text_warranty','text_attributes']
 for c in X:
  if c in texts:continue
  if not pd.api.types.is_numeric_dtype(X[c]):X[c]=X[c].fillna('__missing__').astype(str)
  else:X[c]=X[c].astype(float).replace([np.inf,-np.inf],np.nan).fillna(-1)
 def relational(rows):
  Z=X.copy();train=raw.iloc[rows];lp=np.log1p(raw.price.clip(lower=0))
  for key in ['category_id','seller_id']:
   grp=pd.DataFrame({'key':train[key].astype(str).values,'p':lp.iloc[rows].values}).groupby('key').p
   for stat in ['median','mean','std','count']:
    m=getattr(grp,stat)();v=raw[key].astype(str).map(m)
    Z[key+'_price_'+stat]=v.fillna(0).to_numpy()
    if stat in ['median','mean']:Z[key+'_price_relative_'+stat]=(lp-v).fillna(0).to_numpy()
   count=train.groupby(key).category_id.nunique();Z[key+'_category_diversity']=raw[key].map(count).fillna(0).to_numpy()
  return Z
 options=dict(loss_function='Logloss',thread_count=3,random_seed=20261022,verbose=200,allow_writing_files=False,one_hot_max_size=20,tokenizers=[{'tokenizer_id':'Space','separator_type':'ByDelimiter','delimiter':' '}],dictionaries=[{'dictionary_id':'Word','max_dictionary_size':'10000','occurrence_lower_bound':'3'}],feature_calcers=['BoW','NaiveBayes','BM25'])
 specs=[dict(depth=7,iterations=900,learning_rate=.06,l2_leaf_reg=12),dict(depth=6,iterations=1100,learning_rate=.05,l2_leaf_reg=25)]
 cats=[c for c in X if X[c].dtype=='object' and c not in texts];Z=relational(fit);report={'excluded_condition_attributes':audit,'protocol':'Existing fit49k/tune7k, select tune then refit56k/confirm14k. Label-free stats fit scoped. No test labels.'};best=None
 for i,spec in enumerate(specs):
  model=CatBoostClassifier(**options,**spec);say('TUNE',i,Z.shape);model.fit(Z.iloc[fit],y[fit],cat_features=cats,text_features=texts,eval_set=(Z.iloc[tune],y[tune]),early_stopping_rounds=100)
  p=model.predict_proba(Z.iloc[tune])[:,1];score=metric(y[tune],p,sellers.iloc[tune].isin(sellers.iloc[fit]).to_numpy());report[str(i)]={'spec':spec,'trees':model.tree_count_,'tune':score};np.savez_compressed(OUT/f'config{i}_tune.npz',rows=tune,p=p)
  (OUT/'progress.json').write_text(json.dumps(report,indent=2));say(report[str(i)])
  candidate=(score['accuracy'],-score['logloss']);
  if best is None or candidate>best[0]:best=(candidate,i,model.tree_count_)
 i,trees=best[1:];spec=specs[i].copy();spec['iterations']=trees;Z=relational(dev);model=CatBoostClassifier(**options,**spec);model.fit(Z.iloc[dev],y[dev],cat_features=cats,text_features=texts);p=model.predict_proba(Z.iloc[confirm])[:,1];seen=sellers.iloc[confirm].isin(sellers.iloc[dev]).to_numpy();report['selected']=i;report['confirmation']=metric(y[confirm],p,seen);np.savez_compressed(OUT/'confirm.npz',rows=confirm,p=p,y=y[confirm],seen=seen)
 baseline=np.load(ROOT/'results/new_models_20261021/seller_native_confirm.npz');assert np.array_equal(baseline['rows'],confirm);report['baseline_confirmation']=metric(y[confirm],baseline['p'],seen);say('CONFIRM',report['confirmation']);(OUT/'progress.json').write_text(json.dumps(report,indent=2))
 Z=relational(np.arange(70000));model=CatBoostClassifier(**options,**spec);model.fit(Z.iloc[:70000],y,cat_features=cats,text_features=texts);pt=model.predict_proba(Z.iloc[70000:])[:,1];model.save_model(str(OUT/'model.cbm'));np.savez_compressed(OUT/'test.npz',p=pt);report['elapsed_seconds']=time.time()-t;(OUT/'report.json').write_text(json.dumps(report,indent=2));say('DONE',report)
if __name__=='__main__':main()
