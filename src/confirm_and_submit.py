"""Frozen ensemble check on alternate grouped split and full-data submission."""
import os,sys,gc,time,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from optimize_leaderboard import ROOT,OUT,enrich,split,numeric_text,record,data_folder
from optimize_stack import group_rates,text_predict
from optimize_neighbors import neighbor_features

def linear_predict(x,y,z,C=1.):
 word=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=120000,sublinear_tf=True,strip_accents='unicode',dtype=np.float32)
 char=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=3,max_features=140000,sublinear_tf=True,dtype=np.float32)
 cols=['seller_id','category_id','listing_type_id','shipping_mode','seller_state','title_first','address_city_name']
 enc=OneHotEncoder(handle_unknown='ignore',min_frequency=3,dtype=np.float32)
 nums=x.select_dtypes(include=np.number).columns
 imp=SimpleImputer(strategy='median',add_indicator=True);scale=StandardScaler()
 nd=scale.fit_transform(imp.fit_transform(x[nums].replace([np.inf,-np.inf],np.nan)));nv=scale.transform(imp.transform(z[nums].replace([np.inf,-np.inf],np.nan)))
 a=sparse.hstack([word.fit_transform(x.full_text),char.fit_transform(x.full_text),enc.fit_transform(x[cols].astype(str)),sparse.csr_matrix(nd,dtype=np.float32)],format='csr')
 b=sparse.hstack([word.transform(z.full_text),char.transform(z.full_text),enc.transform(z[cols].astype(str)),sparse.csr_matrix(nv,dtype=np.float32)],format='csr')
 lr=LogisticRegression(C=C,solver='liblinear',max_iter=350);lr.fit(a,y)
 return lr.predict_proba(b)[:,1]

def fit_models(x,y,groups,z,selected,protocol,y_eval=None):
 both=pd.concat([x,z],ignore_index=True);it=np.arange(len(x));iv=np.arange(len(x),len(both))
 tr,va,cats,tf,svd=numeric_text(both,it,iv);del both;gc.collect()
 needs_stack=any('stack' in n or 'neighbors' in n or 'pair_ctr' in n or n.startswith(('lgb_fast','lgb_tpe','xgb_native')) for n in selected['names'])
 if needs_stack:
  st,sv=group_rates(tr,va,y,groups);oof=np.zeros(len(y))
  for k,(tt,vv) in enumerate(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=2026).split(np.zeros(len(y)),y,groups)):
   oof[vv]=text_predict(x.iloc[tt],y[tt],x.iloc[vv]);print(protocol,'OOF',k,'DONE',flush=True)
  st['text_oof_probability']=oof;sv['text_oof_probability']=text_predict(x,y,z)
  if any('neighbors' in n for n in selected['names']):
   na=neighbor_features(x,y,groups,x,groups)
   nb=neighbor_features(x,y,groups,z,np.array(['__QUERY__'+str(i) for i in range(len(z))]))
   st=pd.concat([st,na],axis=1);sv=pd.concat([sv,nb],axis=1)
 probs=[];metrics_table=pd.read_csv(OUT/'metrics.csv')
 for name in selected['names']:
  print(protocol,'MODEL',name,flush=True);ts=time.time()
  cached_path=OUT/(name+'_'+protocol+'.npy')
  if cached_path.exists():
   print(protocol,'MODEL',name,'LOADED FROM CACHE',cached_path,flush=True)
   p=np.load(cached_path);probs.append(p);continue
  if name.startswith('logistic_word'):
   p=linear_predict(x,y,z,C=1. if name.endswith('C1') else 4.)
  elif name=='logistic_text_oof_stack_base':p=text_predict(x,y,z)
  elif name.startswith(('lightgbm','lgb_fast','lgb_tpe','xgb_native','lgb_neighbors')):
   a,b=st.copy(),sv.copy()
   for c in cats:
    dtype=pd.CategoricalDtype(categories=pd.Index(a[c].unique()));a[c]=a[c].astype(dtype);b[c]=b[c].astype(dtype)
   n=int(metrics_table[metrics_table.model==name].iloc[0].best_iteration)
   if name.startswith(('lgb_fast','lgb_tpe','lgb_neighbors')):
    params=json.loads((OUT/(name+'_params.json')).read_text());params['n_estimators']=n
    model=LGBMClassifier(**params)
   elif name.startswith('xgb_native'):
    params=json.loads((OUT/(name+'_params.json')).read_text());params.pop('early_stopping_rounds',None);params['n_estimators']=n+1
    model=XGBClassifier(**params)
   else:
    model=LGBMClassifier(n_estimators=n,num_leaves=int(name.split('leaves')[-1]),learning_rate=.035,max_depth=-1,min_child_samples=40,colsample_bytree=.8,subsample=.85,subsample_freq=1,reg_lambda=8,max_cat_threshold=32,cat_smooth=20,n_jobs=8,verbosity=-1,random_state=42)
   model.fit(a,y);p=model.predict_proba(b)[:,1]
   if name.startswith('xgb'):model.save_model(OUT/(name+'_'+protocol+'.json'))
   else:model.booster_.save_model(str(OUT/(name+'_'+protocol+'.txt')))

  else:
   params=json.loads((OUT/(name+'_params.json')).read_text());n=int(metrics_table[metrics_table.model==name].iloc[0].best_iteration)+1;params['iterations']=n
   a,b=(st.copy(),sv.copy()) if ('stack' in name or 'neighbors' in name or 'pair_ctr' in name) else (tr.copy(),va.copy());texts=[]
   if 'native' in name:a['full_text']=x.full_text.to_numpy();b['full_text']=z.full_text.to_numpy();texts=['full_text']
   model=CatBoostClassifier(**params,loss_function='Logloss',eval_metric='Accuracy',thread_count=12,allow_writing_files=False,one_hot_max_size=10,random_strength=.5)
   model.fit(a,y,cat_features=cats,text_features=texts,verbose=200);p=model.predict_proba(b)[:,1];model.save_model(str(OUT/(name+'_'+protocol+'.cbm')))
  probs.append(p)
  if y_eval is not None:record(name,p,y_eval,protocol=protocol,seconds=time.time()-ts)
  np.save(OUT/(name+'_'+protocol+'.npy'),p);gc.collect()
 combined=sum(w*p for w,p in zip(selected['weights'],probs));return combined

def main():
 selected=json.loads((OUT/'selected_ensemble.json').read_text());folder=data_folder()
 raw=pd.read_json(folder/'train_data.jsonlines',lines=True);y=(raw.condition=='used').astype('int8').to_numpy();groups=raw.parent_item_id.astype('string').fillna('__ROW__'+raw.id.astype(str)).to_numpy()
 t,v=split(y,groups,seed=731,n_holdout=2);print('ALTERNATE SPLIT',len(t),len(v),flush=True)
 # Reselect attribute IDs and refit all representations on the new development split.
 x,ids=enrich(raw.iloc[t].reset_index(drop=True));z,_=enrich(raw.iloc[v].reset_index(drop=True),ids);del raw;gc.collect()
 p=fit_models(x,y[t],groups[t],z,selected,'alternate_seed731',y[v]);pred=p>=selected['threshold']
 result=dict(protocol='alternate_seed731',accuracy=accuracy_score(y[v],pred),balanced_accuracy=balanced_accuracy_score(y[v],pred),macro_f1=f1_score(y[v],pred,average='macro'),n_validation=len(v),n_fit=len(t),threshold=selected['threshold'])
 (OUT/'confirmation.json').write_text(json.dumps(result,indent=2));print('CONFIRMATION',result,flush=True)
 del x,z,p;gc.collect()
 raw=pd.read_json(folder/'train_data.jsonlines',lines=True);full,ids=enrich(raw);del raw;gc.collect()
 test=pd.read_json(folder/'test_data.jsonlines',lines=True);x_test,_=enrich(test,ids);del test;gc.collect()
 p=fit_models(full,y,groups,x_test,selected,'full_train')
 sub=pd.DataFrame({'ID':np.arange(1,len(p)+1),'condition':np.where(p>=selected['threshold'],'used','new')})
 assert len(sub)==30000 and sub.ID.is_unique and set(sub.condition)<={'used','new'}
 sub.to_csv(OUT/'submission_optimized_ensemble.csv',index=False)
 print('SUBMISSION READY',OUT/'submission_optimized_ensemble.csv',flush=True)

if __name__=='__main__':main()
