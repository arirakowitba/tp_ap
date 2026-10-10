import time,json,gc,re,unicodedata
import numpy as np,pandas as pd
from rapidfuzz import process,fuzz
from lightgbm import LGBMClassifier,early_stopping
from catboost import CatBoostClassifier
from optimize_leaderboard import OUT,prepare,record

def normalize(text):
 return re.sub(r'[^a-z0-9 ]',' ',unicodedata.normalize('NFKD',str(text)).encode('ascii','ignore').decode().lower())

def neighbor_features(fit_x,y,fit_groups,query_x,query_groups):
 titles=[normalize(s) for s in fit_x.title_text];query_titles=[normalize(s) for s in query_x.title_text]
 out=pd.DataFrame(index=np.arange(len(query_x)))
 for key in ['seller_id','category_id']:
  trainkeys=fit_x[key].astype(str).to_numpy();querykeys=query_x[key].astype(str).to_numpy()
  indexed=pd.Series(np.arange(len(fit_x))).groupby(trainkeys).apply(lambda z:z.to_numpy()).to_dict()
  queryindexed=pd.Series(np.arange(len(query_x))).groupby(querykeys).apply(lambda z:z.to_numpy()).to_dict()
  features=np.full((len(query_x),4),np.nan,dtype=np.float32)
  for value,rows in queryindexed.items():
   candidates=indexed.get(value)
   if candidates is None:continue
   texts=[titles[i] for i in candidates]
   for start in range(0,len(rows),128):
    batch=rows[start:start+128]
    sims=process.cdist([query_titles[i] for i in batch],texts,scorer=fuzz.ratio,dtype=np.float32,workers=1)/100
    sims[query_groups[batch,None]==fit_groups[candidates][None,:]]=-1
    k=min(5,len(candidates));idx=np.argpartition(sims,-k,axis=1)[:,-k:];vals=np.take_along_axis(sims,idx,axis=1)
    weights=np.maximum(vals,0)**6;den=weights.sum(axis=1)
    rate=np.divide((weights*y[candidates[idx]]).sum(axis=1),den,out=np.full(len(batch),np.nan),where=den>0)
    nearest=np.argmax(sims,axis=1);top=sims[np.arange(len(batch)),nearest]
    exact=sims>=.9999;count=exact.sum(axis=1)
    exactrate=np.divide((exact*y[candidates][None,:]).sum(axis=1),count,out=np.full(len(batch),np.nan),where=count>0)
    features[batch,0]=np.maximum(top,0);features[batch,1]=rate;features[batch,2]=exactrate;features[batch,3]=np.log1p((sims>0).sum(axis=1))
  out=pd.concat([out,pd.DataFrame(features,columns=[key+'_neighbor_similarity',key+'_neighbor_rate',key+'_exact_title_rate',key+'_neighbor_count'])],axis=1)
 return out

def main():
 d=prepare();x,y,t,v,g=d['x'],d['y'],d['t'],d['v'],d['groups']
 tr,va,yt,yv=pd.read_pickle(OUT/'stack_matrix_cache.pkl');ts=time.time()
 a=neighbor_features(x.iloc[t].reset_index(drop=True),yt,g[t],x.iloc[t].reset_index(drop=True),g[t])
 b=neighbor_features(x.iloc[t].reset_index(drop=True),yt,g[t],x.iloc[v].reset_index(drop=True),g[v])
 tr=pd.concat([tr.reset_index(drop=True),a],axis=1);va=pd.concat([va.reset_index(drop=True),b],axis=1)
 print('NEIGHBORS READY',time.time()-ts,flush=True)
 cats=tr.select_dtypes(include='category').columns.tolist()
 pd.to_pickle((tr,va),OUT/'neighbor_matrix_cache.pkl')
 for leaves in [31,63]:
  name=f'lgb_neighbors_{leaves}';params=dict(n_estimators=2000,num_leaves=leaves,learning_rate=.035,min_child_samples=40,colsample_bytree=.8,subsample=.85,subsample_freq=1,reg_lambda=8,max_cat_threshold=32,cat_smooth=20,n_jobs=8,verbosity=-1,random_state=42)
  m=LGBMClassifier(**params);ts=time.time();m.fit(tr,yt,eval_set=[(va,yv)],eval_metric='binary_logloss',callbacks=[early_stopping(100,verbose=False)])
  record(name,m.predict_proba(va)[:,1],yv,seconds=time.time()-ts,best_iteration=m.best_iteration_);(OUT/(name+'_params.json')).write_text(json.dumps(params));del m;gc.collect()
 for c in cats:tr[c]=tr[c].astype('string').fillna('__MISSING__').astype(str);va[c]=va[c].astype('string').fillna('__MISSING__').astype(str)
 name='cat_neighbors_d7';params=dict(iterations=1400,depth=7,learning_rate=.035,l2_leaf_reg=10,random_seed=51)
 m=CatBoostClassifier(**params,loss_function='Logloss',eval_metric='Accuracy',thread_count=8,allow_writing_files=False,early_stopping_rounds=150,one_hot_max_size=10,random_strength=.5)
 ts=time.time();m.fit(tr,yt,cat_features=cats,eval_set=(va,yv),verbose=200)
 record(name,m.predict_proba(va)[:,1],yv,seconds=time.time()-ts,best_iteration=m.best_iteration_);(OUT/(name+'_params.json')).write_text(json.dumps(params));m.save_model(str(OUT/(name+'.cbm')))

if __name__=='__main__':main()
