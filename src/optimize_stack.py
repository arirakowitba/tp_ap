import sys,os,gc,time,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier,early_stopping,log_evaluation
from optimize_leaderboard import ROOT,OUT,prepare,numeric_text,record

def group_rates(tr,va,y,groups):
 tr=tr.copy();va=va.copy();prior=y.mean()
 specs=[('seller_id',5),('category_id',30),('seller_category',5),('seller_listing',5),('title_first',20),('title_first2',10),('warranty_text',10),('listing_type_id',30)]
 folds=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=2026).split(np.zeros(len(y)),y,groups))
 for col,alpha in specs:
  key=tr[col].astype(str).reset_index(drop=True);vk=va[col].astype(str).reset_index(drop=True)
  rates=np.zeros(len(y));counts=np.zeros(len(y))
  for it,iv in folds:
   stats=pd.DataFrame({'key':key.iloc[it].to_numpy(),'y':y[it]}).groupby('key').y.agg(['sum','count'])
   sums=key.iloc[iv].map(stats['sum']).fillna(0).to_numpy();cnt=key.iloc[iv].map(stats['count']).fillna(0).to_numpy()
   rates[iv]=(sums+alpha*y[it].mean())/(cnt+alpha);counts[iv]=np.log1p(cnt)
  stats=pd.DataFrame({'key':key,'y':y}).groupby('key').y.agg(['sum','count'])
  tr[col+'_oof_rate']=rates;tr[col+'_oof_count']=counts
  va[col+'_oof_rate']=(vk.map(stats['sum']).fillna(0).to_numpy()+alpha*prior)/(vk.map(stats['count']).fillna(0).to_numpy()+alpha)
  va[col+'_oof_count']=np.log1p(vk.map(stats['count']).fillna(0).to_numpy())
 return tr,va

def text_predict(fit_x,y,eval_x):
 word=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=100000,sublinear_tf=True,strip_accents='unicode',dtype=np.float32)
 char=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=3,max_features=100000,sublinear_tf=True,dtype=np.float32)
 cols=['seller_id','category_id','listing_type_id','shipping_mode','title_first']
 enc=OneHotEncoder(handle_unknown='ignore',min_frequency=3,dtype=np.float32)
 a=sparse.hstack([word.fit_transform(fit_x.full_text),char.fit_transform(fit_x.title_text+' '+fit_x.warranty_text),enc.fit_transform(fit_x[cols].astype(str))],format='csr')
 b=sparse.hstack([word.transform(eval_x.full_text),char.transform(eval_x.title_text+' '+eval_x.warranty_text),enc.transform(eval_x[cols].astype(str))],format='csr')
 lr=LogisticRegression(C=4,solver='liblinear',max_iter=250);lr.fit(a,y)
 return lr.predict_proba(b)[:,1]

def main():
 d=prepare();x,y,t,v,g=d['x'],d['y'],d['t'],d['v'],d['groups']
 tr,va,cats,tf,svd=numeric_text(x,t,v);tr,va=group_rates(tr,va,y[t],g[t])
 # Full text pipeline is refitted inside every inner group fold.
 oof=np.zeros(len(t));folds=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=2026)
 for k,(it,iv) in enumerate(folds.split(np.zeros(len(t)),y[t],g[t])):
  oof[iv]=text_predict(x.iloc[t[it]],y[t[it]],x.iloc[t[iv]])
  print('TEXT OOF FOLD',k,'DONE',flush=True)
 pred=text_predict(x.iloc[t],y[t],x.iloc[v])
 record('logistic_text_oof_stack_base',pred,y[v])
 tr['text_oof_probability']=oof;va['text_oof_probability']=pred
 np.save(OUT/'text_stack_oof.npy',oof)
 gc.collect()
 for name,params in [('cat_rates_stack_d7',dict(iterations=1600,depth=7,learning_rate=.035,l2_leaf_reg=10,random_seed=51)),('cat_rates_stack_d8',dict(iterations=1600,depth=8,learning_rate=.035,l2_leaf_reg=12,random_seed=73))]:
  model=CatBoostClassifier(**params,loss_function='Logloss',eval_metric='Accuracy',thread_count=8,allow_writing_files=False,early_stopping_rounds=150,one_hot_max_size=10,random_strength=.5)
  ts=time.time();model.fit(tr,y[t],cat_features=cats,eval_set=(va,y[v]),verbose=200)
  record(name,model.predict_proba(va)[:,1],y[v],seconds=time.time()-ts,best_iteration=model.best_iteration_)
  model.save_model(str(OUT/(name+'.cbm')));(OUT/(name+'_params.json')).write_text(json.dumps(params));del model;gc.collect()
 # Different split behavior to complement CatBoost.
 for c in cats:
  levels=pd.Index(tr[c].unique());dtype=pd.CategoricalDtype(categories=levels)
  tr[c]=tr[c].astype(dtype);va[c]=va[c].astype(dtype)
 for leaves in [31,63]:
  model=LGBMClassifier(n_estimators=2000,num_leaves=leaves,learning_rate=.035,max_depth=-1,min_child_samples=40,colsample_bytree=.8,subsample=.85,subsample_freq=1,reg_lambda=8,max_cat_threshold=32,cat_smooth=20,n_jobs=8,verbosity=-1,random_state=42)
  ts=time.time();model.fit(tr,y[t],eval_set=[(va,y[v])],eval_metric='binary_logloss',callbacks=[early_stopping(100,verbose=False),log_evaluation(200)])
  record(f'lightgbm_stack_leaves{leaves}',model.predict_proba(va)[:,1],y[v],seconds=time.time()-ts,best_iteration=model.best_iteration_)

if __name__=='__main__':main()
