import json,time,gc
import numpy as np,pandas as pd
from lightgbm import LGBMClassifier,early_stopping
from xgboost import XGBClassifier
from optimize_leaderboard import OUT,prepare,numeric_text,record
from optimize_stack import group_rates

def main():
 d=prepare();x,y,t,v,g=d['x'],d['y'],d['t'],d['v'],d['groups']
 tr,va,cats,tf,svd=numeric_text(x,t,v);tr,va=group_rates(tr,va,y[t],g[t])
 tr['text_oof_probability']=np.load(OUT/'text_stack_oof.npy');va['text_oof_probability']=np.load(OUT/'logistic_text_oof_stack_base_selection_seed42.npy')
 for c in cats:
  dtype=pd.CategoricalDtype(categories=pd.Index(tr[c].unique()));tr[c]=tr[c].astype(dtype);va[c]=va[c].astype(dtype)
 pd.to_pickle((tr,va,y[t],y[v]),OUT/'stack_matrix_cache.pkl')
 configs=[(15,20,2,1.),(31,10,2,1.),(63,20,2,1.),(127,30,5,.9),(31,60,20,.7),(63,60,20,.7),(127,80,20,.8),(255,40,10,.8)]
 for i,(leaves,child,reg,cols) in enumerate(configs):
  name=f'lgb_fast_{i}';params=dict(n_estimators=2000,num_leaves=leaves,learning_rate=.03,max_depth=-1,min_child_samples=child,colsample_bytree=cols,subsample=.85,subsample_freq=1,reg_lambda=reg,max_cat_threshold=64,cat_smooth=10,n_jobs=8,verbosity=-1,random_state=100+i)
  m=LGBMClassifier(**params);ts=time.time();m.fit(tr,y[t],eval_set=[(va,y[v])],eval_metric='binary_logloss',callbacks=[early_stopping(100,verbose=False)])
  record(name,m.predict_proba(va)[:,1],y[v],seconds=time.time()-ts,best_iteration=m.best_iteration_)
  (OUT/(name+'_params.json')).write_text(json.dumps(params));del m;gc.collect()
 for depth in [5,8]:
  name=f'xgb_native_d{depth}';params=dict(n_estimators=1500,max_depth=depth,learning_rate=.04,min_child_weight=15,reg_lambda=10,subsample=.85,colsample_bytree=.85,max_cat_threshold=64,enable_categorical=True,tree_method='hist',n_jobs=8,random_state=73,early_stopping_rounds=100,eval_metric='logloss')
  m=XGBClassifier(**params);ts=time.time();m.fit(tr,y[t],eval_set=[(va,y[v])],verbose=300)
  record(name,m.predict_proba(va)[:,1],y[v],seconds=time.time()-ts,best_iteration=m.best_iteration)
  (OUT/(name+'_params.json')).write_text(json.dumps(params));del m;gc.collect()

if __name__=='__main__':main()
