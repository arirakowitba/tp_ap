import json,time
import pandas as pd
from catboost import CatBoostClassifier
from optimize_leaderboard import OUT,record
tr,va,y,yv=pd.read_pickle(OUT/'stack_matrix_cache.pkl')
cats=tr.select_dtypes(include='category').columns.tolist()
for c in cats:tr[c]=tr[c].astype('string').fillna('__MISSING__').astype(str);va[c]=va[c].astype('string').fillna('__MISSING__').astype(str)
name='cat_pair_ctr_d6'
params=dict(iterations=1800,depth=6,learning_rate=.04,l2_leaf_reg=10,random_seed=99,max_ctr_complexity=2,ctr_leaf_count_limit=30000)
m=CatBoostClassifier(**params,loss_function='Logloss',eval_metric='Accuracy',thread_count=8,allow_writing_files=False,early_stopping_rounds=150,one_hot_max_size=10,random_strength=.5)
ts=time.time();m.fit(tr,y,cat_features=cats,eval_set=(va,yv),verbose=200)
record(name,m.predict_proba(va)[:,1],yv,seconds=time.time()-ts,best_iteration=m.best_iteration_)
(OUT/(name+'_params.json')).write_text(json.dumps(params));m.save_model(str(OUT/(name+'.cbm')))
