"""Direct sparse text + one-hot categories + numeric metadata for LightGBM."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier,early_stopping
from competition_pipeline_v2 import load_data,groups_for,metrics
from optimize_leaderboard import split,enrich
ROOT=Path(__file__).resolve().parents[1]
def transform(tr,va,raw,other):
    a,_=enrich(raw);b,_=enrich(other)
    word=TfidfVectorizer(ngram_range=(1,2),min_df=5,max_features=20000,sublinear_tf=True,strip_accents='unicode',dtype=np.float32)
    cats=tr.select_dtypes(include=['category','string','object']).columns.tolist()
    enc=OneHotEncoder(handle_unknown='ignore',min_frequency=15,dtype=np.float32)
    numeric=tr.select_dtypes(include=np.number).columns
    med=tr[numeric].median().fillna(0)
    fit=sparse.hstack([sparse.csr_matrix(tr[numeric].fillna(med).astype('float32').to_numpy()),
                      enc.fit_transform(tr[cats].astype(str)),word.fit_transform(a.full_text)],format='csr')
    val=sparse.hstack([sparse.csr_matrix(va[numeric].fillna(med).astype('float32').to_numpy()),
                      enc.transform(va[cats].astype(str)),word.transform(b.full_text)],format='csr')
    return fit,val
def run(folder,out):
    out=Path(out);raw,_,_=load_data(folder,out);y=(raw.condition=='used').to_numpy(dtype='int8');g=groups_for(raw);t,v=split(y,g,42,2)
    d=pd.read_pickle(out/'matrices.pkl');tr,va=transform(d['tr'],d['va'],raw.iloc[t].reset_index(drop=True),raw.iloc[v].reset_index(drop=True))
    pd.to_pickle(dict(tr=tr,va=va,fingerprint=d['fingerprint']),out/'sparse_matrices.pkl')
    rows=[]
    for i,(leaves,child,cols) in enumerate([(31,35,.9),(63,40,.9),(95,60,.8),(31,80,.8)]):
        params=dict(n_estimators=2000,learning_rate=.03,num_leaves=leaves,min_child_samples=child,reg_lambda=5,
          colsample_bytree=cols,subsample=.9,subsample_freq=1,n_jobs=4,verbosity=-1,random_state=4026+i)
        model=LGBMClassifier(**params);model.fit(tr,y[t],eval_set=[(va,y[v])],eval_metric='binary_logloss',callbacks=[early_stopping(100,verbose=False)])
        p=model.predict_proba(va)[:,1];q=float(max(np.arange(.46,.541,.005),key=lambda q:accuracy_score(y[v],p>=q)))
        params['n_estimators']=int(model.best_iteration_);row=dict(name=f'lgb_sparse_text_{i}',**metrics(y[v],p,q),params=params,feature_set='sparse')
        rows.append(row);np.save(out/(row['name']+'.npy'),p);(out/(row['name']+'.json')).write_text(json.dumps(row,indent=2))
        pd.DataFrame([{k:v for k,v in r.items() if k!='params'} for r in rows]).to_csv(out/'sparse_metrics.csv',index=False)
        print('SPARSE_TEXT',row['name'],row['accuracy'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',default=str(ROOT/'results/round_v2'))
    a=p.parse_args();run(a.data_dir,a.output)

