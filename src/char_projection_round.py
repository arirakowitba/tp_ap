"""Character SVD captures text structure beyond a single SVM score."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier,early_stopping
from competition_pipeline_v2 import load_data,groups_for,metrics
from optimize_leaderboard import split
ROOT=Path(__file__).resolve().parents[1]
def project(tr,va,raw,other):
    a=raw.title.fillna('').str.lower()+' '+raw.warranty.fillna('').astype(str).str.lower()
    b=other.title.fillna('').str.lower()+' '+other.warranty.fillna('').astype(str).str.lower()
    tf=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=3,max_features=100000,dtype=np.float32,sublinear_tf=True)
    matrix=tf.fit_transform(a);eval_matrix=tf.transform(b);svd=TruncatedSVD(128,n_iter=4,random_state=42)
    fit=svd.fit_transform(matrix).astype('float32');val=svd.transform(eval_matrix).astype('float32')
    names=[f'char_svd_{i}' for i in range(128)]
    return pd.concat([tr.reset_index(drop=True),pd.DataFrame(fit,columns=names)],axis=1),pd.concat([va.reset_index(drop=True),pd.DataFrame(val,columns=names)],axis=1)
def run(folder,out):
    out=Path(out);raw,_,_=load_data(folder,out);y=(raw.condition=='used').to_numpy(dtype='int8');g=groups_for(raw);t,v=split(y,g,42,2)
    d=pd.read_pickle(out/'matrices.pkl');tr,va=project(d['tr'],d['va'],raw.iloc[t].reset_index(drop=True),raw.iloc[v].reset_index(drop=True))
    pd.to_pickle(dict(tr=tr,va=va,fingerprint=d['fingerprint']),out/'char_matrices.pkl')
    rows=[]
    for i,(leaves,child,reg,cols) in enumerate([(31,40,3,.9),(63,35,3,.9),(106,88,.34,.7),(31,80,5,.85),(63,60,10,.8),(94,50,5,.8)]):
        params=dict(n_estimators=2300,learning_rate=.025,num_leaves=leaves,min_child_samples=child,reg_lambda=reg,
          colsample_bytree=cols,subsample=.92,subsample_freq=1,cat_smooth=25,max_cat_threshold=32,n_jobs=4,verbosity=-1,random_state=3026+i)
        model=LGBMClassifier(**params);model.fit(tr,y[t],eval_set=[(va,y[v])],eval_metric='binary_logloss',callbacks=[early_stopping(100,verbose=False)])
        p=model.predict_proba(va)[:,1];q=float(max(np.arange(.46,.541,.005),key=lambda q:accuracy_score(y[v],p>=q)))
        params['n_estimators']=int(model.best_iteration_);row=dict(name=f'lgb_char_svd_{i}',**metrics(y[v],p,q),params=params,feature_set='char')
        rows.append(row);np.save(out/(row['name']+'.npy'),p);(out/(row['name']+'.json')).write_text(json.dumps(row,indent=2))
        pd.DataFrame([{k:v for k,v in r.items() if k!='params'} for r in rows]).to_csv(out/'char_metrics.csv',index=False)
        print('CHAR_SVD',row['name'],row['accuracy'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',default=str(ROOT/'results/round_v2'))
    a=p.parse_args();run(a.data_dir,a.output)

