"""Additional similarity-key signals, fitted out of fold by family."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier,early_stopping
from optimize_leaderboard import split
from competition_pipeline_v2 import load_data,groups_for,metrics
from ap2026q2_deep_features import parse_nested
ROOT=Path(__file__).resolve().parents[1]
def keys(raw):
    x=pd.DataFrame(index=np.arange(len(raw)))
    title=raw.title.fillna('').str.lower().str.normalize('NFKD').str.encode('ascii',errors='ignore').str.decode('ascii').str.replace(r'[^a-z0-9 ]',' ',regex=True).str.replace(r'\s+',' ',regex=True).str.strip()
    x['title_key']=title.to_numpy()
    x['title_first3']=title.str.split().map(lambda z:' '.join(z[:3])).to_numpy()
    x['title_first4']=title.str.split().map(lambda z:' '.join(z[:4])).to_numpy()
    x['seller_title3']=raw.seller_id.astype(str).to_numpy()+'|'+x.title_first3
    x['category_title3']=raw.category_id.astype(str).to_numpy()+'|'+x.title_first3
    x['seller_warranty']=raw.seller_id.astype(str).to_numpy()+'|'+raw.warranty.fillna('').astype(str).to_numpy()
    x['primary_picture']=raw.pictures.map(lambda z: str((parse_nested(z) or [{}])[0].get('id','')) if isinstance(parse_nested(z),list) and len(parse_nested(z)) else '').to_numpy()
    x['seller_picture']=raw.seller_id.astype(str).to_numpy()+'|'+x.primary_picture
    for c in ['title_key','title_first3','title_first4','primary_picture']:x[c]=x[c].replace('','__MISSING__')
    return x
def augment(tr,va,raw,y,g,other):
    a,b=keys(raw),keys(other);tr,va=tr.copy(),va.copy()
    folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=2026).split(np.zeros(len(y)),y,g))
    for col in a:
        oof=np.zeros(len(y));counts=np.zeros(len(y));alpha=5
        for it,iv in folds:
            stats=pd.DataFrame({'key':a[col].iloc[it].to_numpy(),'y':y[it]}).groupby('key').y.agg(['sum','count'])
            cnt=a[col].iloc[iv].map(stats['count']).fillna(0).to_numpy()
            oof[iv]=(a[col].iloc[iv].map(stats['sum']).fillna(0).to_numpy()+alpha*y[it].mean())/(cnt+alpha);counts[iv]=np.log1p(cnt)
        stats=pd.DataFrame({'key':a[col],'y':y}).groupby('key').y.agg(['sum','count'])
        tr['relation_'+col+'_rate']=oof;tr['relation_'+col+'_count']=counts
        cnt=b[col].map(stats['count']).fillna(0).to_numpy()
        va['relation_'+col+'_rate']=(b[col].map(stats['sum']).fillna(0).to_numpy()+alpha*y.mean())/(cnt+alpha)
        va['relation_'+col+'_count']=np.log1p(cnt)
    return tr,va
def run(folder,out):
    folder,out=Path(folder),Path(out);raw,_,_=load_data(folder,out)
    y=(raw.condition=='used').to_numpy(dtype='int8');g=groups_for(raw);t,v=split(y,g,42,2)
    d=pd.read_pickle(out/'matrices.pkl')
    tr,va=augment(d['tr'],d['va'],raw.iloc[t].reset_index(drop=True),y[t],g[t],raw.iloc[v].reset_index(drop=True))
    pd.to_pickle(dict(tr=tr,va=va,fingerprint=d['fingerprint']),out/'relations_matrices.pkl')
    rows=[];configs=[(31,40,3,.9),(63,35,3,.9),(94,30,.7,.89),(31,80,5,.85),(63,60,10,.8),(94,50,5,.8)]
    for i,(leaves,child,reg,cols) in enumerate(configs):
        params=dict(n_estimators=2200,learning_rate=.025,num_leaves=leaves,min_child_samples=child,reg_lambda=reg,
                    colsample_bytree=cols,subsample=.92,subsample_freq=1,cat_smooth=25,max_cat_threshold=32,n_jobs=4,verbosity=-1,random_state=2026+i)
        model=LGBMClassifier(**params);model.fit(tr,y[t],eval_set=[(va,y[v])],eval_metric='binary_logloss',callbacks=[early_stopping(100,verbose=False)])
        p=model.predict_proba(va)[:,1];q=float(max(np.arange(.46,.541,.005),key=lambda q:accuracy_score(y[v],p>=q)))
        params['n_estimators']=model.best_iteration_
        row=dict(name=f'lgb_relations_{i}',**metrics(y[v],p,q),params=params,feature_set='relations')
        rows.append(row);np.save(out/(row['name']+'.npy'),p);(out/(row['name']+'.json')).write_text(json.dumps(row,indent=2))
        pd.DataFrame([{k:v for k,v in r.items() if k!='params'} for r in rows]).to_csv(out/'relations_metrics.csv',index=False)
        print('RELATIONS',row['name'],row['accuracy'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',default=str(ROOT/'results/round_v2'))
    a=p.parse_args();run(a.data_dir,a.output)
