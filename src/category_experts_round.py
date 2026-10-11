"""Category specialists with training-only routing and global fallback.

Outer selection uses the same family split as the other experiments. Routing uses
only training titles/category IDs. Expert iteration counts are fixed in advance;
outer holdout labels never drive individual expert early stopping.
"""
from pathlib import Path
import argparse, json, time, datetime
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from competition_pipeline_v2 import audit, groups_for, metrics
from optimize_leaderboard import split

ROOT=Path(__file__).resolve().parents[1]
GLOBAL_PARAMS=dict(n_estimators=449,learning_rate=.04,num_leaves=63,min_child_samples=35,
                   reg_lambda=3,colsample_bytree=.9,subsample=.9,subsample_freq=1,
                   cat_smooth=20,max_cat_threshold=16,n_jobs=2,verbosity=-1,random_state=2026)

def deadline_seconds(deadline):
    if deadline is None:return float('inf')
    return datetime.datetime.fromisoformat(deadline.replace('Z','+00:00')).timestamp()-time.time()

def deadline_callback(deadline):
    def check(_):
        if deadline_seconds(deadline)<=0:raise TimeoutError('Category experiment deadline reached')
    return check

def route_categories(raw_train, raw_other, clusters=None, normalize_centroids=False):
    """Fit a category router without using condition or any evaluation titles."""
    cats=raw_train.category_id.fillna('__MISSING__').astype(str).reset_index(drop=True)
    other=raw_other.category_id.fillna('__MISSING__').astype(str).reset_index(drop=True)
    if clusters is None:return cats.to_numpy(),other.to_numpy(),{'kind':'exact_category'}
    text=raw_train.title.fillna('').astype(str).reset_index(drop=True)
    vectorizer=TfidfVectorizer(min_df=3,max_features=12000,ngram_range=(1,2),strip_accents='unicode',dtype=np.float32)
    words=vectorizer.fit_transform(text)
    unique=pd.Index(cats.unique());codes=unique.get_indexer(cats)
    member=sparse.csr_matrix((np.ones(len(cats),dtype=np.float32),(codes,np.arange(len(cats)))),shape=(len(unique),len(cats)))
    counts=np.asarray(member.sum(axis=1)).ravel()
    category_text=sparse.diags(1/counts)@member@words
    if normalize_centroids:category_text=normalize(category_text,norm='l2',copy=False)
    clusterer=MiniBatchKMeans(n_clusters=min(clusters,len(unique)),random_state=2026,n_init=3,batch_size=1024)
    assignments=clusterer.fit_predict(category_text)
    mapping={str(k):int(value) for k,value in zip(unique,assignments)}
    train_route=cats.map(mapping).to_numpy(dtype=int);other_route=other.map(mapping).fillna(-1).to_numpy(dtype=int)
    summary={str(c):{'categories':int((assignments==c).sum()),'training_rows':int((train_route==c).sum()),
             'example_titles':text[train_route==c].head(8).tolist()} for c in np.unique(assignments)}
    return train_route,other_route,{'kind':'training_title_category_clusters','clusters':clusters,'normalize_centroids':normalize_centroids,'mapping':mapping,'summary':summary}

def fit_experts(tr,y,va,raw_train,raw_other,global_params,clusters=None,min_rows=300,min_class=30,deadline=None,normalize_centroids=False):
    a,b,router=route_categories(raw_train,raw_other,clusters,normalize_centroids)
    prediction=np.full(len(va),np.nan);details=[]
    for key,count in pd.Series(a).value_counts().items():
        if deadline_seconds(deadline)<20:raise TimeoutError('Category specialist deadline reached')
        it=np.flatnonzero(a==key);iv=np.flatnonzero(b==key)
        nclass=np.bincount(y[it],minlength=2)
        if count<min_rows or nclass.min()<min_class or not len(iv):continue
        params=dict(global_params);params.update(n_estimators=min(500,global_params['n_estimators']),
              num_leaves=15 if len(it)<1500 else 31,min_child_samples=max(15,min(60,len(it)//60)),
              reg_lambda=5,reg_alpha=.1,n_jobs=2)
        model=LGBMClassifier(**params);model.fit(tr.iloc[it],y[it],callbacks=[deadline_callback(deadline)])
        prediction[iv]=model.predict_proba(va.iloc[iv])[:,1]
        details.append({'route':str(key),'training_rows':len(it),'validation_rows':len(iv),'new_count':int(nclass[0]),'used_count':int(nclass[1]),'params':params})
        print('EXPERT',key,len(it),len(iv),flush=True)
    return prediction,details,router

def run(data_dir,cache,out,deadline=None,normalized_clusters_only=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    folder=Path(data_dir);audits=[audit(folder/'train_data.jsonlines',70000),audit(folder/'test_data.jsonlines',30000)]
    if not all(a['complete'] for a in audits):raise ValueError('Incomplete dataset')
    cols=['id','parent_item_id','title','category_id','condition'];records=[]
    with (folder/'train_data.jsonlines').open() as stream:
        for line in stream:
            item=json.loads(line);records.append({k:item.get(k) for k in cols})
    raw=pd.DataFrame.from_records(records);del records
    y=(raw.condition=='used').to_numpy(dtype='int8');g=groups_for(raw)
    t,v=split(y,g,42,2);pd.DataFrame({'row_index':v,'y_used':y[v]}).to_csv(out/'category_selection_rows.csv',index=False);d=pd.read_pickle(cache);tr,va=d['tr'],d['va']
    if len(tr)!=len(t) or len(va)!=len(v):raise ValueError('Shared cache split length mismatch')
    if d.get('fingerprint') not in (None,audits[0]['sha256']):raise ValueError('Cache dataset mismatch')
    if not np.array_equal(d.get('train_rows',d.get('t',t)),t):raise ValueError('Cache training row mismatch')
    if not np.array_equal(d.get('valid_rows',d.get('v',v)),v):raise ValueError('Cache validation row mismatch')
    params=dict(GLOBAL_PARAMS)
    prior=ROOT/'results/round_v2/selected.json'
    if prior.exists():params.update(json.loads(prior.read_text())['params'][0]);params['n_jobs']=2
    ts=time.monotonic();rows=[]
    if normalized_clusters_only:
        global_p=np.load(out/'category_global_baseline.npy')
        for file in sorted(out.glob('category_*.json')):
            item=json.loads(file.read_text())
            if 'accuracy' in item:rows.append(item)
    else:
        global_model=LGBMClassifier(**params);global_model.fit(tr,y[t],callbacks=[deadline_callback(deadline)])
        global_p=global_model.predict_proba(va)[:,1]
    rawt=raw.iloc[t].reset_index(drop=True);rawv=raw.iloc[v].reset_index(drop=True)
    def save(name,p,extras):
        q=float(max(np.arange(.46,.541,.005),key=lambda q:accuracy_score(y[v],p>=q)))
        row=dict(name=name,**metrics(y[v],p,q),seconds=time.monotonic()-ts,feature_set='category_experts',global_params=params,**extras)
        np.save(out/(name+'.npy'),p);(out/(name+'.json')).write_text(json.dumps(row,indent=2))
        rows.append(row);pd.DataFrame([{k:v for k,v in x.items() if k not in ['global_params','experts','router']} for x in rows]).to_csv(out/'category_metrics.csv',index=False)
        print('CATEGORY RESULT',name,row['accuracy'],flush=True)
    if not normalized_clusters_only:save('category_global_baseline',global_p,{'expert_weight':0,'covered_fraction':0,'status':'selection_only'})
    settings=[(8,1000),(16,1000)] if normalized_clusters_only else [(None,300),(None,700),(8,1000),(16,1000)]
    for clusters,min_rows in settings:
        if deadline_seconds(deadline)<120:break
        experts,details,router=fit_experts(tr,y[t],va,rawt,rawv,params,clusters,min_rows,deadline=deadline,normalize_centroids=normalized_clusters_only)
        mask=np.isfinite(experts);kind=f'exact_{min_rows}' if clusters is None else f'cluster_{clusters}'+('_l2' if normalized_clusters_only else '')
        (out/f'category_{kind}_router.json').write_text(json.dumps(router,indent=2))
        for w in [.25,.5,.75,1.]:
            p=global_p.copy();p[mask]=(1-w)*global_p[mask]+w*experts[mask]
            save(f'category_{kind}_w{w:g}',p,{'expert_weight':w,'covered_fraction':float(mask.mean()),'min_rows':min_rows,'clusters':clusters,'normalize_centroids':normalized_clusters_only,'experts':details,'router_file':f'category_{kind}_router.json','status':'selection_only'})
    (out/'category_protocol.json').write_text(json.dumps({'split_seed':42,'holdout_folds':2,'training_rows':len(t),'validation_rows':len(v),'group_overlap':len(set(g[t])&set(g[v])),'routing':'Training-only category text clustering or exact category IDs; missing/unseen/small categories fall back to global model','expert_training':'Fixed iteration count; no outer holdout early stopping','confirmation':'Required before promotion; this script only searches seed42','completed_configurations':len(rows)},indent=2))
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--cache',required=True);p.add_argument('--output',default=str(ROOT/'results/round_v3'));p.add_argument('--deadline');p.add_argument('--normalized-clusters-only',action='store_true')
    a=p.parse_args();run(a.data_dir,a.cache,a.output,a.deadline,a.normalized_clusters_only)
