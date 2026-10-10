"""CatBoost on complementary native text; validation families remain excluded."""
from pathlib import Path
import argparse,json,time
import numpy as np,pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from competition_pipeline_v2 import load_data,groups_for,metrics
from optimize_leaderboard import split,enrich
ROOT=Path(__file__).resolve().parents[1]
def cat_frames(tr,va):
    tr,va=tr.copy(),va.copy();cats=tr.select_dtypes(include=['category','string','object']).columns.tolist()
    for c in cats:
        tr[c]=tr[c].astype('string').fillna('__MISSING__').astype(str)
        va[c]=va[c].astype('string').fillna('__MISSING__').astype(str)
    return tr,va,cats
def run(folder,out):
    folder,out=Path(folder),Path(out);raw,_,_=load_data(folder,out);y=(raw.condition=='used').to_numpy(dtype='int8')
    groups=groups_for(raw);t,v=split(y,groups,42,2);d=pd.read_pickle(out/'matrices.pkl')
    tr,va,cats=cat_frames(d['tr'],d['va'])
    a,_=enrich(raw.iloc[t].reset_index(drop=True));b,_=enrich(raw.iloc[v].reset_index(drop=True))
    tr['full_text']=a.full_text.to_numpy();va['full_text']=b.full_text.to_numpy()
    rows=[]
    for name,params in [('cat_svm_native_d6',dict(iterations=1600,depth=6,learning_rate=.045,l2_leaf_reg=8,random_seed=2026)),
                        ('cat_svm_native_d7',dict(iterations=1400,depth=7,learning_rate=.04,l2_leaf_reg=12,random_seed=1010))]:
        ts=time.monotonic();m=CatBoostClassifier(**params,loss_function='Logloss',eval_metric='Accuracy',thread_count=8,
                                                allow_writing_files=False,one_hot_max_size=10,random_strength=.5)
        m.fit(tr,y[t],cat_features=cats,text_features=['full_text'],eval_set=(va,y[v]),early_stopping_rounds=130,verbose=200)
        p=m.predict_proba(va)[:,1];q=float(max(np.arange(.46,.541,.005),key=lambda q:accuracy_score(y[v],p>=q)))
        params['iterations']=m.best_iteration_+1;row=dict(name=name,kind='cat',feature_set='native_svm',**metrics(y[v],p,q),params=params,seconds=time.monotonic()-ts)
        rows.append(row);np.save(out/(name+'.npy'),p);(out/(name+'.json')).write_text(json.dumps(row,indent=2));m.save_model(str(out/(name+'.cbm')))
        pd.DataFrame([{k:v for k,v in r.items() if k!='params'} for r in rows]).to_csv(out/'cat_metrics.csv',index=False)
        print('CAT_NATIVE',row['name'],row['accuracy'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',default=str(ROOT/'results/round_v2'))
    a=p.parse_args();run(a.data_dir,a.output)

