"""Ablation: remove rare attributes and replace the SVM stack by logistic text."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier,early_stopping
from competition_pipeline_v2 import load_data,groups_for,metrics
from optimize_leaderboard import split,enrich
from optimize_stack import text_predict
from ap2026q2_advanced_optimized import _attributes
ROOT=Path(__file__).resolve().parents[1]
def augment(tr,va,raw,y,g,other):
    _,ids=_attributes(raw,top_k=100);allowed={f'attr_{i}' for i in ids}
    keep=[c for c in tr if (not c.startswith('attr_') or c in allowed) and c!='svm_oof_probability' and not c.startswith('lex_') and not c.endswith(('_price_relative','_frequency'))]
    tr,va=tr[keep].copy(),va[keep].copy()
    a,_=enrich(raw);b,_=enrich(other);oof=np.zeros(len(y))
    for k,(it,iv) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=2026).split(np.zeros(len(y)),y,g)):
        print('LOGISTIC OOF',k,flush=True);oof[iv]=text_predict(a.iloc[it],y[it],a.iloc[iv])
    tr['text_oof_probability']=oof;va['text_oof_probability']=text_predict(a,y,b)
    return tr,va
def run(folder,out):
    out=Path(out);raw,_,_=load_data(folder,out);y=(raw.condition=='used').to_numpy(dtype='int8');g=groups_for(raw);t,v=split(y,g,42,2)
    d=pd.read_pickle(out/'matrices.pkl');tr,va=augment(d['tr'],d['va'],raw.iloc[t].reset_index(drop=True),y[t],g[t],raw.iloc[v].reset_index(drop=True))
    pd.to_pickle(dict(tr=tr,va=va,fingerprint=d['fingerprint']),out/'ablation_matrices.pkl')
    original=json.loads((ROOT/'optimization/lgb_tpe_2_params.json').read_text());original['n_jobs']=4
    configs=[original]
    for leaves,child,reg in [(31,40,3),(63,35,3),(106,88,.34),(31,80,5),(63,60,10)]:
        p=original.copy();p.update(num_leaves=leaves,min_child_samples=child,reg_lambda=reg,n_estimators=2500);configs.append(p)
    rows=[]
    for i,params in enumerate(configs):
        model=LGBMClassifier(**params);model.fit(tr,y[t],eval_set=[(va,y[v])],eval_metric='binary_logloss',callbacks=[early_stopping(100,verbose=False)])
        p=model.predict_proba(va)[:,1];q=float(max(np.arange(.46,.541,.005),key=lambda q:accuracy_score(y[v],p>=q)))
        params['n_estimators']=int(model.best_iteration_);row=dict(name=f'lgb_ablation_{i}',**metrics(y[v],p,q),params=params,feature_set='ablation')
        rows.append(row);np.save(out/(row['name']+'.npy'),p);(out/(row['name']+'.json')).write_text(json.dumps(row,indent=2))
        pd.DataFrame([{k:v for k,v in r.items() if k!='params'} for r in rows]).to_csv(out/'ablation_metrics.csv',index=False)
        print('ABLATION',row['name'],row['accuracy'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',default=str(ROOT/'results/round_v2'))
    a=p.parse_args();run(a.data_dir,a.output)

