"""Freeze an ensemble on selection; check another family split and refit."""
from pathlib import Path
import argparse,json,itertools
import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from competition_pipeline_v2 import load_data,groups_for,matrices,metrics
from optimize_leaderboard import split,enrich
from relation_features import augment as relation_augment
from char_projection_round import project
from ablation_round import augment as ablation_augment
from cat_native_round import cat_frames
from sparse_text_round import transform as sparse_transform
ROOT=Path(__file__).resolve().parents[1]
def prepare_family(feature,tr,va,raw,y,g,other):
    if feature=='relations':return relation_augment(tr,va,raw,y,g,other)
    if feature=='char':return project(tr,va,raw,other)
    if feature=='ablation':return ablation_augment(tr,va,raw,y,g,other)
    if feature=='sparse':return sparse_transform(tr,va,raw,other)
    return tr,va
def fit(config,tr,y,va,raw,other):
    if config.get('kind')=='cat':
        a,b,cats=cat_frames(tr,va);x,_=enrich(raw);z,_=enrich(other)
        a['full_text']=x.full_text.to_numpy();b['full_text']=z.full_text.to_numpy()
        model=CatBoostClassifier(**config['params'],loss_function='Logloss',eval_metric='Accuracy',thread_count=8,
          allow_writing_files=False,one_hot_max_size=10,random_strength=.5)
        model.fit(a,y,cat_features=cats,text_features=['full_text'],verbose=300)
        return model,model.predict_proba(b)[:,1]
    model=LGBMClassifier(**config['params']);model.fit(tr,y);return model,model.predict_proba(va)[:,1]
def select(folder,out):
    out=Path(out);raw,test,audits=load_data(folder,out);y=(raw.condition=='used').to_numpy(dtype='int8');g=groups_for(raw);t,v=split(y,g,42,2)
    records=[]
    for path in out.glob('*.json'):
        try:r=json.loads(path.read_text())
        except json.JSONDecodeError:continue
        if isinstance(r,dict) and 'name' in r and 'params' in r and (out/(r['name']+'.npy')).exists():
            r.setdefault('feature_set','context_svm');records.append(r)
    records.sort(key=lambda r:r['accuracy'],reverse=True)
    if not records:raise ValueError('No hay candidatos completos.')
    predictions={r['name']:np.load(out/(r['name']+'.npy')) for r in records}
    for p in predictions.values():assert len(p)==len(v)
    best=records[0];chosen=[best];weights=[1.];threshold=best['threshold'];score=best['accuracy']
    top=records[:10]
    for pair in itertools.combinations(top,2):
        for w in [.2,.4,.6,.8]:
            p=w*predictions[pair[0]['name']]+(1-w)*predictions[pair[1]['name']]
            for q in [.47,.48,.49,.5,.51,.52,.53]:
                acc=accuracy_score(y[v],p>=q)
                if acc>score:score=float(acc);chosen=list(pair);weights=[w,1-w];threshold=q
    original=pd.read_csv(ROOT/'optimization/holdout_predictions.csv')
    old=original.set_index('row_index').loc[v]
    assert np.array_equal(old.y_used.to_numpy(),y[v])
    baseline=float(accuracy_score(y[v],old.probability_used>=.5))
    selected=dict(configs=chosen,weights=weights,threshold=threshold,selection_accuracy=score,baseline_same_rows=baseline,
                  baseline_confirmation=.9421428571428572,confirmation=None,status='selected_pending_confirmation',goal_met_local=False)
    (out/'final_ensemble.json').write_text(json.dumps(selected,indent=2))
    pd.DataFrame([{k:v for k,v in r.items() if k!='params'} for r in records]).to_csv(out/'all_candidates.csv',index=False)
    print('FROZEN',json.dumps(selected),flush=True)
    return raw,test,y,g,selected
def finalize(folder,out,refit=True):
    out=Path(out);raw,test,y,g,selected=select(folder,out);t,v=split(y,g,731,2)
    a,b=raw.iloc[t].reset_index(drop=True),raw.iloc[v].reset_index(drop=True)
    tr,va,_=matrices(a,y[t],g[t],b,out);transforms={};probs=[]
    for config in selected['configs']:
        kind=config.get('feature_set','context_svm')
        if kind not in transforms:transforms[kind]=prepare_family(kind,tr,va,a,y[t],g[t],b)
        fit_tr,fit_va=transforms[kind];_,p=fit(config,fit_tr,y[t],fit_va,a,b);probs.append(p)
    combined=sum(w*p for w,p in zip(selected['weights'],probs));q=selected['threshold']
    selected['confirmation']=metrics(y[v],combined,q)
    selected['promote_over_previous']=(selected['selection_accuracy']>selected['baseline_same_rows'] and selected['confirmation']['accuracy']>selected['baseline_confirmation'])
    selected['goal_met_local']=selected['selection_accuracy']>=.95 and selected['confirmation']['accuracy']>=.95
    selected['status']='confirmed'
    (out/'final_ensemble.json').write_text(json.dumps(selected,indent=2))
    pd.DataFrame({'row_index':v,'y_used':y[v],'probability_used':combined}).to_csv(out/'final_confirmation_predictions.csv',index=False)
    print('FINAL CONFIRMATION',selected['confirmation'],flush=True)
    if refit and selected['promote_over_previous']:
        a,b=raw.reset_index(drop=True),test.reset_index(drop=True);tr,va,_=matrices(a,y,g,b,out)
        transforms={};probs=[]
        for k,config in enumerate(selected['configs']):
            kind=config.get('feature_set','context_svm')
            if kind not in transforms:transforms[kind]=prepare_family(kind,tr,va,a,y,g,b)
            fit_tr,fit_va=transforms[kind];model,p=fit(config,fit_tr,y,fit_va,a,b);probs.append(p)
            if config.get('kind')=='cat':model.save_model(str(out/f'final_model_{k}.cbm'))
            else:model.booster_.save_model(str(out/f'final_model_{k}.txt'))
        combined=sum(w*p for w,p in zip(selected['weights'],probs))
        sub=pd.DataFrame({'ID':np.arange(1,len(test)+1),'condition':np.where(combined>=q,'used','new')})
        assert len(sub)==30000 and not sub.isna().any().any() and sub.ID.is_unique
        sub.to_csv(out/'submission_final_ensemble.csv',index=False)
    return selected
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',default=str(ROOT/'results/round_v2'))
    p.add_argument('--select-only',action='store_true');p.add_argument('--no-refit',action='store_true');a=p.parse_args()
    if a.select_only:select(a.data_dir,a.output)
    else:finalize(a.data_dir,a.output,not a.no_refit)

