"""Grouped search, integrity audit and frozen refit for the Mercado Libre TP."""
from pathlib import Path
import argparse, hashlib, json, os, time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from lightgbm import LGBMClassifier, early_stopping
import optuna
from optimize_leaderboard import enrich, numeric_text, split
from optimize_stack import group_rates

ROOT=Path(__file__).resolve().parents[1]

def audit(path, expected):
    count=0; errors=[]; digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for number,line in enumerate(f,1):
            digest.update(line)
            try: json.loads(line);count+=1
            except (json.JSONDecodeError,UnicodeDecodeError) as exc: errors.append({'line':number,'error':str(exc)})
    return dict(filename=Path(path).name,expected_rows=expected,valid_rows=count,invalid_lines=errors,sha256=digest.hexdigest(),complete=count==expected and not errors)

def load_data(folder,out,allow_partial=False):
    folder,out=Path(folder),Path(out);out.mkdir(parents=True,exist_ok=True)
    audits=[audit(folder/'train_data.jsonlines',70000),audit(folder/'test_data.jsonlines',30000)]
    (out/'data_audit.json').write_text(json.dumps(audits,indent=2))
    if not audits[1]['complete'] or (not audits[0]['complete'] and not allow_partial):
        raise ValueError('Datos incompletos: revisar data_audit.json. Se requiere train íntegro de 70.000 filas.')
    def read(path):
        rows=[]
        with path.open() as f:
            for line in f:
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:
                    if not allow_partial:raise
        return pd.DataFrame(rows)
    raw,test=read(folder/'train_data.jsonlines'),read(folder/'test_data.jsonlines')
    assert set(raw.condition.unique())=={'new','used'} and 'condition' not in test
    return raw,test,audits

def groups_for(raw):
    return raw.parent_item_id.astype('string').fillna('__ROW__'+raw.id.astype(str)).to_numpy()

def svm_probability(a,y,b,C=.5):
    word=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=160000,sublinear_tf=True,strip_accents='unicode',dtype=np.float32)
    char=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=3,max_features=140000,sublinear_tf=True,dtype=np.float32)
    cols=['seller_id','category_id','listing_type_id','shipping_mode','title_first']
    enc=OneHotEncoder(handle_unknown='ignore',min_frequency=3,dtype=np.float32)
    fit=sparse.hstack([word.fit_transform(a.full_text),char.fit_transform(a.title_text+' '+a.warranty_text),enc.fit_transform(a[cols].astype(str))],format='csr')
    val=sparse.hstack([word.transform(b.full_text),char.transform(b.title_text+' '+b.warranty_text),enc.transform(b[cols].astype(str))],format='csr')
    model=LinearSVC(C=C,dual='auto',max_iter=3000,random_state=42);model.fit(fit,y)
    return expit(2*model.decision_function(val))

def matrices(raw,y,groups,other,out):
    from ap2026q2_advanced_optimized import _attributes,build_advanced_features
    a,_=enrich(raw);b,_=enrich(other)
    attrs,ids=_attributes(raw,top_k=250)
    extra,_=build_advanced_features(other,attribute_ids=ids)
    a=a.drop(columns=[c for c in a if c.startswith('attr_')])
    b=b.drop(columns=[c for c in b if c.startswith('attr_')])
    a=pd.concat([a.reset_index(drop=True),attrs.reset_index(drop=True)],axis=1)
    b=pd.concat([b.reset_index(drop=True),extra[[f'attr_{i}' for i in ids]].reset_index(drop=True)],axis=1)
    for col in ['category_id','seller_id','seller_category']:
        stats=pd.DataFrame({'key':a[col].astype(str),'price':a.price_raw}).groupby('key').price.agg(['median','count'])
        for x in [a,b]:
            key=x[col].astype(str);median=key.map(stats['median']).fillna(a.price_raw.median())
            x[col+'_price_relative']=np.log1p(x.price_raw.clip(lower=0))-np.log1p(median.clip(lower=0))
            x[col+'_frequency']=np.log1p(key.map(stats['count']).fillna(0))
    patterns={'like_new':r'como nuev|impecable|poco uso','defect':r'reparar|repuesto|roto|no funciona',
              'retail':r'por mayor|mayorista|stock|distribuidor','personal':r'vendo|permuto|escucho',
              'sealed':r'sellad|sin abrir|a estrenar|sin uso'}
    for x in [a,b]:
        for name,pattern in patterns.items():x['lex_'+name]=x.title_text.str.contains(pattern,case=False,regex=True).astype('int8')
    both=pd.concat([a,b],ignore_index=True);t=np.arange(len(a));v=np.arange(len(a),len(both))
    tr,va,cats,_,_=numeric_text(both,t,v)
    tr,va=group_rates(tr,va,y,groups)
    oof=np.zeros(len(y))
    for k,(it,iv) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=2026).split(np.zeros(len(y)),y,groups)):
        assert not set(groups[it])&set(groups[iv])
        print('SVM OOF',k,flush=True)
        oof[iv]=svm_probability(a.iloc[it],y[it],a.iloc[iv])
    tr['svm_oof_probability']=oof;va['svm_oof_probability']=svm_probability(a,y,b)
    for c in cats:
        dtype=pd.CategoricalDtype(pd.Index(tr[c].astype(str).unique()))
        tr[c]=tr[c].astype(str).astype(dtype);va[c]=va[c].astype(str).astype(dtype)
    return tr,va,cats

def metrics(y,p,q=.5):
    return dict(accuracy=float(accuracy_score(y,p>=q)),balanced_accuracy=float(balanced_accuracy_score(y,p>=q)),
                macro_f1=float(f1_score(y,p>=q,average='macro')),logloss=float(log_loss(y,p)),threshold=float(q))

def search(folder,out=None,trials=24,target=.95,allow_partial=False,refit=True):
    out=Path(out or ROOT/'results/round_v2');out.mkdir(parents=True,exist_ok=True)
    raw,test,audits=load_data(folder,out,allow_partial)
    y=(raw.condition=='used').to_numpy(dtype=np.int8);g=groups_for(raw);t,v=split(y,g,42,2)
    pd.DataFrame({'row_index':v,'y_used':y[v]}).to_csv(out/'selection_rows.csv',index=False)
    cache=out/'matrices.pkl';fingerprint=audits[0]['sha256']
    if cache.exists():
        d=pd.read_pickle(cache)
        if d['fingerprint']!=fingerprint:raise ValueError('Cache de otro dataset: cambiar carpeta output.')
        tr,va=d['tr'],d['va']
    else:
        tr,va,_=matrices(raw.iloc[t].reset_index(drop=True),y[t],g[t],raw.iloc[v].reset_index(drop=True),out)
        pd.to_pickle(dict(tr=tr,va=va,fingerprint=fingerprint),cache)
    rows=[];configs=[];preds=[]
    base=dict(n_estimators=2400,learning_rate=.025,num_leaves=63,min_child_samples=35,reg_lambda=3,
              colsample_bytree=.9,subsample=.9,subsample_freq=1,cat_smooth=20,max_cat_threshold=16,
              n_jobs=4,verbosity=-1,random_state=2026)
    def fit(params,name):
        ts=time.monotonic();m=LGBMClassifier(**params)
        m.fit(tr,y[t],eval_set=[(va,y[v])],eval_metric='binary_logloss',callbacks=[early_stopping(100,verbose=False)])
        p=m.predict_proba(va)[:,1];q=float(max(np.arange(.46,.541,.005),key=lambda q:accuracy_score(y[v],p>=q)))
        config=params.copy();config['n_estimators']=int(m.best_iteration_)
        row=dict(name=name,**metrics(y[v],p,q),seconds=time.monotonic()-ts,params=config)
        rows.append(row);configs.append(config);preds.append(p)
        np.save(out/(name+'.npy'),p);(out/(name+'.json')).write_text(json.dumps(row,indent=2))
        pd.DataFrame([{k:v for k,v in r.items() if k!='params'} for r in rows]).to_csv(out/'metrics.csv',index=False)
        pd.DataFrame({'feature':tr.columns,'gain':m.booster_.feature_importance('gain')}).sort_values('gain',ascending=False).to_csv(out/(name+'_importance.csv'),index=False)
        print('RESULT',name,row['accuracy'],flush=True);return row['accuracy']
    fit(base,'lgb_context_svm_base')
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=1010,n_startup_trials=6))
    def objective(trial):
        params=base.copy();params.update(learning_rate=trial.suggest_float('learning_rate',.018,.065,log=True),
          num_leaves=trial.suggest_int('num_leaves',24,120),min_child_samples=trial.suggest_int('min_child_samples',15,100),
          reg_lambda=trial.suggest_float('reg_lambda',.1,25,log=True),reg_alpha=trial.suggest_float('reg_alpha',.001,5,log=True),
          colsample_bytree=trial.suggest_float('colsample_bytree',.65,1),cat_smooth=trial.suggest_float('cat_smooth',5,80,log=True),
          max_cat_threshold=trial.suggest_categorical('max_cat_threshold',[8,16,32]))
        return fit(params,f'lgb_context_svm_tpe_{trial.number}')
    def stop(study,trial):
        if study.best_value>=target:study.stop()
    if max(r['accuracy'] for r in rows)<target:study.optimize(objective,n_trials=trials,callbacks=[stop])
    study.trials_dataframe().to_csv(out/'tpe_trials.csv',index=False)
    i=int(np.argmax([r['accuracy'] for r in rows]));chosen=[i];weights=[1.];q=rows[i]['threshold'];score=rows[i]['accuracy']
    import itertools
    top=np.argsort([-r['accuracy'] for r in rows])[:5]
    for ia,ib in itertools.combinations(top,2):
        for w in [.25,.5,.75]:
            p=w*preds[ia]+(1-w)*preds[ib]
            for threshold in [.48,.49,.5,.51,.52]:
                accuracy=accuracy_score(y[v],p>=threshold)
                if accuracy>score:chosen=[int(ia),int(ib)];weights=[w,1-w];q=threshold;score=float(accuracy)
    selected=dict(names=[rows[i]['name'] for i in chosen],params=[configs[i] for i in chosen],weights=weights,
                  threshold=q,selection_accuracy=score,partial_data=not audits[0]['complete'],training_rows=len(raw),
                  target=target,confirmation=None,goal_met_local=False,status='selected_pending_confirmation')
    (out/'selected.json').write_text(json.dumps(selected,indent=2))
    p=sum(w*preds[i] for w,i in zip(weights,chosen))
    pd.DataFrame({'row_index':v,'y_used':y[v],'probability_used':p}).to_csv(out/'holdout_predictions.csv',index=False)
    tt,vv=split(y,g,731,2)
    at,av,_=matrices(raw.iloc[tt].reset_index(drop=True),y[tt],g[tt],raw.iloc[vv].reset_index(drop=True),out)
    pp=[]
    for params in selected['params']:
        model=LGBMClassifier(**params);model.fit(at,y[tt]);pp.append(model.predict_proba(av)[:,1])
    combined=sum(w*p for w,p in zip(weights,pp));selected['confirmation']=metrics(y[vv],combined,q)
    pd.DataFrame({'row_index':vv,'y_used':y[vv],'probability_used':combined}).to_csv(out/'confirmation_predictions.csv',index=False)
    selected['goal_met_local']=not selected['partial_data'] and score>=target and selected['confirmation']['accuracy']>=target
    selected['status']='partial_exploration' if selected['partial_data'] else 'confirmed'
    (out/'selected.json').write_text(json.dumps(selected,indent=2))
    if refit and not selected['partial_data']:
        ft,fv,_=matrices(raw,y,g,test,out);pp=[]
        for k,params in enumerate(selected['params']):
            model=LGBMClassifier(**params);model.fit(ft,y);pp.append(model.predict_proba(fv)[:,1])
            model.booster_.save_model(str(out/f'model_full_{k}.txt'))
        p=sum(w*p for w,p in zip(weights,pp))
        submission=pd.DataFrame({'ID':np.arange(1,len(test)+1),'condition':np.where(p>=q,'used','new')})
        assert len(submission)==30000 and not submission.isna().any().any() and submission.ID.is_unique
        submission.to_csv(out/'submission_v2.csv',index=False)
    print('FINAL',json.dumps(selected),flush=True);return selected

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',default=os.getenv('AP2026Q2_DATA_DIR'))
    p.add_argument('--output',default=str(ROOT/'results/round_v2'));p.add_argument('--trials',type=int,default=24)
    p.add_argument('--allow-partial',action='store_true');p.add_argument('--no-refit',action='store_true')
    args=p.parse_args()
    if not args.data_dir:p.error('Configurar --data-dir o AP2026Q2_DATA_DIR')
    search(args.data_dir,args.output,args.trials,allow_partial=args.allow_partial,refit=not args.no_refit)
