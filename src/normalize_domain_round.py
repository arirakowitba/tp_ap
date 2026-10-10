"""Domain normalization experiment; every vocabulary is fitted inside grouped training folds."""
from pathlib import Path
import argparse,json,re,time,signal,unicodedata,os
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier,early_stopping
from competition_pipeline_v2 import load_data,groups_for,svm_probability,metrics
from optimize_leaderboard import enrich,split
ROOT=Path(__file__).resolve().parents[1]
RULES=[(r'\b0\s*k\s*m\b',' cero kilometro '),(r'\bs\s*[/.-]\s*uso\b|\bsinuso\b',' sin uso '),
(r'\bs\s*[/.-]\s*abrir\b',' sin abrir '),(r'\bc\s*[/.-]\s*garantia\b',' con garantia '),
(r'\bs\s*[/.-]\s*garantia\b',' sin garantia '),(r'\bp\s*[/.-]\s*repuestos?\b',' para repuestos '),
(r'\bp\s*[/.-]\s*reparar\b',' para reparar '),(r'\bnvo\b|\bnva\b',' nuevo '),
(r'\busdo\b|\busada?\b|\busados?\b',' usado '),(r'\ba\s+estrenar\b',' sin uso '),
(r'\bcomo\s+nuev[oa]\b',' como nuevo '),(r'\bimpecavle\b',' impecable '),
(r'\bfuncionando\b',' funciona '),(r'\bno\s+anda\b',' no funciona ')]
FLAGS={'zero_km':r'cero kilometro','unused':r'sin uso|sin abrir|sellad','like_new':r'como nuevo|poco uso|impecable',
 'repair':r'para reparar|para repuesto|no funciona|roto','explicit_used':r'\busado\b|segunda mano',
 'explicit_new':r'\bnuevo\b','negated_new':r'no es nuevo|no nuevo','personal':r'permuto|vendo|escucho ofertas'}
def normalize(text):
    text=''.join(c for c in unicodedata.normalize('NFKD',str(text or '').lower()) if not unicodedata.combining(c))
    for pattern,replacement in RULES:text=re.sub(pattern,replacement,text)
    return re.sub(r'\s+',' ',re.sub(r'[^\w\s]',' ',text)).strip() or 'empty'
def normalize_frame(x,raw):
    x=x.copy();title=raw.title.fillna('').map(normalize);war=raw.warranty.fillna('').map(normalize)
    x['title_text']=title;x['warranty_text']=war;x['title_first']=title.str.split().str[0]
    x['full_text']=title+' warranty '+war+' attributes '+x.attributes_text.astype(str)
    return x

def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.deadline:
        remaining=int(args.deadline-time.time())
        if remaining<=0:raise SystemExit('Deadline already passed')
        signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Hard deadline reached')))
        signal.alarm(remaining)
    raw,_,_=load_data(args.data_dir,out);y=(raw.condition=='used').to_numpy('int8');g=groups_for(raw);t,v=split(y,g,42,2)
    a,_=enrich(raw.iloc[t].reset_index(drop=True));b,_=enrich(raw.iloc[v].reset_index(drop=True))
    an=normalize_frame(a,raw.iloc[t].reset_index(drop=True));bn=normalize_frame(b,raw.iloc[v].reset_index(drop=True))
    records=[]
    def save(name,p,params,feature_set):
        q=float(max(np.arange(.46,.541,.005),key=lambda q:accuracy_score(y[v],p>=q)))
        record=dict(name=name,**metrics(y[v],p,q),params=params,feature_set=feature_set,split_seed=42,confirmation=None)
        np.save(out/(name+'.npy'),p);(out/(name+'.json')).write_text(json.dumps(record,indent=2));records.append(record)
        pd.DataFrame([{k:x for k,x in r.items() if k not in ('params','confirmation')} for r in records]).to_csv(out/'normalize_metrics.csv',index=False)
        print(name,record['accuracy'],flush=True)
    rawp=np.load(out/'normalize_svm_raw.npy') if (out/'normalize_svm_raw.npy').exists() else svm_probability(a,y[t],b)
    save('normalize_svm_raw',rawp,{'C':.5},'raw_svm')
    normp=np.load(out/'normalize_svm_domain.npy') if (out/'normalize_svm_domain.npy').exists() else svm_probability(an,y[t],bn)
    save('normalize_svm_domain',normp,{'C':.5},'normalized_svm')
    pd.DataFrame({'row_index':v,'y_used':y[v],'raw_probability':rawp,'normalized_probability':normp}).to_csv(out/'normalize_text_predictions.csv',index=False)
    examples=raw[['title']].copy();examples['normalized_title']=examples.title.fillna('').map(normalize)
    examples[examples.title.str.lower()!=examples.normalized_title].head(100).to_csv(out/'normalize_examples.csv',index=False)
    cache=Path(args.cache)/'matrices.pkl'
    if not cache.exists():
        print('No base matrices cache: completed direct raw/normalized SVM comparison only.',flush=True);return
    d=pd.read_pickle(cache);tr,va=d['tr'].copy(),d['va'].copy()
    if len(tr)!=len(t) or len(va)!=len(v):raise ValueError('Cache split shape mismatch')
    oof=np.zeros(len(t))
    for k,(it,iv) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=2026).split(np.zeros(len(t)),y[t],g[t])):
        print('NORMALIZED SVM OOF',k,flush=True);oof[iv]=svm_probability(an.iloc[it],y[t][it],an.iloc[iv])
    tr['svm_oof_probability']=oof;va['svm_oof_probability']=normp
    for name,pat in FLAGS.items():
        tr['norm_'+name]=an.title_text.str.contains(pat,regex=True).astype('int8').to_numpy()
        va['norm_'+name]=bn.title_text.str.contains(pat,regex=True).astype('int8').to_numpy()
    pd.to_pickle(dict(tr=tr,va=va,fingerprint=d.get('fingerprint')),out/'normalize_matrices.pkl')
    for i,param_path in enumerate([ROOT/'optimization/lgb_tpe_2_params.json',ROOT/'results/round_v2/lgb_context_svm_tpe_12.json']):
        if not param_path.exists():continue
        cfg=json.loads(param_path.read_text());cfg=cfg.get('params',cfg);cfg.update(n_jobs=2,n_estimators=2400)
        model=LGBMClassifier(**cfg);model.fit(tr,y[t],eval_set=[(va,y[v])],eval_metric='binary_logloss',callbacks=[early_stopping(100,verbose=False)])
        cfg['n_estimators']=int(model.best_iteration_);save('normalize_lgb_domain_'+str(i),model.predict_proba(va)[:,1],cfg,'normalized')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',default=str(ROOT/'results/round_v3'))
    p.add_argument('--cache',default=str(ROOT/'results/round_v3/cache_base'));p.add_argument('--deadline',type=float)
    args=p.parse_args()
    try:run(args)
    except TimeoutError as exc:print(str(exc),flush=True);raise SystemExit(124)
