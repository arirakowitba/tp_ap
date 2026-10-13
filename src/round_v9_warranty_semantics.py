"""Grouped-fold screen of explicit warranty type and duration features."""
from pathlib import Path
import json, re, time
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/round_v9_warranty_semantics'
OUT.mkdir(parents=True,exist_ok=True)
DURATION=re.compile(r'(\d+(?:[.,]\d+)?)\s*(d[ií]as?|mes(?:es)?|a[nñ]os?|weeks?|months?|years?)',re.I)

def warranty_features(raw, rows):
    text=raw.warranty.iloc[rows].fillna('').astype(str).str.lower().str.normalize('NFKD').str.encode('ascii',errors='ignore').str.decode('ascii').to_numpy()
    days=np.full(len(text),np.nan,dtype=np.float32); units=[]; kinds=[]
    no_phrases=('sin garantia','no tiene garantia','no posee garantia','sin cobertura','no incluye garantia')
    for i,value in enumerate(text):
        if any(term in value for term in no_phrases): kind='none'
        elif 'fabrica' in value or 'fabricante' in value: kind='factory'
        elif 'vendedor' in value or 'tienda' in value: kind='seller'
        elif 'oficial' in value: kind='official'
        elif value.strip(): kind='other'
        else: kind='missing'
        kinds.append(kind)
        match=DURATION.search(value)
        if match is None:
            units.append('no_duration'); continue
        amount=float(match.group(1).replace(',','.')); unit=match.group(2)
        if unit.startswith('d'): days[i]=amount; units.append('day')
        elif unit.startswith('mes'): days[i]=amount*30.4375; units.append('month')
        elif unit.startswith('a') or unit.startswith('year'): days[i]=amount*365.25; units.append('year')
        else: days[i]=amount*7; units.append('week')
    return pd.DataFrame({
        'warranty_duration_days_log':np.log1p(days),
        'warranty_duration_unit':pd.Series(units,dtype='category'),
        'warranty_semantic_type':pd.Series(kinds,dtype='category'),
        'warranty_explicit_type':np.asarray([int(k!='missing') for k in kinds],dtype=np.int8),
    })

def metrics(y,p):
    return {'accuracy':float(accuracy_score(y,p>=.5)),
            'logloss':float(log_loss(y,np.clip(p,1e-7,1-1e-7)))}

def main():
    raw=pd.read_json(ROOT/'data/ap_2026_q_2/raw/train_data.jsonlines',lines=True)
    y_all=(raw.condition=='used').to_numpy(dtype=np.int8)
    t=np.asarray(pd.read_pickle(ROOT/'results/round_v5/cache_base/matrices.pkl')['t'])
    y=y_all[t]; baseline=np.load(ROOT/'results/round_v6/tpe_feature_best_oof.npy')
    params=json.loads((ROOT/'results/round_v6/tpe_feature_best_params.json').read_text())
    params.update(n_estimators=1013,n_jobs=2,verbosity=-1)
    fold_rows=[]; candidate_oof=np.empty_like(baseline); blend_oof=np.empty_like(baseline)
    for k in range(3):
        fold=pd.read_pickle(ROOT/f'results/round_v6/fold_cache/seed42_fold{k}.pkl')
        path=OUT/f'warranty_fold{k}.npy'
        if path.exists(): pred=np.load(path)
        else:
            tr=fold['train'].reset_index(drop=True).copy(); va=fold['valid'].reset_index(drop=True).copy()
            tr=pd.concat([tr,warranty_features(raw,t[fold['train_rows']])],axis=1)
            va=pd.concat([va,warranty_features(raw,t[fold['valid_rows']])],axis=1)
            model=LGBMClassifier(**params); started=time.time()
            model.fit(tr,y[fold['train_rows']])
            pred=model.predict_proba(va)[:,1].astype(np.float32)
            np.save(path,pred)
            print('FOLD_SAVED',k,'seconds',round(time.time()-started,1),flush=True)
        iv=fold['valid_rows']; candidate_oof[iv]=pred; blend_oof[iv]=.5*(baseline[iv]+pred)
        fold_rows.append({'fold':k,'baseline':metrics(y[iv],baseline[iv]),'warranty_model':metrics(y[iv],pred),'blend_50':metrics(y[iv],blend_oof[iv])})
        print('FOLD_RESULT',json.dumps(fold_rows[-1]),flush=True)
    result={'features':['warranty_duration_days_log','warranty_duration_unit','warranty_semantic_type','warranty_explicit_type'],
            'folds':fold_rows,'baseline_oof':metrics(y,baseline),'warranty_model_oof':metrics(y,candidate_oof),'blend_50_oof':metrics(y,blend_oof)}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('SUMMARY',json.dumps(result),flush=True)

if __name__=='__main__':main()
