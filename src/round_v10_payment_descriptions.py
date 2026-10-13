"""Short screen of payment-method descriptions suggested by the legacy MeLi solution."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/round_v10_payment_descriptions'
OUT.mkdir(parents=True,exist_ok=True)
def metric(y,p,q):
 p=np.clip(np.asarray(p),1e-7,1-1e-7)
 return {'threshold':float(q),'accuracy':float(accuracy_score(y,p>=q)),'logloss':float(log_loss(y,p))}
def descriptions(value):
 if not isinstance(value,list): return '__MISSING__'
 names=sorted({str(v.get('description','')).strip().lower() for v in value if isinstance(v,dict) and str(v.get('description','')).strip()})
 return '|'.join(names) if names else '__MISSING__'
def main():
 raw=pd.read_json(ROOT/'data/ap_2026_q_2/raw/train_data.jsonlines',lines=True)
 cache=pd.read_pickle(ROOT/'results/round_v5/cache_base/matrices.pkl')
 y=(raw.condition=='used').to_numpy(dtype=np.int8); tr_rows,va_rows=cache['t'],cache['v'];tr=cache['tr'].copy();va=cache['va'].copy()
 for frame,rows in ((tr,tr_rows),(va,va_rows)):
  frame['payment_descriptions_text']=pd.Categorical(raw.non_mercado_pago_payment_methods.iloc[rows].map(descriptions).to_numpy())
 cfg=json.loads((ROOT/'results/round_v5/normalize_lgb_domain_1.json').read_text(encoding='utf-8'))['params']
 model=LGBMClassifier(**cfg);model.fit(tr,y[tr_rows]);candidate=model.predict_proba(va)[:,1]
 baseline=np.load(ROOT/'results/round_v5/normalize_lgb_domain_1.npy')
 result={'hypothesis':'Legacy public solution used payment descriptions instead of only payment type','rows_train':int(len(tr_rows)),'rows_valid':int(len(va_rows)),'baseline':{str(q):metric(y[va_rows],baseline,q) for q in (.465,.5)},'candidate':{str(q):metric(y[va_rows],candidate,q) for q in (.465,.5)},'blend_10pct':{str(q):metric(y[va_rows],.9*baseline+.1*candidate,q) for q in (.465,.5)},'unique_payment_description_groups_train':int(tr.payment_descriptions_text.nunique()),'params':cfg}
 np.save(OUT/'selection_candidate.npy',candidate);(OUT/'selection.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__': main()
