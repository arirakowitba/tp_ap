from pathlib import Path
import json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/round_v10_payment_descriptions'
def metrics(y,p,q):
 p=np.clip(np.asarray(p),1e-7,1-1e-7);return {'threshold':float(q),'accuracy':float(accuracy_score(y,p>=q)),'logloss':float(log_loss(y,p))}
def description(value):
 if not isinstance(value,list):return '__MISSING__'
 z=sorted({str(v.get('description','')).strip().lower() for v in value if isinstance(v,dict) and str(v.get('description','')).strip()})
 return '|'.join(z) if z else '__MISSING__'
def main():
 raw=pd.read_json(ROOT/'data/ap_2026_q_2/raw/train_data.jsonlines',lines=True);y=(raw.condition=='used').to_numpy(dtype=np.int8)
 frozen=pd.read_csv(ROOT/'results/round_v3/prior_confirmation_predictions.csv');va_rows=frozen.row_index.to_numpy(dtype=np.int64);tr_rows=np.setdiff1d(np.arange(len(y)),va_rows)
 if not np.array_equal(y[va_rows],frozen.y_used.to_numpy(dtype=np.int8)):raise ValueError('frozen labels mismatch')
 groups=raw.parent_item_id.astype('string').fillna('__ROW__'+raw.id.astype(str)).to_numpy()
 if set(groups[tr_rows])&set(groups[va_rows]):raise ValueError('family leakage')
 cache=pd.read_pickle(ROOT/'results/round_v5/confirmation/normalized_matrices.pkl');tr=cache['tr'].copy();va=cache['va'].copy()
 for frame,rows in ((tr,tr_rows),(va,va_rows)):frame['payment_descriptions_text']=pd.Categorical(raw.non_mercado_pago_payment_methods.iloc[rows].map(description).to_numpy())
 cfg=json.loads((ROOT/'results/round_v5/normalize_lgb_domain_1.json').read_text(encoding='utf-8'))['params'];model=LGBMClassifier(**cfg);model.fit(tr,y[tr_rows]);candidate=model.predict_proba(va)[:,1]
 baseline=np.load(ROOT/'results/round_v5/confirmation/normalize_lgb_domain_1.npy')
 result={'split':'frozen seed731','rows_train':len(tr_rows),'rows_valid':len(va_rows),'group_overlap':0,'baseline':{str(q):metrics(y[va_rows],baseline,q) for q in (.465,.5)},'candidate':{str(q):metrics(y[va_rows],candidate,q) for q in (.465,.5)},'blend_10pct':{str(q):metrics(y[va_rows],.9*baseline+.1*candidate,q) for q in (.465,.5)}}
 np.save(OUT/'confirmation_candidate.npy',candidate);(OUT/'confirmation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
