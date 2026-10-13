from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,log_loss
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/round_v12_holdout'
raw=pd.read_json(ROOT/'data/ap_2026_q_2/raw/train_data.jsonlines',lines=True);test=pd.read_json(ROOT/'data/ap_2026_q_2/raw/test_data.jsonlines',lines=True);cache=pd.read_pickle(ROOT/'results/round_v5/cache_base/matrices.pkl');rows=cache['t'];y=raw.condition.eq('used').to_numpy(dtype=np.int8)[rows];p=np.load(ROOT/'results/round_v6/tpe_feature_best_oof.npy')
top=set(test.category_id.fillna('__MISSING__').astype(str).value_counts().head(150).index)
def bucket(x):
 s=pd.Series(x).fillna('__MISSING__').astype(str);return s.where(s.isin(top),'__OTHER__')
vc=bucket(raw.category_id.iloc[rows]);tc=bucket(test.category_id);pt=tc.value_counts(normalize=True);pv=vc.value_counts(normalize=True);covered=float(pt.reindex(pv.index,fill_value=0).sum());w=vc.map((pt/pv).to_dict()).fillna(0).to_numpy();correct=(p>=.5)==y
res={'top_k':150,'test_mass_top_k':float(pt.reindex(list(top),fill_value=0).sum()),'test_mass_covered':covered,'raw_oof_accuracy':float(accuracy_score(y,p>=.5)),'test_category_reweighted_oof_accuracy':float(np.average(correct,weights=w)),'raw_oof_logloss':float(log_loss(y,np.clip(p,1e-7,1-1e-7))),'test_category_reweighted_oof_logloss':float(log_loss(y,np.clip(p,1e-7,1-1e-7),sample_weight=w)),'effective_sample_size':float(w.sum()**2/(w*w).sum())}
old=json.loads((OUT/'diagnostics.json').read_text());old['oof_reweighted_to_test_category_mix']=res;(OUT/'diagnostics.json').write_text(json.dumps(old,indent=2),encoding='utf-8');print(json.dumps(res,indent=2))
