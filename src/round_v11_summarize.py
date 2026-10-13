from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,log_loss
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/round_v11_impute_hierarchical'
cache=pd.read_pickle(ROOT/'results/round_v5/cache_base/matrices.pkl');baseline=np.load(ROOT/'results/round_v6/tpe_feature_best_oof.npy');pred=np.empty(len(baseline),dtype=np.float32)
for k in range(3):
 f=pd.read_pickle(ROOT/f'results/round_v6/fold_cache/seed42_fold{k}.pkl');pred[f['valid_rows']]=np.load(OUT/f'fold{k}_candidate.npy')
yall=pd.read_json(ROOT/'data/ap_2026_q_2/raw/train_data.jsonlines',lines=True).condition.eq('used').to_numpy(dtype=np.int8);y=yall[cache['t']]
def m(p):
 p=np.clip(p,1e-7,1-1e-7);return {'accuracy':float(accuracy_score(y,p>=.5)),'logloss':float(log_loss(y,p)),'correct':int(accuracy_score(y,p>=.5)*len(y))}
res={'rows':len(y),'protocol':'seed42 StratifiedGroupKFold 3 outer folds; predictions cover each of 56,000 rows once','baseline':m(baseline),'candidate_imputation_plus_hierarchical_rate':m(pred),'blend_10pct':m(.9*baseline+.1*pred),'folds':[json.loads((OUT/f'fold{k}.json').read_text()) for k in range(3)]}
(OUT/'summary.json').write_text(json.dumps(res,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in res.items() if k!='folds'},indent=2))


