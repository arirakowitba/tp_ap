"""Reproduce the warranty feature check on the frozen seed-731 confirmation."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/round_v9_warranty_semantics'

def metrics(y,p):
    return {'accuracy':float(accuracy_score(y,p>=.5)),
            'logloss':float(log_loss(y,np.clip(p,1e-7,1-1e-7)))}

def main(retrain=False):
    import sys
    sys.path.insert(0,str(ROOT/'src'))
    from round_v9_warranty_semantics import warranty_features
    raw=pd.read_json(ROOT/'data/ap_2026_q_2/raw/train_data.jsonlines',lines=True)
    y=(raw.condition=='used').to_numpy(dtype=np.int8)
    groups=raw.parent_item_id.astype('string').fillna('__ROW__'+raw.id.astype(str)).to_numpy()
    frozen=pd.read_csv(ROOT/'results/round_v3/prior_confirmation_predictions.csv')
    valid_rows=frozen.row_index.to_numpy(dtype=np.int64)
    if not np.array_equal(y[valid_rows],frozen.y_used.to_numpy(dtype=np.int8)):
        raise ValueError('Frozen confirmation manifest does not match labels')
    train_rows=np.setdiff1d(np.arange(len(y)),valid_rows)
    if set(groups[train_rows]) & set(groups[valid_rows]):
        raise ValueError('Family leakage in confirmation split')
    cache=pd.read_pickle(ROOT/'results/round_v5/confirmation/normalized_matrices.pkl')
    pred_path=OUT/'confirmation_warranty_model.npy'
    if retrain or not pred_path.exists():
        tr=cache['tr'].reset_index(drop=True).copy(); va=cache['va'].reset_index(drop=True).copy()
        tr=pd.concat([tr,warranty_features(raw,train_rows)],axis=1)
        va=pd.concat([va,warranty_features(raw,valid_rows)],axis=1)
        params=json.loads((ROOT/'results/round_v5/normalize_lgb_domain_1.json').read_text())['params']
        params.update(n_jobs=2,verbosity=-1)
        model=LGBMClassifier(**params);model.fit(tr,y[train_rows])
        np.save(pred_path,model.predict_proba(va)[:,1])
    candidate=np.load(pred_path)
    baseline_prior=np.load(ROOT/'results/round_v5/confirmation/prior_ensemble.npy')
    baseline_norm=np.load(ROOT/'results/round_v5/confirmation/normalize_lgb_domain_1.npy')
    result={'seed':731,'train_rows':len(train_rows),'valid_rows':len(valid_rows),
      'group_overlap':len(set(groups[train_rows])&set(groups[valid_rows])),
      'baseline_prior':metrics(y[valid_rows],baseline_prior),
      'baseline_normalized_lgb':metrics(y[valid_rows],baseline_norm),
      'warranty_model':metrics(y[valid_rows],candidate),
      'prior_blends':{str(w):metrics(y[valid_rows],(1-w)*baseline_prior+w*candidate) for w in [.1,.2,.3,.4,.5]},
      'normalized_blends':{str(w):metrics(y[valid_rows],(1-w)*baseline_norm+w*candidate) for w in [.1,.2,.3,.4,.5]}}
    (OUT/'confirmation.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--retrain',action='store_true');main(parser.parse_args().retrain)
