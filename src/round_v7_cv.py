from pathlib import Path
import json,sys,numpy as np,pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score,log_loss
ROOT=Path(r'C:\Users\arira\OneDrive\Documents\GitHub\tp_ap');OUT=Path(__file__).resolve().parent/'experiment_outputs_round_v7';OUT.mkdir(parents=True,exist_ok=True)
folds=[pd.read_pickle(ROOT/f'results/round_v6/fold_cache/seed42_fold{k}.pkl') for k in range(3)]
imp=pd.read_csv(ROOT/'results/round_v6/tpe_feature_importance.csv',index_col=0)['mean_gain'];weak=imp.sort_values().index[:int(.20*len(imp))].tolist()
yall=[]
with (ROOT/'data/ap_2026_q_2/raw/train_data.jsonlines').open(encoding='utf-8') as f:
 for line in f:yall.append(int(json.loads(line)['condition']=='used'))
sel=pd.read_pickle(ROOT/'results/round_v5/cache_base/matrices.pkl');t=np.asarray(sel['t']);y=np.asarray(yall,dtype=np.int8)[t]
base=np.load(ROOT/'results/round_v6/tpe_feature_best_oof.npy');pred=np.zeros(len(y),dtype=np.float32);rows=[];iters=[1013,1025,968]
for k,f in enumerate(folds):
 cols=[c for c in f['train'].columns if c not in weak];tr=f['train'][cols];va=f['valid'][cols];itr=f['train_rows'];iva=f['valid_rows']
 p=json.loads((ROOT/'results/round_v6/tpe_feature_best_params.json').read_text());p.update(n_estimators=iters[k],n_jobs=2,verbosity=-1)
 model=LGBMClassifier(**p);model.fit(tr,y[itr]);pred[iva]=model.predict_proba(va)[:,1]
 row={'fold':k,'baseline_accuracy':float(accuracy_score(y[iva],base[iva]>=.5)),'accuracy':float(accuracy_score(y[iva],pred[iva]>=.5)),'baseline_logloss':float(log_loss(y[iva],base[iva])),'logloss':float(log_loss(y[iva],np.clip(pred[iva],1e-7,1-1e-7))),'features':len(cols)};rows.append(row)
 np.save(OUT/'weak_features_cv_oof.partial.npy',pred);pd.DataFrame(rows).to_csv(OUT/'weak_features_cv_folds.partial.csv',index=False);print('FOLD',json.dumps(row),flush=True)
result={'candidate':'drop_bottom20_gain','features':int(len(cols)),'baseline_accuracy':float(accuracy_score(y,base>=.5)),'accuracy':float(accuracy_score(y,pred>=.5)),'baseline_logloss':float(log_loss(y,base)),'logloss':float(log_loss(y,np.clip(pred,1e-7,1-1e-7))),'folds':rows}
(OUT/'weak_features_cv.json').write_text(json.dumps(result,indent=2));np.save(OUT/'weak_features_cv_oof.npy',pred);print('CV',json.dumps(result),flush=True)
