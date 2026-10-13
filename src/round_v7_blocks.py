"""One-fold screen for grouped feature ablations; never changes the production model."""
from pathlib import Path
import json,os,numpy as np,pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score,log_loss
ROOT=Path(r'C:\Users\arira\OneDrive\Documents\GitHub\tp_ap');OUT=Path(os.environ.get('AP_ROUND_V7_OUT',str(ROOT/'results/round_v7_runs')));OUT.mkdir(parents=True,exist_ok=True)
f=pd.read_pickle(ROOT/'results/round_v6/fold_cache/seed42_fold0.pkl');tr,va=f['train'],f['valid'];it,iv=f['train_rows'],f['valid_rows']
importance=pd.read_csv(ROOT/'results/round_v6/tpe_feature_importance.csv',index_col=0)
yall=[]
with (ROOT/'data/ap_2026_q_2/raw/train_data.jsonlines').open(encoding='utf-8') as h:
 for line in h:yall.append(int(json.loads(line)['condition']=='used'))
sel=pd.read_pickle(ROOT/'results/round_v5/cache_base/matrices.pkl');t=np.asarray(sel['t']);y=np.asarray(yall,dtype=np.int8)[t];base=np.load(ROOT/'results/round_v6/tpe_feature_best_oof.npy')
def block(name,c):
 l=c.lower()
 if name=='image':return any(k in l for k in ('picture','img_','image'))
 if name=='attributes':return c.startswith('attr_')
 if name=='text':return c.startswith(('title_','lex_','svm_')) or 'warranty_text' in l
 if name=='seller':return c=='seller_id' or c.startswith(('seller_','seller_category','seller_listing'))
 return False
rows=[]
for name in ['image','attributes','text','seller']:
 removed=[c for c in tr.columns if block(name,c)];cols=[c for c in tr.columns if c not in removed]
 p=json.loads((ROOT/'results/round_v6/tpe_feature_best_params.json').read_text());p.update(n_estimators=1013,n_jobs=2,verbosity=-1)
 m=LGBMClassifier(**p);m.fit(tr[cols],y[it]);pred=m.predict_proba(va[cols])[:,1]
 row={'block_removed':name,'removed_count':len(removed),'remaining_features':len(cols),'baseline_accuracy':float(accuracy_score(y[iv],base[iv]>=.5)),'accuracy':float(accuracy_score(y[iv],pred>=.5)),'baseline_logloss':float(log_loss(y[iv],base[iv])),'logloss':float(log_loss(y[iv],np.clip(pred,1e-7,1-1e-7)))}
 rows.append(row);pd.DataFrame(rows).to_csv(OUT/'block_ablation_fold0.csv',index=False);np.save(OUT/f'block_{name}_fold0.npy',pred);print('BLOCK',json.dumps(row),flush=True)
(OUT/'block_ablation_fold0.json').write_text(json.dumps(rows,indent=2))
