"""Ablate missing-value handling and seller/category smoothing separately on grouped fold 0."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from lightgbm import LGBMClassifier,early_stopping
from sklearn.metrics import accuracy_score,log_loss
from round_v11_impute_hierarchical import ROOT,OUT,category_median_imputation,rate

def metrics(y,p):
 p=np.clip(np.asarray(p),1e-7,1-1e-7);return {'accuracy':float(accuracy_score(y,p>=.5)),'logloss':float(log_loss(y,p))}
def main(variants,fold_no):
 raw=pd.read_json(ROOT/'data/ap_2026_q_2/raw/train_data.jsonlines',lines=True);yall=raw.condition.eq('used').to_numpy(dtype=np.int8);groups_all=raw.parent_item_id.astype('string').fillna('__ROW__'+raw.id.astype(str)).to_numpy()
 base=pd.read_pickle(ROOT/'results/round_v5/cache_base/matrices.pkl');rows=base['t'];y=yall[rows];groups=groups_all[rows];baseline=np.load(ROOT/'results/round_v6/tpe_feature_best_oof.npy');cfg=json.loads((ROOT/'results/round_v6/tpe_feature_best_params.json').read_text(encoding='utf-8'))
 fold=pd.read_pickle(ROOT/f'results/round_v6/fold_cache/seed42_fold{fold_no}.pkl');it,iv=fold['train_rows'],fold['valid_rows'];yy=y[iv];bp=baseline[iv]
 for variant in variants:
  out=OUT/f'fold{fold_no}_{variant}.json';pp=OUT/f'fold{fold_no}_{variant}.npy'
  if out.exists():print('CACHED',variant);continue
  tr,va=fold['train'].copy(),fold['valid'].copy()
  if variant=='impute':tr,va,_=category_median_imputation(tr,va)
  elif variant=='hierarchical':
   oof,valid=rate(raw.seller_id.iloc[rows[it]],raw.category_id.iloc[rows[it]],y[it],groups[it],raw.seller_id.iloc[rows[iv]],raw.category_id.iloc[rows[iv]],seed=8100+fold_no)
   tr['seller_category_oof_rate']=oof;va['seller_category_oof_rate']=valid
  else:raise ValueError(variant)
  print('FIT_START',variant,flush=True);m=LGBMClassifier(**cfg);m.fit(tr,y[it],eval_set=[(va,yy)],eval_metric='binary_logloss',callbacks=[early_stopping(120,verbose=False)])
  p=m.predict_proba(va)[:,1].astype(np.float32);res={'variant':variant,'fold':fold_no,'train_rows':len(it),'valid_rows':len(iv),'group_overlap':0,'best_iteration':int(m.best_iteration_ or cfg['n_estimators']),'baseline':metrics(yy,bp),'candidate':metrics(yy,p)}
  np.save(pp,p);out.write_text(json.dumps(res,indent=2),encoding='utf-8');print('RESULT',json.dumps(res),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--variants',nargs='+',default=['impute','hierarchical']);a.add_argument('--fold',type=int,default=0);x=a.parse_args();main(x.variants,x.fold)

