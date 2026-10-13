"""Test-only category importance weighting on one leakage-safe outer grouped fold."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from lightgbm import LGBMClassifier,early_stopping
from sklearn.metrics import accuracy_score,log_loss
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/round_v13_category_weights';OUT.mkdir(parents=True,exist_ok=True)
TOP_K=150;CLIP=(.25,4.)
def bucket(x,top):
 s=pd.Series(x).fillna('__MISSING__').astype(str);return s.where(s.isin(top),'__OTHER__')
def weights(source,target):
 """Importance ratio P(target category)/P(source category), clipped and mean-normalized."""
 a=pd.Series(source);b=pd.Series(target);cats=a.value_counts().index.union(b.value_counts().index);ca=a.value_counts().reindex(cats,fill_value=0).astype(float);cb=b.value_counts().reindex(cats,fill_value=0).astype(float)
 pa=(ca+1)/(ca.sum()+len(cats));pb=(cb+1)/(cb.sum()+len(cats));mapping=(pb/pa).to_dict();w=a.map(mapping).to_numpy(dtype=float);w=np.clip(w,*CLIP);return w/w.mean()
def metric(y,p,w=None):
 p=np.clip(np.asarray(p),1e-7,1-1e-7);correct=(p>=.5)==y
 return {'accuracy':float(accuracy_score(y,p>=.5)),'test_category_weighted_accuracy':float(np.average(correct,weights=w)) if w is not None else None,'test_category_weighted_logloss':float(log_loss(y,p,sample_weight=w)) if w is not None else None}
def main(fold_no):
 raw=pd.read_json(ROOT/'data/ap_2026_q_2/raw/train_data.jsonlines',lines=True);test=pd.read_json(ROOT/'data/ap_2026_q_2/raw/test_data.jsonlines',lines=True);yall=raw.condition.eq('used').to_numpy(dtype=np.int8)
 cache=pd.read_pickle(ROOT/'results/round_v5/cache_base/matrices.pkl');rows=cache['t'];y=yall[rows];baseline=np.load(ROOT/'results/round_v6/tpe_feature_best_oof.npy')
 f=pd.read_pickle(ROOT/f'results/round_v6/fold_cache/seed42_fold{fold_no}.pkl');it,iv=f['train_rows'],f['valid_rows'];tr,va=f['train'].copy(),f['valid'].copy();yy=y[iv];basep=baseline[iv]
 top=set(test.category_id.fillna('__MISSING__').astype(str).value_counts().head(TOP_K).index)
 cat_train=bucket(raw.category_id.iloc[rows[it]],top);cat_valid=bucket(raw.category_id.iloc[rows[iv]],top);cat_test=bucket(test.category_id,top)
 wtrain=weights(cat_train,cat_test);wvalid=weights(cat_valid,cat_test)
 # Coverage: fraction of test-category mass represented by this valid fold.
 ptest=cat_test.value_counts(normalize=True);pvalid=cat_valid.value_counts();represented=float(ptest.reindex(pvalid.index,fill_value=0).sum())
 cfg=json.loads((ROOT/'results/round_v6/tpe_feature_best_params.json').read_text(encoding='utf-8'))
 print('FIT_START',fold_no,'train',len(it),'valid',len(iv),'topk_test_mass',float(ptest.reindex(list(top),fill_value=0).sum()),'weight_range',float(wtrain.min()),float(wtrain.max()),flush=True)
 model=LGBMClassifier(**cfg);model.fit(tr,y[it],sample_weight=wtrain,eval_set=[(va,yy)],eval_sample_weight=[wvalid],eval_metric='binary_logloss',callbacks=[early_stopping(120,verbose=False)])
 pred=model.predict_proba(va)[:,1].astype(np.float32);res={'fold':fold_no,'train_rows':len(it),'valid_rows':len(iv),'group_overlap':0,'top_k_categories':TOP_K,'test_mass_in_top_categories':float(ptest.reindex(list(top),fill_value=0).sum()),'valid_test_category_mass_covered':represented,'weight_clip':CLIP,'weight_min':float(wtrain.min()),'weight_max':float(wtrain.max()),'weight_ess':float(wtrain.sum()**2/np.square(wtrain).sum()),'best_iteration':int(model.best_iteration_ or cfg['n_estimators']),'baseline_oof':metric(yy,basep,wvalid),'category_weighted_candidate':metric(yy,pred,wvalid)}
 np.save(OUT/f'fold{fold_no}.npy',pred);(OUT/f'fold{fold_no}.json').write_text(json.dumps(res,indent=2),encoding='utf-8');print('RESULT',json.dumps(res),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--fold',type=int,default=0);main(a.parse_args().fold)

