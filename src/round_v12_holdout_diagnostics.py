"""Label-free comparison of grouped holdout feature mix against the official unlabeled test."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/round_v12_holdout';OUT.mkdir(parents=True,exist_ok=True)

def dist(a,b):
 a=pd.Series(a).fillna('__MISSING__').astype(str).value_counts(normalize=True);b=pd.Series(b).fillna('__MISSING__').astype(str).value_counts(normalize=True);ix=a.index.union(b.index);return float(.5*(a.reindex(ix,fill_value=0)-b.reindex(ix,fill_value=0)).abs().sum())
def parent_overlap(a,b):
 a=pd.Series(a).dropna().astype(str);b=pd.Series(b).dropna().astype(str);aset=set(a);bset=set(b);return {'shared_group_ids':len(aset&bset),'fraction_eval_rows_with_train_parent':float(b.isin(aset).mean()),'fraction_unique_eval_parents_seen':float(len(aset&bset)/max(len(bset),1))}
def main():
 train=pd.read_json(ROOT/'data/ap_2026_q_2/raw/train_data.jsonlines',lines=True);test=pd.read_json(ROOT/'data/ap_2026_q_2/raw/test_data.jsonlines',lines=True)
 out={'rows':{'labeled_train':len(train),'unlabeled_test':len(test)},'parent_item_overlap_train_test':parent_overlap(train.parent_item_id,test.parent_item_id),'seller_overlap_train_test':parent_overlap(train.seller_id,test.seller_id),'category_tv_train_test':dist(train.category_id,test.category_id),'categorical_tv':{},'numeric_ks':{},'missing_rate_gap':{}}
 for col in ['category_id','listing_type_id','buying_mode','status','site_id']:
  if col in train and col in test:out['categorical_tv'][col]=dist(train[col],test[col])
 for col in ['price','base_price','initial_quantity','available_quantity','sold_quantity']:
  a=pd.to_numeric(train[col],errors='coerce').dropna();b=pd.to_numeric(test[col],errors='coerce').dropna()
  out['numeric_ks'][col]={'statistic':float(ks_2samp(a,b).statistic),'train_median':float(a.median()),'test_median':float(b.median())}
 for col in ['warranty','original_price','variations','seller_address','attributes','pictures']:
  out['missing_rate_gap'][col]={'train':float(train[col].isna().mean()),'test':float(test[col].isna().mean()),'gap':float(test[col].isna().mean()-train[col].isna().mean())}
 cache=pd.read_pickle(ROOT/'results/round_v5/cache_base/matrices.pkl');rows=cache['t'];folds=[]
 for k in range(3):
  f=pd.read_pickle(ROOT/f'results/round_v6/fold_cache/seed42_fold{k}.pkl');it,iv=f['train_rows'],f['valid_rows'];tr=train.iloc[rows[it]];va=train.iloc[rows[iv]]
  folds.append({'fold':k,'train_rows':len(tr),'valid_rows':len(va),'group_overlap':parent_overlap(tr.parent_item_id,va.parent_item_id)['shared_group_ids'],'category_tv_valid_vs_official_test':dist(va.category_id,test.category_id),'listing_type_tv_valid_vs_test':dist(va.listing_type_id,test.listing_type_id),'valid_parent_overlap_with_official_test':parent_overlap(tr.parent_item_id,test.parent_item_id)})
 out['seed42_fold_comparison']=folds
 (OUT/'diagnostics.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
