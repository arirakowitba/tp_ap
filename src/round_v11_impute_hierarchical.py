"""Grouped OOF screen: category-hierarchical seller encoding + explicit structured-missingness features."""
from pathlib import Path
import argparse, json, time
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/round_v11_impute_hierarchical'
OUT.mkdir(parents=True,exist_ok=True)

def rate(keys_seller, keys_category, y, groups, valid_seller, valid_category, alpha=10., cat_alpha=20., seed=2026):
    """OOF rate shrunk to a smoothed category prior; groups never cross folds."""
    s=pd.Series(keys_seller).fillna('__MISSING__').astype(str).reset_index(drop=True)
    c=pd.Series(keys_category).fillna('__MISSING__').astype(str).reset_index(drop=True)
    vs=pd.Series(valid_seller).fillna('__MISSING__').astype(str).reset_index(drop=True)
    vc=pd.Series(valid_category).fillna('__MISSING__').astype(str).reset_index(drop=True)
    y=np.asarray(y,dtype=np.int8); groups=np.asarray(groups).astype(str); oof=np.zeros(len(y),dtype=np.float32)
    folds=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=seed)
    def predict(it, iv_s, iv_c):
        prior=float(y[it].mean()); base=pd.DataFrame({'c':c.iloc[it].to_numpy(),'y':y[it]})
        cat=base.groupby('c').y.agg(['sum','count']); cat_rate=(cat['sum']+cat_alpha*prior)/(cat['count']+cat_alpha)
        joint=pd.DataFrame({'key':s.iloc[it].to_numpy()+'|'+c.iloc[it].to_numpy(),'y':y[it]}).groupby('key').y.agg(['sum','count'])
        cp=iv_c.map(cat_rate).fillna(prior).to_numpy(); key=iv_s.to_numpy()+'|'+iv_c.to_numpy()
        n=pd.Series(key).map(joint['count']).fillna(0).to_numpy(); sm=pd.Series(key).map(joint['sum']).fillna(0).to_numpy()
        return ((sm+alpha*cp)/(n+alpha)).astype(np.float32)
    for it,iv in folds.split(np.zeros(len(y)),y,groups): oof[iv]=predict(it,s.iloc[iv],c.iloc[iv])
    full= predict(np.arange(len(y)),vs,vc)
    return oof,full

def category_median_imputation(tr,va):
    tr=tr.copy();va=va.copy();cat_tr=tr['category_id'].astype('string').fillna('__MISSING__');cat_va=va['category_id'].astype('string').fillna('__MISSING__')
    num=tr.select_dtypes(include='number').columns
    cols=[c for c in num if tr[c].isna().any() and tr[c].notna().any()]
    for col in cols:
        median=float(tr[col].median());stats=pd.DataFrame({'category':cat_tr.to_numpy(),'value':tr[col].to_numpy()}).dropna().groupby('category').value.median()
        fill_tr=cat_tr.map(stats).fillna(median);fill_va=cat_va.map(stats).fillna(median)
        tr[col+'__cat_median']=tr[col].fillna(fill_tr).astype('float32');va[col+'__cat_median']=va[col].fillna(fill_va).astype('float32')
        tr[col+'__missing']=tr[col].isna().astype('int8');va[col+'__missing']=va[col].isna().astype('int8')
    return tr,va,cols

def metric(y,p):
    p=np.clip(np.asarray(p),1e-7,1-1e-7);return {'accuracy':float(accuracy_score(y,p>=.5)),'logloss':float(log_loss(y,p))}

def main(folds_to_run):
    raw=pd.read_json(ROOT/'data/ap_2026_q_2/raw/train_data.jsonlines',lines=True);y_all=(raw.condition=='used').to_numpy(dtype=np.int8)
    groups_all=raw.parent_item_id.astype('string').fillna('__ROW__'+raw.id.astype(str)).to_numpy()
    base=pd.read_pickle(ROOT/'results/round_v5/cache_base/matrices.pkl');rows=base['t'];y=y_all[rows];groups=groups_all[rows]
    baseline=np.load(ROOT/'results/round_v6/tpe_feature_best_oof.npy')
    cfg=json.loads((ROOT/'results/round_v6/tpe_feature_best_params.json').read_text(encoding='utf-8'))
    for k in folds_to_run:
        path=OUT/f'fold{k}.pkl';result_path=OUT/f'fold{k}.json'
        if result_path.exists():print('FOLD_CACHED',k,flush=True);continue
        fold=pd.read_pickle(ROOT/f'results/round_v6/fold_cache/seed42_fold{k}.pkl')
        it,iv=fold['train_rows'],fold['valid_rows'];gtr=groups[it];gva=groups[iv]
        overlap=set(gtr)&set(gva)
        if overlap:raise ValueError(f'family overlap in fold {k}: {len(overlap)}')
        tr,va,imputed=category_median_imputation(fold['train'],fold['valid'])
        # Use raw keys and labels for a family-safe, category-centered seller prior.
        rr_tr,rr_va=rows[it],rows[iv]
        oof,valid=rate(raw.seller_id.iloc[rr_tr],raw.category_id.iloc[rr_tr],y[it],gtr,
                       raw.seller_id.iloc[rr_va],raw.category_id.iloc[rr_va],seed=8100+k)
        tr['seller_category_oof_rate']=oof;va['seller_category_oof_rate']=valid
        yy=y[iv];base_p=baseline[iv]
        print('FIT_START',k,'train',len(it),'valid',len(iv),'imputed_cols',len(imputed),flush=True)
        model=LGBMClassifier(**cfg)
        model.fit(tr,y[it],eval_set=[(va,yy)],eval_metric='binary_logloss',callbacks=[early_stopping(120,verbose=False)])
        pred=model.predict_proba(va)[:,1].astype(np.float32)
        result={'fold':k,'train_rows':len(it),'valid_rows':len(iv),'group_overlap':0,'best_iteration':int(model.best_iteration_ or cfg['n_estimators']),
                'explicit_imputation_features':imputed,'baseline':metric(yy,base_p),'candidate':metric(yy,pred),
                'fold_blend_10pct':metric(yy,.9*base_p+.1*pred)}
        np.save(OUT/f'fold{k}_candidate.npy',pred);result_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('FOLD_RESULT',json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--folds',nargs='+',type=int,default=[0]);args=parser.parse_args();main(args.folds)

