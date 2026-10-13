"""Short, leakage-aware tests for temporal seller priors and error correction."""
from pathlib import Path
import argparse, json, sys
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(r'C:\Users\arira\OneDrive\Documents\GitHub\tp_ap')
OUT = Path(__file__).resolve().parent / 'experiment_outputs_round_v7'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'src'))
from competition_pipeline_v2 import groups_for


def load_context():
    rows=[]; labels=[]
    with (ROOT/'data/ap_2026_q_2/raw/train_data.jsonlines').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line); rows.append(r); labels.append(int(r['condition']=='used'))
    raw=pd.DataFrame(rows); y=np.asarray(labels,dtype=np.int8)
    sel=pd.read_pickle(ROOT/'results/round_v5/cache_base/matrices.pkl')
    return raw,y,groups_for(raw),np.asarray(sel['t'])


def past_seller_features(raw, source_rows, query_rows, y, groups):
    """Use only strictly earlier labeled rows from training; remove same-family history."""
    date=pd.to_datetime(raw.date_created,utc=True,errors='coerce').astype('int64').to_numpy()
    seller=raw.seller_id.fillna('__MISSING__').astype(str).to_numpy()
    fam=np.asarray(groups).astype(str)
    src=pd.DataFrame({'seller':seller[source_rows], 'family':fam[source_rows], 'date':date[source_rows], 'y':y[source_rows]})
    src=src[src.date > -9_000_000_000_000_000_000].sort_values('date')
    by_seller={}; by_family={}
    for s,g in src.groupby('seller',sort=False):
        ts=g.date.to_numpy(); ys=g.y.to_numpy(dtype=np.int32)
        by_seller[s]=(ts,np.r_[0,np.cumsum(ys)],len(ys))
        for f,h in g.groupby('family',sort=False):
            hts=h.date.to_numpy(); hys=h.y.to_numpy(dtype=np.int32)
            by_family[(s,f)]=(hts,np.r_[0,np.cumsum(hys)])
    prior=float(y[source_rows].mean()); out=np.zeros((len(query_rows),3),dtype=np.float32)
    for i,r in enumerate(query_rows):
        s=seller[r]; f=fam[r]; qdate=date[r]
        if qdate <= -9_000_000_000_000_000_000 or s not in by_seller: continue
        ts,csum,total=by_seller[s]; j=int(np.searchsorted(ts,qdate,side='left'))
        n=j; used=int(csum[j])
        hist=by_family.get((s,f))
        if hist is not None:
            fts,fcsum=hist; k=int(np.searchsorted(fts,qdate,side='left'))
            n-=k; used-=int(fcsum[k])
        out[i,0]=np.log1p(max(n,0)); out[i,1]=(used+20*prior)/(max(n,0)+20); out[i,2]=np.log1p(max(used,0))
    return pd.DataFrame(out,index=None,columns=['seller_past_log_count','seller_past_used_rate','seller_past_log_used_count'])


def metrics(y,p):
    return {'accuracy':float(accuracy_score(y,p>=.5)),'logloss':float(log_loss(y,np.clip(p,1e-7,1-1e-7)))}


def run_correction():
    raw,y,groups,t=load_context(); folds=[pd.read_pickle(ROOT/f'results/round_v6/fold_cache/seed42_fold{k}.pkl') for k in range(3)]
    base=np.load(ROOT/'results/round_v6/tpe_feature_best_oof.npy'); hist=np.zeros(len(t),dtype=np.float32);
    for k,f in enumerate(folds):
        src=t[f['train_rows']]; qry=t[f['valid_rows']]
        x=past_seller_features(raw,src,qry,y,groups)
        hist[f['valid_rows']]=x.seller_past_used_rate.to_numpy()
    yy=y[t]; rows=[]
    for w in [0.0,.05,.10,.15,.20]:
        p=(1-w)*base+w*hist; rows.append({'prior_weight':w,**metrics(yy,p)})
    pd.DataFrame(rows).to_csv(OUT/'seller_history_correction_cv.csv',index=False)
    (OUT/'seller_history_correction.json').write_text(json.dumps({'baseline':metrics(yy,base),'weights':rows},indent=2))
    print('CORRECTION',json.dumps({'baseline':metrics(yy,base),'weights':rows}),flush=True)


def run_model(mode):
    raw,y,groups,t=load_context();f=pd.read_pickle(ROOT/'results/round_v6/fold_cache/seed42_fold0.pkl')
    tr,va=f['train'].copy(),f['valid'].copy(); it,iv=f['train_rows'],f['valid_rows']; ydev=y[t]
    importance=pd.read_csv(ROOT/'results/round_v6/tpe_feature_importance.csv',index_col=0)['mean_gain']
    base_oof=np.load(ROOT/'results/round_v6/tpe_feature_best_oof.npy')
    baseline=metrics(ydev[iv],base_oof[iv])
    if mode=='history':
        trh=past_seller_features(raw,t[it],t[it],y,groups); vah=past_seller_features(raw,t[it],t[iv],y,groups)
        tr=pd.concat([tr.reset_index(drop=True),trh.reset_index(drop=True)],axis=1);va=pd.concat([va.reset_index(drop=True),vah.reset_index(drop=True)],axis=1); cols=list(tr.columns)
    else:
        cols=list(tr.columns); weak=importance.sort_values().index[:max(1,int(.20*len(importance)))].tolist(); cols=[c for c in cols if c not in weak]
    p=json.loads((ROOT/'results/round_v6/tpe_feature_best_params.json').read_text()); p.update(n_estimators=1013,n_jobs=2,verbosity=-1)
    model=LGBMClassifier(**p);model.fit(tr[cols],ydev[it]);pred=model.predict_proba(va[cols])[:,1]
    row={'mode':mode,'baseline_fold0':baseline,'candidate_fold0':metrics(ydev[iv],pred),'features':len(cols),'iteration':1013}
    if mode=='history': row['new_features']=['seller_past_log_count','seller_past_used_rate','seller_past_log_used_count']
    else: row['removed_features']=len(importance)-len(cols)
    (OUT/f'{mode}_screen_fold0.json').write_text(json.dumps(row,indent=2));np.save(OUT/f'{mode}_screen_fold0.npy',pred)
    print(mode.upper(),json.dumps(row),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['correction','history','weak_features'],required=True);a=ap.parse_args()
    if a.mode=='correction':run_correction()
    else:run_model(a.mode)
