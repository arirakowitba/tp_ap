"""Freeze a small complementary blend, check seed731, then optionally refit.

Selection and confirmation are different grouped splits, not independent datasets.
The prior seed42 baseline was trained on49k rows; new candidates use56k. Its
predictions are aligned to the same14k rows, but the confirmation comparison is
the fair same-training-size comparison. No public Kaggle score is inferred.
"""
from pathlib import Path
import argparse,json,time,signal,gc,itertools,hashlib
import numpy as np,pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold
from competition_pipeline_v2 import load_data,groups_for,matrices,metrics,svm_probability
from optimize_leaderboard import split,enrich
ROOT=Path(__file__).resolve().parents[1]

def prior_probability(protocol,rows=None):
    out=ROOT/'optimization';cfg=json.loads((out/'selected_ensemble.json').read_text())
    if protocol=='selection_seed42':
        original=pd.read_csv(out/'holdout_predictions.csv');p=original.set_index('row_index').loc[rows,'probability_used'].to_numpy()
    elif protocol=='alternate_seed731':p=pd.read_csv(ROOT/'results/round_v3/prior_confirmation_predictions.csv').probability_used.to_numpy()
    elif protocol=='full_train':p=pd.read_csv(ROOT/'results/round_v3/prior_full_predictions.csv').probability_used.to_numpy()
    else:raise ValueError('Unknown baseline protocol')
    if protocol=='full_train':
        sub=pd.read_csv(ROOT/'submissions/submission_optimized_ensemble.csv')
        assert np.all(np.where(p>=cfg['threshold'],'used','new')==sub.condition)
    return p

def collect(out,rows,y):
    result=[dict(name='prior_ensemble',family='prior',threshold=.5,params={},probability=prior_probability('selection_seed42',rows))]
    for prefix in ['normalize','embedding','category','combined']:
        for p in sorted(out.glob(prefix+'*.json')):
            cfg=json.loads(p.read_text())
            if not ('name' in cfg and 'accuracy' in cfg and (out/(cfg['name']+'.npy')).exists()):continue
            prob=np.load(out/(cfg['name']+'.npy'))
            if len(prob)!=len(rows) or not np.isfinite(prob).all():raise ValueError('Prediction alignment/finite values')
            result.append(dict(cfg,family=prefix,probability=prob))
    for p in out.glob('stacking*_C*_predictions.csv'):
        tab=pd.read_csv(p)
        if not np.array_equal(tab.row_index,rows) or not np.array_equal(tab.y_used,y):raise ValueError('Stacking alignment mismatch')
        name=p.name.removesuffix('_predictions.csv');design=name.split('_')[1];C=float(name.split('_C')[-1])
        result.append(dict(name=name,family='stacking',design=design,C=C,base_count=5 if name.startswith('stacking5_') else 4 if name.startswith('stacking4_') else 3,probability=tab.probability_used.to_numpy()))
    for cfg in result:cfg['accuracy']=metrics(y,cfg['probability'],cfg.get('threshold',.5))['accuracy']
    return result

def select(out,rows,y):
    candidates=collect(out,rows,y)
    # Keep one representative per method, avoiding a massive combinatorial sweep.
    top=[max([r for r in candidates if r['family']==family],key=lambda r:(r['accuracy'],r.get('clusters') is None)) for family in sorted(set(r['family'] for r in candidates))]
    best=None;trial_rows=[]
    combinations=[([r],[1.]) for r in top]
    combinations += [([a,b],[w,1-w]) for a,b in itertools.combinations(top,2) for w in [.25,.5,.75]]
    for components,weights in combinations:
        prob=sum(w*c['probability'] for w,c in zip(weights,components))
        q=max([.48,.49,.5,.51,.52],key=lambda q:metrics(y,prob,q)['accuracy'])
        row=dict(names=[c['name'] for c in components],weights=weights,**metrics(y,prob,q))
        trial_rows.append(row)
        if best is None or row['accuracy']>best['accuracy']:
            best=dict(row,components=[{k:v for k,v in c.items() if k!='probability'} for c in components],status='frozen_pending_confirmation',confirmation=None)
            pd.DataFrame({'row_index':rows,'y_used':y,'probability_used':prob}).to_csv(out/'round3_selection_predictions.csv',index=False)
    best['prior_selection_accuracy']=candidates[0]['accuracy'];best['selection_comparison_note']='Same14k evaluation rows; prior trained49k, new candidates56k.'
    (out/'round3_selected.json').write_text(json.dumps(best,indent=2))
    pd.DataFrame(trial_rows).to_csv(out/'round3_blend_trials.csv',index=False)
    return best

def cache_identity(raw,y,g,other):
    h=hashlib.sha256()
    for values in [raw.id.astype(str),pd.Series(y),pd.Series(g).astype(str),other.id.astype(str)]:
        h.update(pd.util.hash_pandas_object(values,index=False).to_numpy().tobytes())
    h.update(json.dumps(json.loads((ROOT/'results/round_v3/prior_prediction_metadata.json').read_text()),sort_keys=True).encode())
    return h.hexdigest()

def core(raw,y,g,other,out):
    cache=out/'matrices.pkl'
    fingerprint=cache_identity(raw,y,g,other)
    if cache.exists():
        d=pd.read_pickle(cache)
        if d.get('fingerprint')!=fingerprint or len(d['tr'])!=len(raw) or len(d['va'])!=len(other):raise ValueError('Core cache incompatible dataset/split/shape')
        return d['tr'],d['va']
    tr,va,_=matrices(raw,y,g,other,out)
    pd.to_pickle(dict(tr=tr,va=va,fingerprint=fingerprint),cache)
    return tr,va

def normalized_features(tr,va,raw,y,g,other,out,flags_only=False):
    cache=out/('flags_only_matrices.pkl' if flags_only else 'normalized_matrices.pkl')
    fingerprint=cache_identity(raw,y,g,other)
    if cache.exists():
        d=pd.read_pickle(cache)
        if d.get('fingerprint')!=fingerprint:raise ValueError('Normalized cache incompatible dataset/split')
        return d['tr'],d['va']
    from normalize_domain_round import normalize_frame,FLAGS
    a,ids=enrich(raw);b,_=enrich(other,ids);a=normalize_frame(a,raw);b=normalize_frame(b,other)
    tr,va=tr.copy(),va.copy()
    if not flags_only:
        oof=np.zeros(len(y))
        for it,iv in StratifiedGroupKFold(3,shuffle=True,random_state=2026).split(np.zeros(len(y)),y,g):oof[iv]=svm_probability(a.iloc[it],y[it],a.iloc[iv])
        tr['svm_oof_probability']=oof;va['svm_oof_probability']=svm_probability(a,y,b)
    for key,pat in FLAGS.items():
        tr['norm_'+key]=a.title_text.str.contains(pat,regex=True).to_numpy(dtype='int8')
        va['norm_'+key]=b.title_text.str.contains(pat,regex=True).to_numpy(dtype='int8')
    pd.to_pickle(dict(tr=tr,va=va,fingerprint=fingerprint),cache);return tr,va

def stack_features(raw,y,g,other,out,train_rows,eval_rows,deadline,base_count=3,embedding_pair=None):
    if base_count==5:
        from learned_stacking5_round import partition,base_predictions
        prefix='stacking5'
    elif base_count==4:
        from learned_stacking4_round import partition,base_predictions
        prefix='stacking4'
    else:
        from learned_stacking_round import partition,base_predictions
        prefix='stacking'
    if len(other)==14000:
        path=out.parent/f'{prefix}_seed731_base.npz'
        if path.exists():
            d=np.load(path)
            if not np.array_equal(d['train_rows'],train_rows) or not np.array_equal(d['holdout_rows'],eval_rows):raise ValueError('Stack confirmation split mismatch')
            return d['oof'],d['holdout']
        # Reconstruct the original row order so all three cache layers align.
        order=np.r_[train_rows,eval_rows];inverse=np.argsort(order)
        if not np.array_equal(np.sort(order),np.arange(len(order))):raise ValueError('Incomplete confirmation row indices')
        full=pd.concat([raw,other],ignore_index=True).iloc[inverse].reset_index(drop=True)
        yy=np.r_[y,(other.condition=='used').to_numpy(dtype='int8')][inverse];gg=np.r_[g,groups_for(other)][inverse]
        from learned_stacking_round import partition as partition3
        partition3(full,yy,gg,train_rows,eval_rows,out,731,deadline)
        if base_count>=4:
            from learned_stacking4_round import partition as partition4
            partition4(full,yy,gg,train_rows,eval_rows,out,731,deadline)
        if base_count==5:
            if embedding_pair is None:raise ValueError('Five-base stacking requires frozen embeddings')
            np.save(out/'embedding_titles.npy',np.concatenate(embedding_pair)[inverse])
        return partition(full,yy,gg,train_rows,eval_rows,out,731,deadline)
    oof=np.zeros((len(y),base_count))
    for it,iv in StratifiedGroupKFold(3,shuffle=True,random_state=2026).split(np.zeros(len(y)),y,g):
        extra={'train_embeddings':embedding_pair[0][it],'eval_embeddings':embedding_pair[0][iv]} if base_count==5 else {}
        oof[iv]=base_predictions(raw.iloc[it].reset_index(drop=True),y[it],g[it],raw.iloc[iv].reset_index(drop=True),deadline,**extra)
    extra={'train_embeddings':embedding_pair[0],'eval_embeddings':embedding_pair[1]} if base_count==5 else {}
    return oof,base_predictions(raw,y,g,other,deadline,**extra)

def predict_component(cfg,raw,y,g,other,out,all_embeddings,train_rows,eval_rows,deadline):
    family=cfg['family'];name=cfg['name']
    if family=='prior':return prior_probability('alternate_seed731' if len(other)==14000 else 'full_train')
    if family=='stacking':
        from learned_stacking_round import meta_fit
        oof,p=stack_features(raw,y,g,other,out,train_rows,eval_rows,deadline,cfg.get('base_count',3),all_embeddings)
        return meta_fit(oof,y,p,cfg['design'],cfg['C'])[0]
    if family=='embedding' and name.startswith('embedding_logistic'):
        from sklearn.linear_model import LogisticRegression
        m=LogisticRegression(C=cfg['params']['C'],solver='lbfgs',max_iter=600)
        m.fit(all_embeddings[0],y);return m.predict_proba(all_embeddings[1])[:,1]
    if family=='normalize' and name.startswith('normalize_svm'):
        from normalize_domain_round import normalize_frame
        a,ids=enrich(raw);b,_=enrich(other,ids)
        if name.endswith('domain'):a=normalize_frame(a,raw);b=normalize_frame(b,other)
        return svm_probability(a,y,b,C=cfg['params']['C'])
    tr,va=core(raw,y,g,other,out)
    tr,va=tr.copy(),va.copy()
    if family=='normalize':
        tr,va=normalized_features(tr,va,raw,y,g,other,out,cfg.get('feature_set')=='domain_flags_only')
    if family=='combined':
        from combined_round3_search import prepare_joint
        blocks=cfg['blocks'];normal=normalized_features(tr,va,raw,y,g,other,out) if blocks['normalize'] else None
        stack=stack_features(raw,y,g,other,out,train_rows,eval_rows,deadline) if blocks['stacking_base'] else (None,None)
        tr,va=prepare_joint(tr,va,*stack,all_embeddings if blocks['embedding'] else None,normal)
    if family=='embedding':
        cols=['title_embed_'+str(i) for i in range(all_embeddings[0].shape[1])]
        tr=pd.concat([tr,pd.DataFrame(all_embeddings[0],columns=cols)],axis=1);va=pd.concat([va,pd.DataFrame(all_embeddings[1],columns=cols)],axis=1)
    params=dict(cfg.get('params',cfg.get('global_params',{})));params['n_jobs']=2
    m=LGBMClassifier(**params);m.fit(tr,y);p=m.predict_proba(va)[:,1]
    if family=='category' and cfg.get('expert_weight',0)>0:
        from category_experts_round import fit_experts
        iso=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(deadline))
        expert,_,_=fit_experts(tr,y,va,raw,other,params,cfg.get('clusters'),cfg.get('min_rows',300),deadline=iso,normalize_centroids=cfg.get('normalize_centroids',False))
        mask=np.isfinite(expert);w=cfg['expert_weight'];p[mask]=(1-w)*p[mask]+w*expert[mask]
    return p

def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Global deadline')))
    signal.alarm(max(1,int(args.deadline-time.time())))
    reference=json.loads((ROOT/'results/round_v3/prior_prediction_metadata.json').read_text())
    for filename,key in [('train_data.jsonlines','train_sha256'),('test_data.jsonlines','test_sha256')]:
        with (Path(args.data_dir)/filename).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=reference[key]:raise ValueError('Frozen reference belongs to another dataset: '+filename)
    if args.mode=='select':
        records=[]
        with (Path(args.data_dir)/'train_data.jsonlines').open() as f:
            for line in f:
                row=json.loads(line);records.append({k:row.get(k) for k in ['id','parent_item_id','condition']})
        raw=pd.DataFrame(records);test=None
    else:raw,test,_=load_data(args.data_dir,out)
    y=(raw.condition=='used').to_numpy(dtype='int8');g=groups_for(raw)
    t,v=split(y,g,42,2)
    selected=select(out,v,y[v]) if args.mode=='select' else json.loads((out/'round3_selected.json').read_text())
    if args.mode=='select':print(json.dumps(selected),flush=True);return
    if args.mode=='confirm':
        del test;gc.collect()
        t,v=split(y,g,731,2);other=raw.iloc[v].reset_index(drop=True);fit_raw=raw.iloc[t].reset_index(drop=True);fit_y=y[t];fit_g=g[t];protocol='confirmation'
    else:
        if not selected.get('promote_over_previous'):raise ValueError('Refit requires a confirmed improvement')
        t=np.arange(len(raw));v=np.arange(len(test));fit_raw=raw;other=test;fit_y=y;fit_g=g;protocol='full'
    work=out/protocol;work.mkdir(parents=True,exist_ok=True);emb=None
    if any(c['family']=='embedding' or c.get('blocks',{}).get('embedding') or c.get('base_count')==5 for c in selected['components']):
        train_emb=np.load(out/'embedding_titles.npy')
        if protocol=='confirmation':emb=(train_emb[t],train_emb[v])
        else:
            from embedding_round import encode_titles
            eval_emb=encode_titles(other.title.fillna('').astype(str).tolist(),args.model_dir,work,args.deadline)
            emb=(train_emb,eval_emb)
    probabilities=[]
    for cfg in selected['components']:
        print('FROZEN',protocol,cfg['name'],flush=True)
        p=predict_component(cfg,fit_raw,fit_y,fit_g,other,work,emb,t,v,args.deadline);probabilities.append(p)
        np.save(work/(cfg['name']+'.npy'),p);gc.collect()
    p=sum(w*z for w,z in zip(selected['weights'],probabilities))
    if protocol=='confirmation':
        selected['confirmation']=metrics(y[v],p,selected['threshold']);selected['status']='confirmed'
        baseline=prior_probability('alternate_seed731');selected['prior_confirmation']=metrics(y[v],baseline,.5)
        selected['promote_over_previous']=selected['confirmation']['accuracy']>selected['prior_confirmation']['accuracy'] and selected['accuracy']>selected['prior_selection_accuracy']
        selected['goal_met_local']=selected['confirmation']['accuracy']>=.95 and selected['accuracy']>=.95
        pd.DataFrame({'row_index':v,'y_used':y[v],'probability_used':p,'prior_probability':baseline}).to_csv(out/'round3_confirmation_predictions.csv',index=False)
    else:
        sub=pd.DataFrame({'ID':np.arange(1,len(p)+1),'condition':np.where(p>=selected['threshold'],'used','new')})
        assert len(sub)==30000 and sub.ID.is_unique
        sub.to_csv(out/'submission_round3.csv',index=False);selected['submission']='results/round_v3/submission_round3.csv'
    (out/'round3_selected.json').write_text(json.dumps(selected,indent=2));print(json.dumps(selected),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',default=str(ROOT/'results/round_v3'));p.add_argument('--mode',choices=['select','confirm','refit'],default='select');p.add_argument('--deadline',type=float,required=True);p.add_argument('--model-dir',default=str(ROOT.parent/'models/title_minilm'));args=p.parse_args()
    main(args)

