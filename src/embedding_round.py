"""Real pretrained multilingual title embeddings, pooled locally using OpenVINO CPU and quantized ONNX weights.

No training text or image is sent to an external inference service. The pretrained
weights are downloaded separately; models/tokenizers and derived caches are ignored
by git. Validation rows reuse the frozen grouped split from the competition pipeline.
"""
from pathlib import Path
import argparse,json,time,signal,hashlib,os,gc
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
MODEL_ID='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
MODEL_FILE='onnx/model_quint8_avx2.onnx'

def guard(deadline):
    if deadline and time.time()>=deadline:raise TimeoutError('Hard deadline reached')

def title_data(folder):
    titles=[];labels=[];groups=[];digest=hashlib.sha256()
    with (Path(folder)/'train_data.jsonlines').open('rb') as f:
        for line in f:
            digest.update(line);r=json.loads(line);titles.append(str(r.get('title') or ''))
            labels.append(int(r['condition']=='used'));groups.append(str(r['parent_item_id']) if r.get('parent_item_id') is not None else '__ROW__'+str(r['id']))
    if len(titles)!=70000:raise ValueError('Expected 70,000 complete training rows')
    return titles,np.asarray(labels,dtype=np.int8),np.asarray(groups),digest.hexdigest()

def encode_titles(titles,model_dir,out,deadline=None,batch_size=64):
    import openvino as ov
    from tokenizers import Tokenizer
    # Use the direct C++ CPU Core only; ONNXRuntime is intentionally never imported.
    # The optional openvino_telemetry package is not installed or used.
    tokenizer=Tokenizer.from_file(str(Path(model_dir)/'tokenizer.json'))
    tokenizer.enable_truncation(max_length=128);tokenizer.enable_padding()
    core=ov.Core();model=core.read_model(str(Path(model_dir)/MODEL_FILE))
    session=core.compile_model(model,'CPU',{'INFERENCE_NUM_THREADS':2,'NUM_STREAMS':1})
    names={i.any_name for i in session.inputs};embeddings=np.zeros((len(titles),384),dtype=np.float32)
    # Sort only for efficient dynamic padding; restore exact input order afterward.
    order=sorted(range(len(titles)),key=lambda i:len(titles[i]))
    started=time.time()
    for start in range(0,len(order),batch_size):
        guard(deadline);idx=order[start:start+batch_size];enc=tokenizer.encode_batch([titles[i] for i in idx])
        z={k:np.asarray([getattr(e,p) for e in enc],dtype=np.int64) for k,p in [('input_ids','ids'),('attention_mask','attention_mask'),('token_type_ids','type_ids')]}
        hidden=session({k:v for k,v in z.items() if k in names})[session.output(0)]
        mask=z['attention_mask'][...,None];pooled=(hidden*mask).sum(axis=1)/np.maximum(mask.sum(axis=1),1)
        embeddings[idx]=pooled/np.maximum(np.linalg.norm(pooled,axis=1,keepdims=True),1e-12)
        if start%4096==0:print('EMBEDDING',start,len(titles),'seconds',round(time.time()-started),flush=True)
    pending=out/'.embedding_titles.pending.npy'
    np.save(pending,embeddings);pending.replace(out/'embedding_titles.npy')
    return embeddings

def append_embedding_features(tr,va,embeddings,train_rows,valid_rows):
    """Append frozen unsupervised vectors for any grouped split (including seed731)."""
    names=['title_embed_'+str(i) for i in range(384)]
    if embeddings.ndim!=2 or embeddings.shape[1]!=384:raise ValueError('Expected 384-dimensional embeddings')
    left=pd.concat([tr.reset_index(drop=True),pd.DataFrame(embeddings[train_rows],columns=names)],axis=1)
    right=pd.concat([va.reset_index(drop=True),pd.DataFrame(embeddings[valid_rows],columns=names)],axis=1)
    if len(left)!=len(train_rows) or len(right)!=len(valid_rows):raise ValueError('Split shape mismatch')
    return left,right

def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.deadline:
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Hard deadline reached')))
            signal.alarm(max(1,int(args.deadline-time.time())))
    titles,y,g,fingerprint=title_data(args.data_dir)
    weight_sha256=hashlib.sha256((Path(args.model_dir)/MODEL_FILE).read_bytes()).hexdigest()
    metadata=dict(model_id=MODEL_ID,onnx_file=MODEL_FILE,weight_sha256=weight_sha256,pretrained=True,embedding_dimensions=384,
                  pooling='attention-masked mean with L2 normalization',max_tokens=128,n_rows=len(y),
                  train_sha256=fingerprint,local_inference=True,external_data_upload=False,split_seed=42,
                  max_inference_threads=2,inference_backend='OpenVINO CPU',onnxruntime_used=False,telemetry_package_used=False)
    metadata_path=out/'embedding_metadata.json'
    embpath=out/'embedding_titles.npy'
    if embpath.exists():
        if not metadata_path.exists():raise ValueError('Embedding cache has no integrity metadata')
        old_metadata=json.loads(metadata_path.read_text())
        if old_metadata.get('train_sha256')!=fingerprint or old_metadata.get('model_id')!=MODEL_ID or old_metadata.get('weight_sha256')!=weight_sha256:
            raise ValueError('Embedding cache belongs to different data or model')
    metadata_path.write_text(json.dumps(metadata,indent=2))
    embeddings=np.load(embpath) if embpath.exists() else encode_titles(titles,args.model_dir,out,args.deadline)
    if embeddings.shape!=(len(y),384):raise ValueError('Embedding cache shape mismatch')
    del titles;gc.collect()
    from optimize_leaderboard import split
    from competition_pipeline_v2 import metrics
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from lightgbm import LGBMClassifier,early_stopping
    t,v=split(y,g,42,2);records=[]
    selection=Path(args.cache)/'selection_rows.csv'
    if selection.exists():
        expected=pd.read_csv(selection)
        if not np.array_equal(expected.row_index.to_numpy(),v) or not np.array_equal(expected.y_used.to_numpy(),y[v]):
            raise ValueError('Embedding split does not match frozen baseline selection rows')
    if not np.isfinite(embeddings).all():raise ValueError('Non-finite embedding values')
    if not np.allclose(np.linalg.norm(embeddings,axis=1),1,atol=1e-4):raise ValueError('Embedding vectors are not normalized')
    def save(name,p,params,feature_set):
        q=float(max(np.arange(.46,.541,.005),key=lambda z:accuracy_score(y[v],p>=z)))
        row=dict(name=name,**metrics(y[v],p,q),params=params,feature_set=feature_set,split_seed=42,
                 model_id=MODEL_ID,confirmation=None,selection_only=True,
                 estimator='LogisticRegression' if feature_set=='embedding_only' else 'LGBMClassifier')
        np.save(out/(name+'.npy'),p);(out/(name+'.json')).write_text(json.dumps(row,indent=2));records.append(row)
        pd.DataFrame([{k:z for k,z in r.items() if k not in ('params',)} for r in records]).to_csv(out/'embedding_metrics.csv',index=False)
        pd.DataFrame({'row_index':v,'y_used':y[v],name:p}).to_csv(out/(name+'_predictions.csv'),index=False)
        print('RESULT',name,row['accuracy'],flush=True)
    for c in [.3,3,30]:
        guard(args.deadline);model=LogisticRegression(C=c,max_iter=600,solver='lbfgs',n_jobs=2)
        model.fit(embeddings[t],y[t]);save('embedding_logistic_'+str(c).replace('.','p'),model.predict_proba(embeddings[v])[:,1],{'C':c},'embedding_only')
    cache=Path(args.cache)/'matrices.pkl'
    if not cache.exists():
        print('Base cache absent: embeddings-only comparisons saved; rerun to test incremental LGBM.',flush=True);return
    d=pd.read_pickle(cache)
    if d.get('fingerprint')!=fingerprint:raise ValueError('Base cache belongs to different data')
    tr=d['tr'].copy();va=d['va'].copy();del d;gc.collect()
    tr,va=append_embedding_features(tr,va,embeddings,t,v)
    pending=out/'.embedding_matrices.pending.pkl'
    pd.to_pickle(dict(tr=tr,va=va,fingerprint=fingerprint),pending);pending.replace(out/'embedding_matrices.pkl')
    for i,param_path in enumerate([ROOT/'optimization/lgb_tpe_2_params.json',ROOT/'results/round_v2/lgb_context_svm_tpe_12.json']):
        guard(args.deadline)
        if not param_path.exists():continue
        cfg=json.loads(param_path.read_text());cfg=cfg.get('params',cfg);cfg.update(n_jobs=2,n_estimators=2200,verbosity=-1)
        model=LGBMClassifier(**cfg);model.fit(tr,y[t],eval_set=[(va,y[v])],eval_metric='binary_logloss',callbacks=[early_stopping(100,verbose=False)])
        cfg['n_estimators']=int(model.best_iteration_);save('embedding_lgb_'+str(i),model.predict_proba(va)[:,1],cfg,'base_plus_embedding')
        pd.DataFrame({'feature':tr.columns,'gain':model.booster_.feature_importance('gain')}).sort_values('gain',ascending=False).to_csv(out/('embedding_lgb_'+str(i)+'_importance.csv'),index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--model-dir',required=True)
    p.add_argument('--output',default=str(ROOT/'results/round_v3'));p.add_argument('--cache',default=str(ROOT/'results/round_v3/cache_base'));p.add_argument('--deadline',type=float)
    args=p.parse_args()
    try:run(args)
    except TimeoutError as exc:print(str(exc),flush=True);raise SystemExit(124)
