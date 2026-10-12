"""TabNet grouped holdout experiment; uses frozen text vectors and typed categories."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score
from neural_round import prepare

def train_predict(tr,va,y,train_embeddings=None,eval_embeddings=None,epochs=18,validation_y=None):
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
    torch.set_num_threads(2)
    ct,cv,xt,xv,_,_=prepare(tr,va,train_embeddings,eval_embeddings)
    cat_cols=[c for c in tr if str(tr[c].dtype)=='category'];dims=[len(tr[c].cat.categories)+1 for c in cat_cols]
    cats_t=np.column_stack([tr[c].cat.codes.to_numpy(dtype=np.int64)+1 for c in cat_cols]).astype(np.float32);cats_v=np.column_stack([va[c].cat.codes.to_numpy(dtype=np.int64)+1 for c in cat_cols]).astype(np.float32)
    X=np.column_stack([xt,cats_t]).astype(np.float32);V=np.column_stack([xv,cats_v]).astype(np.float32);idx=list(range(xt.shape[1],X.shape[1]))
    model=TabNetClassifier(n_d=24,n_a=24,n_steps=4,gamma=1.3,lambda_sparse=1e-4,cat_idxs=idx,cat_dims=dims,cat_emb_dim=4,optimizer_fn=torch.optim.Adam,optimizer_params={'lr':0.008,'weight_decay':1e-5},mask_type='entmax',device_name='cpu',verbose=0,seed=42)
    opts=dict(X_train=X,y_train=np.asarray(y,dtype=np.int64),max_epochs=int(epochs),batch_size=1024,virtual_batch_size=128,num_workers=0,drop_last=False)
    if validation_y is not None:
        opts.update(eval_set=[(V,np.asarray(validation_y,dtype=np.int64))],eval_name=['valid'],eval_metric=['logloss'],patience=4)
    model.fit(**opts);p=model.predict_proba(V)[:,1]
    return p,int(getattr(model,'best_epoch',epochs))

def main(a):
    from competition_pipeline_v2 import load_data,metrics
    from sklearn.metrics import accuracy_score
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);d=pd.read_pickle(a.cache);tr,va=d['tr'],d['va'];t,v=d['t'],d['v'];raw,_,_=load_data(a.data_dir,out);y=(raw.condition=='used').to_numpy(dtype=np.int8)
    rows=pd.read_csv(Path(a.cache).parent/'selection_rows.csv')
    if not np.array_equal(rows.row_index.to_numpy(),v) or not np.array_equal(rows.y_used.to_numpy(),y[v]):raise ValueError('frozen split mismatch')
    emb=np.load(a.embeddings) if a.embeddings and Path(a.embeddings).exists() else None
    p,epoch=train_predict(tr,va,y[t],emb[t] if emb is not None else None,emb[v] if emb is not None else None,a.epochs,validation_y=y[v])
    q=max(np.arange(.45,.551,.005),key=lambda z:accuracy_score(y[v],p>=z));m=metrics(y[v],p,float(q))
    rec=dict(name='neural_tabnet',family='neural',model_type='tabnet',accuracy=m['accuracy'],balanced_accuracy=m['balanced_accuracy'],macro_f1=m['macro_f1'],logloss=m['logloss'],threshold=float(q),best_epoch=epoch,epochs=a.epochs,seed=42,feature_set='tabular_plus_title_embeddings' if emb is not None else 'tabular',split_seed=42,selection_only=True,confirmation=None,device='cpu')
    np.save(out/'neural_tabnet.npy',p);(out/'neural_tabnet.json').write_text(json.dumps(rec,indent=2));pd.DataFrame({'row_index':v,'y_used':y[v],'neural_tabnet':p}).to_csv(out/'neural_tabnet_predictions.csv',index=False);print('RESULT',json.dumps(rec),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--cache',required=True);p.add_argument('--output',required=True);p.add_argument('--embeddings');p.add_argument('--epochs',type=int,default=18);main(p.parse_args())
