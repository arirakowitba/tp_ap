"""Grouped validation for a categorical/numeric PyTorch MLP fused with frozen title embeddings."""
from pathlib import Path
import argparse,json,time
import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,log_loss


def prepare(tr,va,train_embeddings=None,eval_embeddings=None):
    cats=[c for c in tr if str(tr[c].dtype)=='category']
    nums=[c for c in tr if c not in cats]
    card=np.asarray([len(tr[c].cat.categories) for c in cats],dtype=np.int64)
    offsets=np.r_[0,np.cumsum(card+1)[:-1]]
    def catcodes(df):
        if not cats:return np.zeros((len(df),0),np.int64)
        return np.column_stack([df[c].cat.codes.to_numpy(dtype=np.int64)+1 for c in cats])+offsets
    a=np.column_stack([pd.to_numeric(tr[c],errors='coerce').to_numpy(dtype=np.float32) for c in nums]) if nums else np.zeros((len(tr),0),np.float32)
    b=np.column_stack([pd.to_numeric(va[c],errors='coerce').to_numpy(dtype=np.float32) for c in nums]) if nums else np.zeros((len(va),0),np.float32)
    if train_embeddings is not None:
        a=np.column_stack([a,np.asarray(train_embeddings,dtype=np.float32)]);b=np.column_stack([b,np.asarray(eval_embeddings,dtype=np.float32)])
    med=np.nanmedian(np.where(np.isfinite(a),a,np.nan),axis=0);med=np.nan_to_num(med,nan=0.,posinf=0.,neginf=0.)
    a=np.where(np.isfinite(a),a,med);b=np.where(np.isfinite(b),b,med)
    mean=a.mean(0);std=a.std(0);std=np.where(std<1e-6,1.,std)
    a=np.clip((a-mean)/std,-10,10).astype(np.float32);b=np.clip((b-mean)/std,-10,10).astype(np.float32)
    return catcodes(tr),catcodes(va),a,b,int(card.sum()+len(card)),len(cats)


def train_predict(tr,va,y,train_embeddings=None,eval_embeddings=None,epochs=12,seed=42,deadline=None,validation_y=None):
    import torch
    from torch import nn
    torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed)
    ct,cv,xt,xv,cat_total,ncat=prepare(tr,va,train_embeddings,eval_embeddings)
    dim=8 if ncat else 0
    class Net(nn.Module):
        def __init__(self):
            super().__init__();self.emb=nn.Embedding(max(1,cat_total),dim,padding_idx=0) if ncat else None
            width=xt.shape[1]+dim*ncat
            self.net=nn.Sequential(nn.Linear(width,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(.40),nn.Linear(512,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(.35),nn.Linear(256,1))
        def forward(self,c,x):
            z=torch.cat([x,self.emb(c).flatten(1)],1) if self.emb is not None else x
            return self.net(z).squeeze(1)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=Net().to(device)
    train_ds=torch.utils.data.TensorDataset(torch.from_numpy(ct),torch.from_numpy(xt),torch.from_numpy(np.asarray(y,dtype=np.float32)))
    loader=torch.utils.data.DataLoader(train_ds,batch_size=1024,shuffle=True,drop_last=True)
    optim=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-3);lossfn=nn.BCEWithLogitsLoss()
    cvalid=torch.from_numpy(cv).to(device);xvalid=torch.from_numpy(xv).to(device);best=None;bestloss=float('inf');wait=0;bestepoch=0
    for epoch in range(1,epochs+1):
        if deadline and time.time()>deadline:break
        model.train()
        for c,x,target in loader:
            optim.zero_grad(set_to_none=True);loss=lossfn(model(c.to(device),x.to(device)),target.to(device));loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step()
        model.eval()
        with torch.no_grad():proba=torch.sigmoid(model(cvalid,xvalid)).cpu().numpy()
        if validation_y is None:best=proba;bestepoch=epoch
        else:
            score=log_loss(np.asarray(validation_y,dtype=np.int8),proba)
            if score<bestloss-1e-5:bestloss=score;best=proba.copy();bestepoch=epoch;wait=0
            else:wait+=1
            print('EPOCH',epoch,'valid_logloss',round(score,6),'best',bestepoch,flush=True)
            if wait>=3:break
    if best is None:raise TimeoutError('No neural epoch completed before deadline')
    return best,bestepoch

def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);d=pd.read_pickle(args.cache);tr,va=d['tr'],d['va'];t,v=d['t'],d['v']
    from competition_pipeline_v2 import load_data,groups_for,metrics
    raw,_,audit=load_data(args.data_dir,out);y=(raw.condition=='used').to_numpy(dtype=np.int8)
    if not np.array_equal(t,d['t']) or not np.array_equal(v,d['v']):raise ValueError('cached grouped split mismatch')
    check=pd.read_csv(Path(args.cache).parent/'selection_rows.csv')
    if not np.array_equal(check.row_index.to_numpy(),v) or not np.array_equal(check.y_used.to_numpy(),y[v]):raise ValueError('selection rows mismatch')
    embs=np.load(args.embeddings) if args.embeddings and Path(args.embeddings).exists() else None
    p,epoch=train_predict(tr,va,y[t],embs[t] if embs is not None else None,embs[v] if embs is not None else None,args.epochs,args.seed,args.deadline,y[v])
    threshold=max(np.arange(.45,.551,.005),key=lambda q:accuracy_score(y[v],p>=q));m=metrics(y[v],p,float(threshold))
    rec=dict(name='neural_mlp',family='neural',accuracy=m['accuracy'],balanced_accuracy=m['balanced_accuracy'],macro_f1=m['macro_f1'],logloss=m['logloss'],threshold=float(threshold),best_epoch=epoch,epochs=args.epochs,seed=args.seed,feature_set='tabular_plus_title_embeddings' if embs is not None else 'tabular',split_seed=42,selection_only=True,confirmation=None,device='cuda' if __import__('torch').cuda.is_available() else 'cpu')
    np.save(out/'neural_mlp.npy',p);(out/'neural_mlp.json').write_text(json.dumps(rec,indent=2));pd.DataFrame({'row_index':v,'y_used':y[v],'neural_mlp':p}).to_csv(out/'neural_mlp_predictions.csv',index=False)
    print('RESULT',json.dumps(rec),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--cache',required=True);p.add_argument('--output',required=True);p.add_argument('--embeddings');p.add_argument('--epochs',type=int,default=12);p.add_argument('--seed',type=int,default=42);p.add_argument('--deadline',type=float);main(p.parse_args())
