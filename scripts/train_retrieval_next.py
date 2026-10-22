"""Train-only text retrieval and seller priors, development-selected blend."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score,log_loss
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits
threadpool_limits(2)
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/training_next/retrieval';OUT.mkdir(parents=True,exist_ok=True)
def say(*a):print(*a,flush=True)
def metrics(y,p,seen):
 p=np.clip(p,1e-5,1-1e-5)
 return {'accuracy':float(accuracy_score(y,p>=.5)),'log_loss':float(log_loss(y,p)), 'seen_accuracy':float(accuracy_score(y[seen],p[seen]>=.5)), 'unseen_accuracy':float(accuracy_score(y[~seen],p[~seen]>=.5))}
def prior(train,query,y,col,strength):
 z=pd.DataFrame({'key':train[col].astype(str).to_numpy(),'y':y});s=z.groupby('key').y.agg(['sum','count'])
 return query[col].astype(str).map((s['sum']+strength*y.mean())/(s['count']+strength)).fillna(y.mean()).to_numpy()
def retrieve(raw,y,tr,qr,label):
 t=time.time();cache=OUT/(label+'_candidates.npz')
 if cache.exists():
  saved=np.load(cache)
  assert np.array_equal(saved['rows'],qr)
  return {k:saved[k] for k in saved.files if k!='rows'}
 train=raw.iloc[tr];query=raw.iloc[qr]
 corpus=(raw.title.fillna('')+' warranty '+raw.warranty.fillna('')).astype(str)
 vectors=[]
 for kind,kw,weight in [('word',dict(analyzer='word',ngram_range=(1,2),max_features=60000,min_df=2),.6),('char',dict(analyzer='char_wb',ngram_range=(3,5),max_features=90000,min_df=3),.8)]:
  v=TfidfVectorizer(dtype=np.float32,sublinear_tf=True,strip_accents='unicode',**kw)
  x=v.fit_transform(corpus.iloc[tr]);q=v.transform(corpus.iloc[qr]);vectors.append((weight*x,weight*q));say(label,kind,x.shape,time.time()-t)
 X=normalize(hstack([v[0] for v in vectors]).tocsr());Q=normalize(hstack([v[1] for v in vectors]).tocsr())
 ps={f'{c}_k{k}_pow{power}':np.empty(len(qr),dtype=np.float32) for c in ['global','category'] for k in [8,32,128] for power in [2,6]}
 cats=train.category_id.to_numpy();qcats=query.category_id.to_numpy();yt=y[tr];mu=yt.mean()
 for start in range(0,len(qr),96):
  end=min(start+96,len(qr));sim=(Q[start:end]@X.T).toarray()
  ids=np.argpartition(sim,-128,axis=1)[:,-128:];vals=np.take_along_axis(sim,ids,axis=1);order=np.argsort(-vals,axis=1);ids=np.take_along_axis(ids,order,axis=1);vals=np.take_along_axis(vals,order,axis=1)
  for scope in ['global','category']:
   for k in [8,32,128]:
    s=vals[:,:k];ix=ids[:,:k]
    for power in [2,6]:
     w=s**power
     if scope=='category':w=w*(cats[ix]==qcats[start:end,None])
     # One prior observation stabilizes weak matches.
     ps[f'{scope}_k{k}_pow{power}'][start:end]=((w*yt[ix]).sum(axis=1)+mu)/(w.sum(axis=1)+1)
  if start%1920==0:say(label,'neighbors',start,len(qr),round(time.time()-t,1))
 for strength in [1,5,20]:ps[f'seller_{strength}']=prior(train,query,yt,'seller_id',strength)
 ps['category_prior']=prior(train,query,yt,'category_id',10)
 np.savez_compressed(OUT/(label+'_candidates.npz'),rows=qr,**ps)
 say(label,'done',time.time()-t)
 return ps
def load_minimal(path):
 cols=['title','warranty','category_id','seller_id','condition']
 with path.open() as f:
  return pd.DataFrame([{k:r.get(k) for k in cols} for r in map(json.loads,f)])
def main():
 train=load_minimal(ROOT.parent/'recovered/data/train_data.jsonlines');test=load_minimal(ROOT.parent/'recovered/data/test_data.jsonlines')
 raw=pd.concat([train,test],ignore_index=True);y=train.condition.eq('used').to_numpy(dtype=int)
 split=np.load(ROOT/'results/new_models_20261021/partitions.npz');fit,tune,confirm=[split[k] for k in ['fit','tune','confirm']];dev=np.r_[fit,tune]
 pt=retrieve(raw,y,fit,tune,'tune')
 seen=np.isin(raw.seller_id.iloc[tune],raw.seller_id.iloc[fit]);grid=[{'key':k,**metrics(y[tune],p,seen)} for k,p in pt.items()]
 best=max(grid,key=lambda z:(z['accuracy'],-z['log_loss']));weight=.1
 declaration={'selected_standalone_on_tune':best,'grid':grid,'baseline_reproduction':'Failed OOM exit137. No baseline tune vector available.','blend_weight':weight,'selection_policy':'Select standalone by tune accuracy, logloss tiebreak. Blend weight .1 fixed before reading confirmation labels; not selected on confirmation.'}
 (OUT/'tune_selection.json').write_text(json.dumps(declaration,indent=2));say('PREDECLARED',declaration)
 pc=retrieve(raw,y,dev,confirm,'confirm');bc=np.load(ROOT/'results/new_models_20261021/seller_native_confirm.npz');assert np.array_equal(bc['rows'],confirm)
 p=(1-weight)*bc['p']+weight*pc[best['key']];seen=np.isin(raw.seller_id.iloc[confirm],raw.seller_id.iloc[dev])
 report={'selection':declaration,'baseline_confirm':metrics(y[confirm],bc['p'],seen),'standalone_confirm':metrics(y[confirm],pc[best['key']],seen),'blend_confirm':metrics(y[confirm],p,seen),'protocol':'Fit49k/tune7k; standalone selected on tune; refit56k/confirm14k. Weight .1 fixed before confirmation analysis because baseline tune reproduction OOM. No test/probe labels. Text vocabulary and neighbor labels trained only on fit/dev. Confirmation historically reused by previous work, not virgin.'}
 a=(bc['p']>=.5)==y[confirm];c=(p>=.5)==y[confirm];gain=int((~a&c).sum());loss=int((a&~c).sum())
 report['paired_diagnostic']={'corrected':gain,'broken':loss,'net':gain-loss,'mcnemar_exact_two_sided_p':float(binomtest(gain,gain+loss).pvalue) if gain+loss else 1.,'scope':'Descriptive only: reused confirmation and many project experiments. No proof of public or hidden improvement.'}
 np.savez_compressed(OUT/'selected_confirm.npz',rows=confirm,p=p,retrieval_p=pc[best['key']],y=y[confirm],seen=seen)
 (OUT/'report.json').write_text(json.dumps(report,indent=2));say('REPORT',report)
 if report['blend_confirm']['accuracy']>report['baseline_confirm']['accuracy']:
  ptest=retrieve(raw,y,np.arange(70000),np.arange(70000,100000),'test');native=np.load(ROOT/'results/new_models_20261021/seller_native_test.npz')['test'];p=(1-weight)*native+weight*ptest[best['key']];pd.DataFrame({'ID':np.arange(1,30001),'p_used':p}).to_csv(OUT/'selected_test_predictions.csv',index=False)
 else:say('No full refit: blend does not improve baseline confirmation accuracy')
 (OUT/'status.json').write_text(json.dumps({'stage':'complete','full_refit':report['blend_confirm']['accuracy']>report['baseline_confirm']['accuracy']},indent=2))
 del train,test,raw,pt,pc
if __name__=='__main__':main()
