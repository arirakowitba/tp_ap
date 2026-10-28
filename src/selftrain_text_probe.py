import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import pickle,json,time
from pathlib import Path
import numpy as np
from scipy.sparse import hstack,vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
R=Path(__file__).resolve().parent
D=pickle.load(open(R/'work20/original_docs.pkl','rb'))
docs=D.docs.fillna('').to_numpy(dtype=str);y=D.y.to_numpy()[:70000].astype(np.int8)
seed=int(os.environ.get('AP_SELFTRAIN_SEED','883'));va=np.load(R/f'work20/round30_hist_group_{seed}.npz')['rows'];tr=np.setdiff1d(np.arange(70000),va);test=np.arange(70000,100000)
features=[];preds=[];models=[]
for analyzer,ngram in [('word',(1,2)),('char_wb',(2,5))]:
 st=time.time();v=TfidfVectorizer(analyzer=analyzer,ngram_range=ngram,min_df=3,max_features=100000,sublinear_tf=True,strip_accents='unicode',dtype=np.float32)
 x=v.fit_transform(docs[tr]);xv=v.transform(docs[va]);xt=v.transform(docs[test])
 m=LogisticRegression(C=2,solver='liblinear',max_iter=300);m.fit(x,y[tr]);pv=m.predict_proba(xv)[:,1];pt=m.predict_proba(xt)[:,1]
 features.append((x,xv,xt));preds.append((pv,pt));print(analyzer,'features',x.shape[1],'elapsed',round(time.time()-st,1),flush=True)
pv0,pv1=preds[0][0],preds[1][0];pt0,pt1=preds[0][1],preds[1][1]
for th in [.99,.995]:
 hi=(pt0>=th)&(pt1>=th);lo=(pt0<=1-th)&(pt1<=1-th);sel=hi|lo
 vh=(pv0>=th)&(pv1>=th);vl=(pv0<=1-th)&(pv1<=1-th);valid=vh|vl
 print('selection',th,int(sel.sum()),int(hi.sum()),int(lo.sum()),'eval',int(valid.sum()),'errors',int(np.sum(valid&(vh!=y[va]))),flush=True)
x=hstack([features[0][0],features[1][0]]).tocsr();xv=hstack([features[0][1],features[1][1]]).tocsr();xt=hstack([features[0][2],features[1][2]]).tocsr()
base=LogisticRegression(C=2,solver='liblinear',max_iter=300);base.fit(x,y[tr]);pbase=base.predict_proba(xv)[:,1]
result={'seed':seed,'n_valid':len(va),'selection':{},'baseline_acc':float(np.mean((pbase>=.5)==y[va])),'baseline_log_loss':float(log_loss(y[va],pbase))}
for th in [.99,.995]:
 hi=(pt0>=th)&(pt1>=th);lo=(pt0<=1-th)&(pt1<=1-th);sel=hi|lo
 result['selection'][str(th)]={'n':int(sel.sum()),'used':int(hi.sum()),'new':int(lo.sum()),'models':{}}
 if sel.sum()<100:continue
 aug=vstack([x,xt[sel]]).tocsr();lab=np.r_[y[tr],hi[sel].astype(np.int8)]
 for weight in [.25,1.0]:
  sw=np.r_[np.ones(len(tr)),np.full(sel.sum(),weight)]
  m=LogisticRegression(C=2,solver='liblinear',max_iter=300);m.fit(aug,lab,sample_weight=sw);p=m.predict_proba(xv)[:,1]
  result['selection'][str(th)]['models'][str(weight)]={'acc':float(np.mean((p>=.5)==y[va])),'delta':int(np.sum((p>=.5)==y[va])-np.sum((pbase>=.5)==y[va])),'log_loss':float(log_loss(y[va],p)),'changes':int(np.sum((p>=.5)!=(pbase>=.5)))}
  np.savez_compressed(R/f'selftrain_text_{seed}_{th}_{weight}.npz',rows=va,y=y[va],p=p,p_base=pbase)
 print('trained',th,result['selection'][str(th)],flush=True)
(R/f'selftrain_text_{seed}.json').write_text(json.dumps(result,indent=2)+'\n')
