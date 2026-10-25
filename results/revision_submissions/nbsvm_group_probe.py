import os
for x in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[x]='2'
import json,pickle,time
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
R=Path(__file__).resolve().parent
D=pickle.load(open(R/'work20/original_docs.pkl','rb'))
docs=D.docs.fillna('').to_numpy(dtype=str)[:70000];y=D.y.to_numpy(dtype=np.int8)[:70000]
results={}
for seed in [883,2026]:
 start=time.time();base=np.load(R/f'work20/round30_hist_group_{seed}.npz');va=base['rows'].astype(int);tr=np.setdiff1d(np.arange(70000),va,assume_unique=True)
 assert np.array_equal(y[va],base['y'])
 vec=TfidfVectorizer(analyzer='word',ngram_range=(1,2),min_df=3,max_features=90000,sublinear_tf=True,strip_accents='unicode',dtype=np.float32)
 x=vec.fit_transform(docs[tr]);xt=vec.transform(docs[va]);pos=x[y[tr]==1].sum(axis=0);neg=x[y[tr]==0].sum(axis=0)
 r=np.log((np.asarray(pos).ravel()+1)/(int((y[tr]==1).sum())+1))-np.log((np.asarray(neg).ravel()+1)/(int((y[tr]==0).sum())+1))
 xx=x.multiply(r).tocsr();xxv=xt.multiply(r).tocsr()
 model=LogisticRegression(C=1.0,solver='liblinear',max_iter=300)
 model.fit(xx,y[tr]);p=model.predict_proba(xxv)[:,1];bp=base['p'];yy=y[va]
 out={'n':len(va),'features':x.shape[1],'model_acc_05':float(np.mean((p>=.5)==yy)),'model_logloss':float(log_loss(yy,p)),'hist_acc_0475':float(np.mean((bp>=.475)==yy)),'hist_logloss':float(log_loss(yy,bp))}
 for w in [.1,.2,.3,.5]:
  mix=(1-w)*bp+w*p;out[f'blend_{w}']={'acc_0475':float(np.mean((mix>=.475)==yy)),'fix':int(np.sum((mix>=.475)==yy)-np.sum((bp>=.475)==yy)),'changes':int(np.sum((mix>=.475)!=(bp>=.475)))}
 np.savez_compressed(R/f'nbsvm_group_{seed}.npz',rows=va,y=yy,p_nb=p,p_hist=bp)
 out['elapsed_sec']=round(time.time()-start,1);results[str(seed)]=out;print(seed,json.dumps(out),flush=True)
(R/'nbsvm_group_probe.json').write_text(json.dumps(results,indent=2)+'\n')
