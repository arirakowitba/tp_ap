"""NB-weighted sparse logistic text challenger; all label statistics fit-only.
Tune selects C/representation. Confirmation assesses once, including a predeclared
85% native/15% NB blend if native tune vector is unavailable. No test labels used.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,time,gc
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,log_loss
from threadpoolctl import threadpool_limits
threadpool_limits(2)
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/training_next/nbsvm';OUT.mkdir(parents=True,exist_ok=True)
START=time.time()
def log(s):print(s,flush=True)
def metric(y,p,seen=None):
 d={'accuracy':float(accuracy_score(y,p>=.5)),'log_loss':float(log_loss(y,p))}
 if seen is not None:
  d.update(seen_accuracy=float(accuracy_score(y[seen],p[seen]>=.5)),unseen_accuracy=float(accuracy_score(y[~seen],p[~seen]>=.5)))
 return d
def read(path):
 docs=[];ys=[];sellers=[]
 for line in open(path):
  x=json.loads(line);t=str(x.get('title') or '');w=str(x.get('warranty') or '');a=' '.join(str(v.get('name',''))+' '+str(v.get('value_name','')) for v in x.get('attributes',[]) if isinstance(v,dict))
  docs.append((t+' '+t+' warranty '+w+' attributes '+a+' category_'+str(x.get('category_id',''))).lower());sellers.append(str(x['seller_id']));ys.append(x.get('condition')=='used')
 return np.array(docs,dtype=object),np.array(ys,dtype=np.int8),np.array(sellers)
def vectorize(doc,fit,other):
 vs=[TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=100000,sublinear_tf=True,dtype=np.float32,strip_accents='unicode'),TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=3,max_features=100000,sublinear_tf=True,dtype=np.float32,strip_accents='unicode')]
 xs=[];zs=[[] for _ in other]
 for v in vs:
  xs.append(v.fit_transform(doc[fit]));
  for j,ix in enumerate(other):zs[j].append(v.transform(doc[ix]))
 return hstack(xs,format='csr'),[hstack(z,format='csr') for z in zs],vs
def ratio(x,y):
 p=np.asarray(x[y==1].sum(axis=0)).ravel()+1;q=np.asarray(x[y==0].sum(axis=0)).ravel()+1
 return np.log((p/p.sum())/(q/q.sum())).astype(np.float32)
def model(x,y,c):return LogisticRegression(C=c,solver='liblinear',max_iter=250).fit(x,y)
def main():
 docs,y,seller=read(ROOT.parent/'recovered/data/train_data.jsonlines');test,_,_=read(ROOT.parent/'recovered/data/test_data.jsonlines');docs=np.r_[docs,test];part=np.load(ROOT/'results/new_models_20261021/partitions.npz');fit,tune,confirm=[part[k] for k in ['fit','tune','confirm']];dev=np.r_[fit,tune];ti=np.arange(70000,100000)
 report={'protocol':'Tune-only C selection; inherited row partitions, not a historically virgin holdout. No public feedback labels used.','predeclared_blend_native_weight':.85,'trials':[]}
 log('vectorizing fit');x,[xt],_=vectorize(docs,fit,[tune]);r=ratio(x,y[fit]);x=x.multiply(r).tocsr();xt=xt.multiply(r).tocsr();best=None
 for c in [.25,1,4]:
  m=model(x,y[fit],c);p=m.predict_proba(xt)[:,1];score=metric(y[tune],p);report['trials'].append({'C':c,**score});log(report['trials'][-1]);np.savez_compressed(OUT/f'tune_C{c}.npz',rows=tune,p=p)
  if best is None or (score['accuracy'],-score['log_loss'])>best[0]:best=((score['accuracy'],-score['log_loss']),c,p)
  (OUT/'progress.json').write_text(json.dumps(report,indent=2))
 c=best[1];pt=best[2];report['selected_C']=c;np.savez_compressed(OUT/'selected_tune.npz',rows=tune,p=pt)
 native_t=ROOT/'results/new_models_20261021/seller_native_tune.npz';w=.85
 if native_t.exists():
  a=np.load(native_t);assert np.array_equal(a['rows'],tune)
  weights=[0,.25,.5,.75,.85,.95,1];w=max(weights,key=lambda z:(accuracy_score(y[tune],z*a['p']+(1-z)*pt>=.5),-log_loss(y[tune],z*a['p']+(1-z)*pt)))
  report['blend_selection']='Native weight selected on tune';report['blend_tune']=metric(y[tune],w*a['p']+(1-w)*pt)
 else:report['blend_selection']='Native tune vector unavailable; fixed 0.85 native weight preregistered before confirmation'
 report['native_weight']=w
 del x,xt,m;gc.collect();log('vectorizing dev');x,[xc],_=vectorize(docs,dev,[confirm]);r=ratio(x,y[dev]);m=model(x.multiply(r).tocsr(),y[dev],c);pc=m.predict_proba(xc.multiply(r).tocsr())[:,1];seen=np.isin(seller[confirm],seller[dev]);report['confirmation']=metric(y[confirm],pc,seen)
 base=np.load(ROOT/'results/new_models_20261021/seller_native_confirm.npz');assert np.array_equal(base['rows'],confirm);pb=w*base['p']+(1-w)*pc
 report['native_confirmation']=metric(y[confirm],base['p'],seen);report['blend_confirmation']=metric(y[confirm],pb,seen);report['seconds']=time.time()-START
 np.savez_compressed(OUT/'confirmation.npz',rows=confirm,p=pc,blend=pb,y=y[confirm],seen=seen);(OUT/'report.json').write_text(json.dumps(report,indent=2));log(report)
 if report['blend_confirmation']['accuracy']>report['native_confirmation']['accuracy']:
  del x,xc,m;gc.collect();log('Refit full: fixed/selected blend improved confirmation');x,[xtest],vs=vectorize(docs,np.arange(70000),[ti]);r=ratio(x,y);m=model(x.multiply(r).tocsr(),y,c);p=m.predict_proba(xtest.multiply(r).tocsr())[:,1];np.savez_compressed(OUT/'test.npz',p=p)
  import joblib
  joblib.dump({'vectorizers':vs,'ratio':r,'model':m,'native_weight':w},OUT/'nb_model.joblib',compress=3)
  report['full_refit']='Complete';report['seconds']=time.time()-START;(OUT/'report.json').write_text(json.dumps(report,indent=2))
 else:report['full_refit']='Skipped: no confirmation accuracy improvement';(OUT/'report.json').write_text(json.dumps(report,indent=2))
 log('DONE')
if __name__=='__main__':main()
