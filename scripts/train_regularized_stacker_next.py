"""Tune-only regularized stacking; confirmation is never used for selection.
NB/retrieval/rich hyperparameters were themselves tuned on tune; inner CV here is
selection machinery, NOT an unbiased score. No public labels or flips are used.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.metrics import accuracy_score,log_loss
from scipy.special import logit
from scipy.stats import binomtest
import joblib
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'results/training_next';OUT=D/'stacker';OUT.mkdir(parents=True,exist_ok=True)
def status(stage,**kw):
 o={'stage':stage,'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw};(OUT/'status.json').write_text(json.dumps(o,indent=2));print(o,flush=True)
def raw_small():
 ys=[];seller=[];listing=[];qty=[]
 for file in ['train_data.jsonlines','test_data.jsonlines']:
  for line in open(ROOT.parent/'recovered/data'/file):
   x=json.loads(line);ys.append(x.get('condition')=='used');seller.append(str(x['seller_id']));listing.append(str(x.get('listing_type_id','missing')));qty.append(float(x.get('initial_quantity') or 0))
 return np.array(ys,dtype=np.int8),np.array(seller),np.array(listing),np.array(qty)
def load_p(path,rows,key='p'):
 z=np.load(path);assert np.array_equal(z['rows'],rows),(path,'row mismatch');return z[key]
def features(probs,ix,seller,listing,qty,fit_sellers):
 pp=np.column_stack([probs[n] for n in ['native','nb','retrieval','rich','context']]);ll=logit(np.clip(pp,1e-5,1-1e-5));seen=np.isin(seller[ix],fit_sellers).astype(float);q1=(qty[ix]<=1).astype(float);free=(listing[ix]=='free').astype(float)
 # Small fixed feature vocabulary, no high-cardinality label statistics.
 cols=[ll,seen[:,None],q1[:,None],free[:,None],np.minimum(np.log1p(qty[ix]),8)[:,None]];names=['logit_native','logit_nb','logit_retrieval','logit_rich','logit_context','seen_seller','quantity_le1','listing_free','log_quantity']
 for j,n in enumerate(['nb','retrieval','rich','context'],start=1):
  delta=ll[:,j]-ll[:,0]
  for label,gate in [('seen',seen),('qty1',q1),('free',free)]:cols.append((delta*gate)[:,None]);names.append(n+'_minus_native_x_'+label)
 return np.hstack(cols),names,seen.astype(bool)
def met(y,p,seen):
 return {'accuracy':float(accuracy_score(y,p>=.5)),'log_loss':float(log_loss(y,p)),'seen_accuracy':float(accuracy_score(y[seen],p[seen]>=.5)),'unseen_accuracy':float(accuracy_score(y[~seen],p[~seen]>=.5))}
def main():
 baseline=D/'baseline_reproduction/seller_native_tune.npz';deadline=time.time()+600
 while not baseline.exists():
  status('waiting_for_native_tune',remaining_seconds=int(deadline-time.time()))
  if time.time()>=deadline:return
  time.sleep(15)
 status('loading_small_inputs');y,seller,listing,qty=raw_small();part=np.load(ROOT/'results/new_models_20261021/partitions.npz');fit,tune,confirm=[part[k] for k in ['fit','tune','confirm']];dev=np.r_[fit,tune]
 pt={'native':load_p(baseline,tune),'nb':load_p(D/'nbsvm/selected_tune.npz',tune),'rich':load_p(D/'rich/config1_tune.npz',tune),'retrieval':load_p(D/'retrieval/tune_candidates.npz',tune,'global_k128_pow2'),'context':load_p(D/'context/tune_1.npz',tune)}
 xt,names,seent=features(pt,tune,seller,listing,qty,seller[fit]);cv=StratifiedKFold(5,shuffle=True,random_state=20261022);trials=[]
 for c in [.01,.1,1]:
  m=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1000,solver='lbfgs'))
  scores=cross_val_score(m,xt,y[tune],cv=cv,scoring='accuracy',n_jobs=1);trials.append({'C':c,'cv_accuracy_mean':float(scores.mean()),'cv_scores':scores.tolist()});status('inner_cv',trials=trials)
 c=max(trials,key=lambda x:(x['cv_accuracy_mean'],-x['C']))['C'];model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1000,solver='lbfgs')).fit(xt,y[tune]);joblib.dump(model,OUT/'meta_model.joblib');(OUT/'selection_lock.json').write_text(json.dumps({'C':c,'trials':trials,'feature_names':names,'scope':'Tune-only selection. Inner CV conditional on previously selected base hyperparameters; not unbiased validation.'},indent=2));status('selection_locked',C=c)
 pc={'native':load_p(ROOT/'results/new_models_20261021/seller_native_confirm.npz',confirm),'nb':load_p(D/'nbsvm/confirmation.npz',confirm),'rich':load_p(D/'rich/confirm.npz',confirm),'retrieval':load_p(D/'retrieval/confirm_candidates.npz',confirm,'global_k128_pow2'),'context':load_p(D/'context/confirm.npz',confirm)}
 xc,_,seen=features(pc,confirm,seller,listing,qty,seller[dev]);p=model.predict_proba(xc)[:,1];np.savez_compressed(OUT/'confirmation.npz',rows=confirm,p=p,y=y[confirm],seen=seen)
 report={'selected_C':c,'trials':trials,'feature_names':names,'confirmation':met(y[confirm],p,seen),'bases_confirmation':{k:met(y[confirm],v,seen) for k,v in pc.items()},'scope':'Meta trained ONLY on7k tune predictions from49k base fits. Bases refitted56k for14k confirmation; base-refit distribution shift applies. Existing splits repeatedly reused and base parameters previously tuned on tune. No public labels or flip labels; no weights chosen on confirmation.'}
 old=(pc['native']>=.5)==y[confirm];new=(p>=.5)==y[confirm];g=int((new&~old).sum());l=int((old&~new).sum());report['paired_vs_native']={'gain':g,'loss':l,'net':g-l,'p_unadjusted':float(binomtest(g,g+l,.5).pvalue) if g+l else 1}
 report['coefficients']=dict(zip(names,model[-1].coef_[0].tolist()));(OUT/'report.json').write_text(json.dumps(report,indent=2));status('confirmation_complete',metrics=report['confirmation'])
 # Export only when all five full-train base vectors exist; no fallback substitutes.
 ret=D/'retrieval/test_candidates.npz'
 if ret.exists():
  rt=np.load(ret);rp=rt['global_k128_pow2']
 elif (D/'retrieval/test.npz').exists():
  rt=np.load(D/'retrieval/test.npz');rp=rt['global_k128_pow2'] if 'global_k128_pow2' in rt.files else rt['p']
 else:rp=None
 if rp is not None:
  bn=np.load(ROOT/'results/new_models_20261021/seller_native_test.npz');ps={'native':bn['test'],'nb':np.load(D/'nbsvm/test.npz')['p'],'rich':np.load(D/'rich/test.npz')['p'],'retrieval':rp,'context':np.load(D/'context/test.npz')['p']};ix=np.arange(70000,100000);xx,_,_=features(ps,ix,seller,listing,qty,seller[:70000]);p=model.predict_proba(xx)[:,1];np.savez_compressed(OUT/'test.npz',p=p);report['test_export']='Complete, full70k five base models with tune7k meta weights unchanged'
 else:report['test_export']='Pending full70k retrieval vector; rerun script or export using saved meta_model'
 (OUT/'report.json').write_text(json.dumps(report,indent=2));status('done',test_export=report['test_export'])
if __name__=='__main__':main()
