"""Offset logistic correction: anchor historical logits, penalize only corrections.
Select penalty on common tune CV, never confirmation; inherited splits reused.
"""
from pathlib import Path
import json,numpy as np,pandas as pd
from scipy.special import expit,logit
from scipy.optimize import minimize
from scipy.stats import binomtest
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,log_loss
R=Path(__file__).resolve().parents[1];D=R/'results/training_next';O=R/'results/training_round2/offset_meta';O.mkdir(parents=True,exist_ok=True)
H=R.parent/'recovered/mercadolibre_checkpoint_20261017/mercadolibre_recovery_20261017/prior_results/warranty/results';h=pd.read_csv(H/'composition/holdout_predictions.csv').set_index('row');part=np.load(R/'results/new_models_20261021/partitions.npz');col='refit49k_without_public'
def load(path,rows,key='p'):
 z=np.load(path);assert np.array_equal(z['rows'],rows);return z[key]
def inputs(rows,stage):
 if stage=='tune':p=[load(D/'baseline_reproduction/seller_native_tune.npz',rows),load(D/'nbsvm/selected_tune.npz',rows),load(D/'rich/config1_tune.npz',rows),load(D/'retrieval/tune_candidates.npz',rows,'global_k128_pow2'),load(D/'context/tune_1.npz',rows)]
 else:p=[load(R/'results/new_models_20261021/seller_native_confirm.npz',rows),load(D/'nbsvm/confirmation.npz',rows),load(D/'rich/confirm.npz',rows),load(D/'retrieval/confirm_candidates.npz',rows,'global_k128_pow2'),load(D/'context/confirm.npz',rows)]
 m=np.isin(rows,h.index);ix=rows[m];b=h.loc[ix,col].to_numpy();return ix,h.loc[ix,'y_true'].to_numpy(),b,np.column_stack(p)[m]
def design(b,p):
 off=logit(np.clip(b,1e-5,1-1e-5));return off,logit(np.clip(p,1e-5,1-1e-5))-off[:,None]
def fit(off,x,y,alpha):
 mean=x.mean(0);std=x.std(0);std[std<1e-8]=1;z=np.column_stack([np.ones(len(y)),(x-mean)/std]);n=len(y)
 def fun(w):
  a=off+z@w;pen=np.ones(len(w));pen[0]=.2
  loss=np.logaddexp(0,a).sum()-y@a+alpha/2*np.sum(pen*w*w);g=z.T@(expit(a)-y)+alpha*pen*w
  return loss/n,g/n
 sol=minimize(fun,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':500});assert sol.success,sol.message
 return sol.x,mean,std
def predict(model,off,x):
 w,mean,std=model;return expit(off+w[0]+((x-mean)/std)@w[1:])
ix,y,b,p=inputs(part['tune'],'tune');off,x=design(b,p);trials=[]
for alpha in [1,10,100,1000]:
 pred=np.zeros(len(y))
 for tr,va in StratifiedKFold(5,shuffle=True,random_state=20261022).split(x,y):pred[va]=predict(fit(off[tr],x[tr],y[tr],alpha),off[va],x[va])
 trials.append({'alpha':alpha,'cv_accuracy':float(accuracy_score(y,pred>=.5)),'cv_logloss':float(log_loss(y,pred))})
a=max(trials,key=lambda q:(q['cv_accuracy'],-q['cv_logloss']))['alpha'];model=fit(off,x,y,a);lock={'alpha':a,'n_common_tune':len(y),'trials':trials,'scope':'Offset historical logit coefficient fixed1, Gaussian penalty on correction and intercept. Common tune CV selection only. Reused holdouts, exploratory comparison.'};(O/'selection_lock.json').write_text(json.dumps(lock,indent=2));np.savez_compressed(O/'model.npz',weights=model[0],mean=model[1],std=model[2])
ix,y,b,p=inputs(part['confirm'],'confirm');off,x=design(b,p);p=predict(model,off,x);old=(b>=.505)==y;new=(p>=.5)==y;g=int((new&~old).sum());l=int((old&~new).sum());report={**lock,'n_common_confirm':len(y),'baseline_accuracy_locked':float(accuracy_score(y,b>=.505)),'accuracy':float(accuracy_score(y,p>=.5)),'logloss':float(log_loss(y,p)),'gains':g,'losses':l,'net':g-l,'paired_p_unadjusted':float(binomtest(g,g+l,.5).pvalue) if g+l else 1};np.savez_compressed(O/'confirm.npz',rows=ix,y=y,p=p,baseline=b)
b=pd.read_csv(H/'production_probabilities.csv').p_used_full70k_selected.to_numpy();p=np.column_stack([np.load(R/'results/new_models_20261021/seller_native_test.npz')['test'],np.load(D/'nbsvm/test.npz')['p'],np.load(D/'rich/test.npz')['p'],np.load(D/'retrieval/selected_retrieval_test.npz')['p'],np.load(D/'context/test.npz')['p']]);off,x=design(b,p);p=predict(model,off,x);np.savez_compressed(O/'test.npz',p=p);report['test_export']='Historical production selected (hybrid base70k/frozen auxiliaries), new base full70k';(O/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
