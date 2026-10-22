"""Historical+new base stacking on mutually excluded tune rows; no public labels."""
from pathlib import Path
import sys,json,joblib
import numpy as np,pandas as pd
from scipy.special import logit
from scipy.stats import binomtest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.metrics import accuracy_score,log_loss
R=Path(__file__).resolve().parents[1];D=R/'results/training_next';OUT=R/'results/training_round2/historical_gated';OUT.mkdir(parents=True,exist_ok=True)
H=R.parent/'recovered/mercadolibre_checkpoint_20261017/mercadolibre_recovery_20261017/prior_results/warranty/results/composition'
part=np.load(R/'results/new_models_20261021/partitions.npz');tune,confirm=part['tune'],part['confirm'];h=pd.read_csv(H/'holdout_predictions.csv').set_index('row');ht=pd.read_csv(H.parent/'production_probabilities.csv');col='refit49k_without_public'
def get(path,rows,key='p'):
 z=np.load(path);assert np.array_equal(z['rows'],rows);return z[key]
def pred(rows,stage):
 if stage=='tune':
  a=[get(D/'baseline_reproduction/seller_native_tune.npz',rows),get(D/'nbsvm/selected_tune.npz',rows),get(D/'rich/config1_tune.npz',rows),get(D/'retrieval/tune_candidates.npz',rows,'global_k128_pow2'),get(D/'context/tune_1.npz',rows)]
 else:
  a=[get(R/'results/new_models_20261021/seller_native_confirm.npz',rows),get(D/'nbsvm/confirmation.npz',rows),get(D/'rich/confirm.npz',rows),get(D/'retrieval/confirm_candidates.npz',rows,'global_k128_pow2'),get(D/'context/confirm.npz',rows)]
 return np.column_stack(a)
def met(y,p):return {'accuracy':float(accuracy_score(y,p>=.5)),'log_loss':float(log_loss(y,p))}

# Deterministic gate vocabulary, no labels used in gate construction.
sellers=[];listing=[];qty=[]
for file in ['train_data.jsonlines','test_data.jsonlines']:
 for line in open(R.parent/'recovered/data'/file):
  v=json.loads(line);sellers.append(str(v['seller_id']));listing.append(str(v.get('listing_type_id','')));qty.append(float(v.get('initial_quantity') or 0))
sellers=np.array(sellers);listing=np.array(listing);qty=np.array(qty)
def gated(x,rows,trainrows):
 seen=np.isin(sellers[rows],sellers[trainrows]).astype(float);q=(qty[rows]<=1).astype(float);f=(listing[rows]=='free').astype(float)
 a=[x,seen[:,None],q[:,None],f[:,None]]
 for j in range(1,6):
  diff=x[:,j]-x[:,0]
  for gate in [seen,q,f]:a.append((diff*gate)[:,None])
 return np.hstack(a)
m=np.isin(tune,h.index);rows=tune[m];y=h.loc[rows,'y_true'].to_numpy();pt=pred(tune,'tune')[m];xt=logit(np.clip(np.column_stack([h.loc[rows,col],pt]),1e-5,1-1e-5));xt=gated(xt,rows,part['fit']);trials=[]
for c in [.01,.1,1]:
 model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1000));s=cross_val_score(model,xt,y,cv=StratifiedKFold(5,shuffle=True,random_state=20261022),scoring='accuracy');trials.append({'C':c,'cv_accuracy':float(s.mean())})
c=max(trials,key=lambda q:(q['cv_accuracy'],-q['C']))['C'];model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1000)).fit(xt,y);joblib.dump(model,OUT/'model.joblib');lock={'C':c,'n_tune_common':len(rows),'trials':trials,'historical_column':col,'scope':'No public coefficients. Inner CV on common tune rows; historically reused holdouts. Confirmation not consulted for selection. Base fitting excludes corresponding predictions. Historical coefficient selection itself used historical development.'};(OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2))
m=np.isin(confirm,h.index);rows=confirm[m];y=h.loc[rows,'y_true'].to_numpy();base=h.loc[rows,col].to_numpy();xc=logit(np.clip(np.column_stack([base,pred(confirm,'confirm')[m]]),1e-5,1-1e-5));xc=gated(xc,rows,np.r_[part['fit'],part['tune']]);p=model.predict_proba(xc)[:,1];old=(base>=.505)==y;new=(p>=.5)==y;g=int((new&~old).sum());l=int((old&~new).sum());report={**lock,'n_confirm_common':len(rows),'historical_at_05':met(y,base),'historical_accuracy_at_locked_threshold_0505':float(accuracy_score(y,base>=.505)),'stacker':met(y,p),'gain':g,'loss':l,'net':g-l,'paired_p_unadjusted':float(binomtest(g,g+l,.5).pvalue) if g+l else 1};np.savez_compressed(OUT/'confirmation.npz',rows=rows,y=y,p=p,baseline=base)
a=[np.load(R/'results/new_models_20261021/seller_native_test.npz')['test'],np.load(D/'nbsvm/test.npz')['p'],np.load(D/'rich/test.npz')['p'],np.load(D/'retrieval/selected_retrieval_test.npz')['p'],np.load(D/'context/test.npz')['p']];assert len(ht)==30000;xx=logit(np.clip(np.column_stack([ht['p_used_full70k_selected'].to_numpy(),*a]),1e-5,1-1e-5));xx=gated(xx,np.arange(70000,100000),np.arange(70000));p=model.predict_proba(xx)[:,1];np.savez_compressed(OUT/'test.npz',p=p);report['test_export']='complete: historical production full70k selected no-public coefficients; 5 new bases full70k';(OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
