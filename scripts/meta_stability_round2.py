"""Repeated nested meta CV on common tune only; inherited base selection is not nested."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,joblib
import numpy as np,pandas as pd
from scipy.special import logit,expit
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,log_loss
R=Path(__file__).resolve().parents[1];D=R/'results/training_next';O=R/'results/training_round2/meta';O.mkdir(parents=True,exist_ok=True)
H=R.parent/'recovered/mercadolibre_checkpoint_20261017/mercadolibre_recovery_20261017/prior_results/warranty/results/composition';COL='refit49k_matched_without_public'
def get(path,rows,key='p'):
 z=np.load(path);assert np.array_equal(z['rows'],rows);return z[key]
def probs(rows,stage):
 if stage=='tune':fs=[(D/'baseline_reproduction/seller_native_tune.npz','p'),(D/'nbsvm/selected_tune.npz','p'),(D/'rich/config1_tune.npz','p'),(D/'retrieval/tune_candidates.npz','global_k128_pow2'),(D/'context/tune_1.npz','p')]
 else:fs=[(R/'results/new_models_20261021/seller_native_confirm.npz','p'),(D/'nbsvm/confirmation.npz','p'),(D/'rich/confirm.npz','p'),(D/'retrieval/confirm_candidates.npz','global_k128_pow2'),(D/'context/confirm.npz','p')]
 return np.column_stack([get(p,rows,k) for p,k in fs])
def met(y,p,threshold=.5):return {'accuracy':float(accuracy_score(y,p>=threshold)),'log_loss':float(log_loss(y,np.clip(p,1e-6,1-1e-6)))}
def family(c):return c['family']
configs=[{'family':'history_05'},{'family':'history_052'}]+[{'family':f,'C':c} for f in ['sigmoid','logistic6'] for c in [.01,.1,1,10]]
for j in range(1,6):
 for w in [.05,.1,.2,.35,.5]:configs.append({'family':'sparse_mix','partners':[j],'weights':[w]})
for j in range(1,6):
 for k in range(j+1,6):
  for w in [.1,.2]:configs.append({'family':'sparse_mix','partners':[j,k],'weights':[w,w]})
def fit(c,X,y):
 if c['family'] in ['sigmoid','logistic6']:
  xx=logit(np.clip(X[:,:1] if c['family']=='sigmoid' else X,1e-5,1-1e-5));return make_pipeline(StandardScaler(),LogisticRegression(C=c['C'],max_iter=1000)).fit(xx,y)
 return None
def predict(c,m,X):
 f=c['family']
 if f=='history_05':return X[:,0]
 if f=='history_052':return expit(logit(np.clip(X[:,0],1e-6,1-1e-6))-logit(.52))
 if f=='sparse_mix':return (1-sum(c['weights']))*X[:,0]+sum(w*X[:,j] for w,j in zip(c['weights'],c['partners']))
 xx=logit(np.clip(X[:,:1] if f=='sigmoid' else X,1e-5,1-1e-5));return m.predict_proba(xx)[:,1]
def inner_select(X,y,seed):
 folds=list(StratifiedKFold(3,shuffle=True,random_state=seed).split(X,y));results=[]
 for i,c in enumerate(configs):
  p=np.zeros(len(y))
  for tr,va in folds:p[va]=predict(c,fit(c,X[tr],y[tr]),X[va])
  results.append({'config_index':i,**met(y,p)})
 def best(ix):return max(ix,key=lambda i:(results[i]['accuracy'],-results[i]['log_loss'],-i))
 return best(range(len(configs))),{f:best([i for i,c in enumerate(configs) if family(c)==f]) for f in sorted(set(map(family,configs)))},results
h=pd.read_csv(H/'holdout_predictions.csv').set_index('row');part=np.load(R/'results/new_models_20261021/partitions.npz');t=part['tune'];mask=np.isin(t,h.index);rows=t[mask];X=np.column_stack([h.loc[rows,COL],probs(t,'tune')[mask]]);y=h.loc[rows,'y_true'].to_numpy();assert len(y)==2098
families=sorted(set(map(family,configs)));records=[];preds={f:np.zeros((3,len(y))) for f in families+['selected_procedure']};locks=[]
for rep in range(3):
 for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20261022+rep).split(X,y)):
  best,byfam,res=inner_select(X[tr],y[tr],20261040+rep*5+fold);locks.append({'repeat':rep,'fold':fold,'selected_config':configs[best]})
  for f,idx in {**byfam,'selected_procedure':best}.items():
   c=configs[idx];p=predict(c,fit(c,X[tr],y[tr]),X[va]);preds[f][rep,va]=p;records.append({'repeat':rep,'fold':fold,'family':f,'config':json.dumps(c),**met(y[va],p)})
  print('nested',rep,fold,configs[best],flush=True)
  (O/'progress.json').write_text(json.dumps({'complete_outer_folds':len(locks),'total_outer_folds':15},indent=2))
summary={f:{'repeat_accuracy':[met(y,p)['accuracy'] for p in pp],'mean_accuracy':float(np.mean([met(y,p)['accuracy'] for p in pp])),'repeat_log_loss':[met(y,p)['log_loss'] for p in pp]} for f,pp in preds.items()};pd.DataFrame(records).to_csv(O/'outer_results.csv',index=False);np.savez_compressed(O/'nested_oof.npz',rows=rows,y=y,**preds)
# Final choice follows tune-only inner selection on all common tune; outer evaluation describes selection procedure.
best,byfam,res=inner_select(X,y,20261100);c=configs[best];model=fit(c,X,y);lock={'selected_config':c,'n_common_tune':len(y),'all_inner_results':[{**configs[v['config_index']],**v} for v in res],'nested_summary':summary,'outer_locks':locks,'scope':'All family/parameter choices use 2098 common tune only. Existing base models and hyperparameters had already been selected using tune; nested meta CV is conditional, not fresh end-to-end validation. 4147 confirm historically reused.'};(O/'selection_lock.json').write_text(json.dumps(lock,indent=2));joblib.dump({'config':c,'model':model},O/'selected_meta.joblib')
# Confirmation loaded only after selection lock is persisted.
t=part['confirm'];mask=np.isin(t,h.index);cr=t[mask];yc=h.loc[cr,'y_true'].to_numpy();xc=np.column_stack([h.loc[cr,COL],probs(t,'confirm')[mask]]);pc=predict(c,model,xc);report={**lock,'n_common_confirm':len(yc),'confirmation_selected':met(yc,pc),'confirmation_history05':met(yc,xc[:,0]),'confirmation_history052':met(yc,xc[:,0],.52)}
for name,th in [('history05',.5),('history052',.52)]:
 a=(xc[:,0]>=th)==yc;b=(pc>=.5)==yc;g=int((b&~a).sum());l=int((a&~b).sum());report['paired_vs_'+name]={'gain':g,'loss':l,'net':g-l,'p_unadjusted':float(binomtest(g,g+l,.5).pvalue) if g+l else 1}
np.savez_compressed(O/'confirmation.npz',rows=cr,y=yc,p=pc)
ht=pd.read_csv(H/'test_probabilities.csv');a=[np.load(R/'results/new_models_20261021/seller_native_test.npz')['test'],np.load(D/'nbsvm/test.npz')['p'],np.load(D/'rich/test.npz')['p'],np.load(D/'retrieval/selected_retrieval_test.npz')['p'],np.load(D/'context/test.npz')['p']];assert np.array_equal(ht.row,np.arange(30000));xx=np.column_stack([ht[COL].to_numpy(),*a]);pp=predict(c,model,xx);np.savez_compressed(O/'test.npz',p=pp);report['test_export']='complete';(O/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['all_inner_results','outer_locks']},indent=2))
