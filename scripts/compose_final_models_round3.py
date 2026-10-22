"""Two predeclared final stacks, tune-only C selection and fixed .5 threshold.
No confirm vectors are opened before BOTH selection locks exist. Five inputs are
mandatory: deep, seller dropout, field teachers, selected TPE and standalone NB.
No public labels, no replacement for missing inputs, no confirmation family choice.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import time,json,hashlib,sys
import numpy as np,pandas as pd,joblib
from scipy.special import logit
from scipy.stats import binomtest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.metrics import accuracy_score,log_loss,brier_score_loss
from threadpoolctl import threadpool_limits
threadpool_limits(1)
R=Path(__file__).resolve().parents[1];O=R/'results/training_round3/final';O.mkdir(parents=True,exist_ok=True)
P=R/'results/new_models_20261021/partitions.npz';part=np.load(P);fit,tune,confirm=[part[k] for k in ['fit','tune','confirm']];dev=np.r_[fit,tune]
H=R.parent/'recovered/mercadolibre_checkpoint_20261017/mercadolibre_recovery_20261017/prior_results/warranty/results';HC='refit49k_without_public'
D=R/'results/training_round2/deep_teacher';B=R/'results/training_round3/dropout';F=R/'results/training_round3/fields';T=R/'results/training_round3/tpe';N=R/'results/training_next/nbsvm'
SPECS={'deep':{'tune':D/'tune.npz','confirm':D/'confirm.npz','test':D/'test.npz'},'dropout':{'tune':B/'tune_0.2.npz','confirm':B/'confirm.npz','test':B/'test.npz'},'fields':{'tune':F/'tune.npz','confirm':F/'confirm.npz','test':F/'test.npz'},'tpe':{'tune':T/'tune.npz','confirm':T/'confirm.npz','test':T/'test.npz'},'nb':{'tune':N/'selected_tune.npz','confirm':N/'confirmation.npz','test':N/'test.npz'}}
NAMES=list(SPECS);DEADLINE=time.time()+int(os.environ.get('FINAL_WAIT_SECONDS','5400'))
def status(stage,**kw):
 obj={'stage':stage,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw};(O/'status.json').write_text(json.dumps(obj,indent=2));print(obj,flush=True)
def hashfile(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wait(paths,stage):
 paths=list(paths)
 while True:
  missing=[str(p.relative_to(R)) if p.is_relative_to(R) else str(p) for p in paths if not p.exists()]
  if not missing:return
  status(stage,missing=missing)
  if time.time()>=DEADLINE:raise TimeoutError('Missing mandatory inputs; no substitute permitted: '+str(missing))
  time.sleep(15)
def probs(path,rows=None):
 # Producer writes small NPZ at stage end; allow an in-flight final rename/write.
 for attempt in range(12):
  try:
   with np.load(path) as z:
    p=z['p'].copy()
    if rows is not None:assert 'rows' in z.files and np.array_equal(z['rows'],rows),(str(path),'row mismatch')
   assert p.ndim==1 and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),str(path)
   return p
  except (ValueError,EOFError,OSError):
   if attempt==11:raise
   time.sleep(2)
def matrix(stage,rows=None):return np.column_stack([probs(SPECS[n][stage],rows) for n in NAMES])
def logits(p):return logit(np.clip(p,1e-5,1-1e-5))
def read_minimal():
 y=[];s=[]
 for filename in ['train_data.jsonlines','test_data.jsonlines']:
  for line in open(R.parent/'recovered/data'/filename):
   z=json.loads(line);y.append(z.get('condition')=='used');s.append(str(z['seller_id']))
 return np.array(y,dtype=np.int8),np.array(s)
def metric(y,p,seen,threshold=.5):
 out={'n':len(y),'threshold':threshold,'accuracy':float(accuracy_score(y,p>=threshold)),'logloss':float(log_loss(y,p)),'brier':float(brier_score_loss(y,p))}
 for name,m in [('seen',seen),('new',~seen)]:out[name]={'n':int(m.sum()),'accuracy':float(accuracy_score(y[m],p[m]>=threshold))} if m.any() else {'n':0,'accuracy':None}
 return out
def paired(y,p,b,threshold=.5):
 old=(b>=threshold)==y;new=(p>=.5)==y;g=int((new&~old).sum());l=int((old&~new).sum());return {'fixes':g,'breaks':l,'net':g-l,'p_unadjusted':float(binomtest(g,g+l,.5).pvalue) if g+l else 1}
def select(name,x,y,names,sources):
 d=O/name;d.mkdir(exist_ok=True);cv=StratifiedKFold(5,shuffle=True,random_state=20261023);trials=[]
 for c in [.01,.1,1]:
  m=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1500,solver='lbfgs'))
  v=cross_val_score(m,x,y,cv=cv,scoring='accuracy',n_jobs=1);trials.append({'C':c,'accuracy_mean':float(v.mean()),'fold_accuracies':v.tolist()})
 c=max(trials,key=lambda q:(q['accuracy_mean'],-q['C']))['C'];m=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1500,solver='lbfgs')).fit(x,y)
 lock={'C':c,'n_tune':len(y),'threshold':.5,'trials':trials,'input_names':names,'source_hashes':sources,'selection':'5-fold tune CV accuracy; ties prefer smaller C. No confirm vectors opened before both locks. Families predeclared and evaluated separately, not selected on confirm.','limits':'Base hyperparameters previously selected on tune, holdouts historically reused; CV is conditional selection machinery, not unbiased end-to-end validation.'};(d/'selection_lock.json').write_text(json.dumps(lock,indent=2));tmp=d/'meta_model.tmp.joblib';joblib.dump(m,tmp);check=joblib.load(tmp);assert np.allclose(check.predict_proba(x[:10]),m.predict_proba(x[:10]));os.replace(tmp,d/'meta_model.joblib');return m,lock

def main():
 wait([s['tune'] for s in SPECS.values()]+[T/'selection_lock.json',B/'selection_lock.json',F/'selection_lock.json'],'waiting_tune')
 dropout_lock=json.loads((B/'selection_lock.json').read_text());choice=dropout_lock.get('choice',dropout_lock);assert float(choice['dropout_fraction'])==.2,'Unexpected dropout winner: update path explicitly, do not silently substitute'
 y,sellers=read_minimal();pt=matrix('tune',tune);assert pt.shape==(7000,5)
 hist=pd.read_csv(H/'composition/holdout_predictions.csv').set_index('row');common_t=np.isin(tune,hist.index);ct=tune[common_t];assert np.array_equal(y[ct],hist.loc[ct,'y_true'].to_numpy())
 hashes={n:hashfile(SPECS[n]['tune']) for n in NAMES};hashes['historical_holdout']=hashfile(H/'composition/holdout_predictions.csv')
 # Construct/select both models before reading even one confirmation probability.
 model1,lock1=select('new_only',logits(pt),y[tune],NAMES,hashes)
 ht=hist.loc[ct,HC].to_numpy();model2,lock2=select('with_history',logits(np.column_stack([ht,pt[common_t]])),y[ct],['historical_without_public']+NAMES,hashes)
 status('both_selection_locks_written',new_only_C=lock1['C'],history_C=lock2['C'])
 wait([s['confirm'] for s in SPECS.values()],'waiting_confirmation')
 pc=matrix('confirm',confirm);seen=np.isin(sellers[confirm],sellers[dev]);pred1=model1.predict_proba(logits(pc))[:,1];deep_th=float(json.loads((D/'selection_lock.json').read_text())['threshold'])
 common_c=np.isin(confirm,hist.index);cc=confirm[common_c];assert np.array_equal(y[cc],hist.loc[cc,'y_true'].to_numpy());hc=hist.loc[cc,HC].to_numpy();pred2=model2.predict_proba(logits(np.column_stack([hc,pc[common_c]])))[:,1]
 report1={**lock1,'confirmation':metric(y[confirm],pred1,seen),'deep_reference_locked':metric(y[confirm],pc[:,0],seen,deep_th),'paired_vs_deep_locked':paired(y[confirm],pred1,pc[:,0],deep_th),'bases_at_05':{n:metric(y[confirm],pc[:,j],seen) for j,n in enumerate(NAMES)}}
 report2={**lock2,'confirmation':metric(y[cc],pred2,seen[common_c]),'historical_reference_locked':metric(y[cc],hc,seen[common_c],.505),'paired_vs_historical_locked':paired(y[cc],pred2,hc,.505),'deep_reference_locked':metric(y[cc],pc[common_c,0],seen[common_c],deep_th),'paired_vs_deep_locked':paired(y[cc],pred2,pc[common_c,0],deep_th)}
 for name,rows,pred,report in [('new_only',confirm,pred1,report1),('with_history',cc,pred2,report2)]:
  d=O/name;np.savez_compressed(d/'confirm.npz',rows=rows,p=pred,y=y[rows],seen=np.isin(sellers[rows],sellers[dev]));(d/'report.json').write_text(json.dumps(report,indent=2))
 status('confirmation_complete',new_only=report1['confirmation'],with_history=report2['confirmation'])
 wait([s['test'] for s in SPECS.values()],'waiting_test_no_substitutes')
 pp=matrix('test');assert pp.shape==(30000,5);production=pd.read_csv(H/'production_probabilities.csv');assert np.array_equal(production.ID.to_numpy(),np.arange(1,30001));hp=production.p_used_full70k_selected.to_numpy();test1=model1.predict_proba(logits(pp))[:,1];test2=model2.predict_proba(logits(np.column_stack([hp,pp])))[:,1]
 baseline=pd.read_csv(R/'submissions/20261021/submission_26_A_E2_complemento16.csv');assert np.array_equal(baseline.ID.to_numpy(),np.arange(1,30001))
 for name,p,report in [('new_only',test1,report1),('with_history',test2,report2)]:
  d=O/name;assert len(p)==30000 and np.isfinite(p).all();np.savez_compressed(d/'test.npz',p=p,ID=np.arange(1,30001));csv=d/('candidate_'+name+'_model_only.csv');f=pd.DataFrame({'ID':np.arange(1,30001),'condition':np.where(p>=.5,'used','new')});f.to_csv(csv,index=False,lineterminator='\r\n');report['test_export']={'candidate':csv.name,'n':len(f),'sha256':hashfile(csv),'differences_vs26':int((f.condition!=baseline.condition).sum()),'public_score_observed':None,'scope':'Candidate model only, no flips overlaid. Historical input (if present) is production selected with base70k and frozen auxiliary models; all new bases full70k.'};(d/'report.json').write_text(json.dumps(report,indent=2));(d/'standardized_coefficients.json').write_text(json.dumps(dict(zip(report['input_names'],(model1 if name=='new_only' else model2)[-1].coef_[0].tolist())),indent=2))
 (O/'README.md').write_text('''# Final ensembles, round3\n\nTwo families preregistered: five new bases only on7k tune, and historical+samefive on mutually excluded common tune rows. C selected5CV on tune, threshold.5 fixed. Both selection locks precede loading confirm vectors. No candidate was selected by confirmation.\n\nAll base hyperparameter searches used the same tune set, so inner CV here is conditional; all historical holdouts were reused. Confirmation comparisons are exploratory, not independent proof of public improvement. Historical production is hybrid: base70k with older auxiliary fits frozen. Seen/new is relative to newdev56k.\n\nBoth CSVs are offline model-only candidates. No flips, recovered labels, public-score calibration or individual public memberships enter training. No missing base may be replaced silently. Public scores are unobserved.\n''')
 status('done',new_only=report1['test_export'],with_history=report2['test_export'])
if __name__=='__main__':main()
