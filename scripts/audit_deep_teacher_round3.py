"""Read-only audit of deep teacher artifacts; never retrains or chooses weights."""
from pathlib import Path
import json,numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,log_loss
from scipy.stats import binomtest
R=Path(__file__).resolve().parents[1];D=R/'results/training_round2/deep_teacher';O=R/'results/training_round3';parts=np.load(R/'results/new_models_20261021/partitions.npz');fit,tune,confirm=[parts[k] for k in ['fit','tune','confirm']];dev=np.r_[fit,tune]
audit={'scope':'Read-only code and artifact audit. Existing inherited holdouts repeatedly reused; comparisons are diagnostic. No weight/threshold selected here.','checks':{},'limitations':['Caches identify rows but not source bytes/configuration; invalidate before future parameter/data changes.','Confirm/full teacher vectorizers and NB model are serialized when those stages are recomputed; context price mappings still require raw/cache. CBM alone is not a standalone scorer.','OOF teacher sees 2/3 train rows whereas eval teacher sees all training rows; standard stacking refit distribution shift remains.','Production historical reference is hybrid: base refitted70k, some auxiliary models frozen on development.','Condition is removed before features; top-level condition attributes filtered. Variation attribute values enter only aggregate counts, not direct value encodings.']}
for stage,tr,ev in [('tune',fit,tune),('confirm',dev,confirm),('full',np.arange(70000),np.arange(70000,100000))]:
 f=D/(stage+'_teacher.npz')
 if not f.exists():audit['checks'][stage+'_teacher']='pending';continue
 z=np.load(f);union=[];checks={'train_rows_match':bool(np.array_equal(z['train_rows'],tr)),'eval_rows_match':bool(np.array_equal(z['eval_rows'],ev)),'all_finite':bool(np.isfinite(z['oof']).all() and np.isfinite(z['p']).all()),'train_eval_disjoint':len(np.intersect1d(tr,ev))==0}
 for i in range(3):
  q=np.load(D/(stage+'_fold'+str(i)+'.npz'));union.extend(q['rows'].tolist());checks['fold'+str(i)+'_eval_disjoint']=len(np.intersect1d(q['rows'],ev))==0;pos=pd.Index(tr).get_indexer(q['rows']);checks['fold'+str(i)+'_matches_stage_oof']=bool(np.array_equal(z['oof'][pos],q['p']))
 checks['fold_union_exact']=bool(np.array_equal(np.sort(union),np.sort(tr)));checks['fold_union_unique']=len(union)==len(set(union));audit['checks'][stage+'_teacher']=checks
x=pd.read_pickle(D/'features.pkl');audit['checks']['base_feature_shape']=list(x.shape);audit['checks']['suspicious_columns']=[c for c in x if any(s in c.lower() for s in ['condition','target','label'])];del x
if (D/'selection_lock.json').exists():audit['selection_lock']=json.loads((D/'selection_lock.json').read_text())
if (D/'confirm.npz').exists():
 a=np.load(D/'confirm.npz');assert np.array_equal(a['rows'],confirm);h=R.parent/'recovered/mercadolibre_checkpoint_20261017/mercadolibre_recovery_20261017/prior_results/warranty/results/composition/holdout_predictions.csv';v=pd.DataFrame({'row':a['rows'],'y':a['y'],'p':a['p']}).merge(pd.read_csv(h),on='row',validate='one_to_one');assert np.array_equal(v.y,v.y_true);th=audit['selection_lock']['threshold'];base=v.refit49k_without_public.to_numpy();p=v.p.to_numpy();old=(base>=.505)==v.y;new=(p>=th)==v.y;g=int((new&~old).sum());l=int((old&~new).sum());blend=.85*base+.15*p
 audit['historical_common']={'n':len(v),'baseline_threshold':.505,'candidate_threshold':th,'historical_accuracy':float(accuracy_score(v.y,base>=.505)),'candidate_accuracy':float(accuracy_score(v.y,p>=th)),'historical_logloss':float(log_loss(v.y,base)),'candidate_logloss':float(log_loss(v.y,p)),'gains':g,'losses':l,'net':g-l,'paired_p_unadjusted':float(binomtest(g,g+l,.5).pvalue) if g+l else 1,'fixed_blend_85historical_accuracy':float(accuracy_score(v.y,blend>=.505)),'fixed_blend_logloss':float(log_loss(v.y,blend))}
 for modelname in ['confirmation_model.cbm','model.cbm']:
  f=D/modelname
  if f.exists():
   try:
    from catboost import CatBoostClassifier
    m=CatBoostClassifier();m.load_model(str(f));audit['checks'][modelname]={'reload':'ok','trees':m.tree_count_}
   except Exception as e:audit['checks'][modelname]={'reload':'failed','error':str(e)}
else:audit['historical_common']='pending confirmation'
for teachername in ['confirm_teacher_model.joblib','full_teacher_model.joblib']:
 f=D/teachername
 if f.exists():
  try:
   import joblib
   obj=joblib.load(f);expected=dev if teachername.startswith('confirm') else np.arange(70000);audit['checks'][teachername]={'reload':'ok','keys':list(obj),'training_rows_match':bool(np.array_equal(obj['training_rows'],expected)),'classes':obj['model'].classes_.tolist(),'feature_width_matches':sum(len(v.get_feature_names_out()) for v in obj['vectorizers'])==len(obj['ratio'])==obj['model'].coef_.shape[1]}
  except Exception as e:audit['checks'][teachername]={'reload':'failed','error':str(e)}
if (D/'test.npz').exists():
 z=np.load(D/'test.npz');audit['checks']['test_vector']={'n':len(z['p']),'finite':bool(np.isfinite(z['p']).all()),'range_ok':bool(((z['p']>=0)&(z['p']<=1)).all())}
(O/'deep_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
