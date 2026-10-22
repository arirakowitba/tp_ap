"""Fresh row holdouts; report unseen/seen sellers. No test labels or probe labels in fit."""
from pathlib import Path
import sys,json,time,os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,log_loss
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ap2026q2_common import build_features
OUT=ROOT/'results/training_next/baseline_reproduction';OUT.mkdir(parents=True,exist_ok=True)
def say(*a):print(*a,flush=True)
def metrics(y,p,seen):
 return {'accuracy':float(accuracy_score(y,p>=.5)),'log_loss':float(log_loss(y,p)), 'seen_accuracy':float(accuracy_score(y[seen],p[seen]>=.5)), 'unseen_accuracy':float(accuracy_score(y[~seen],p[~seen]>=.5)), 'n_seen':int(seen.sum()),'n_unseen':int((~seen).sum())}
def main():
 start=time.time();data=Path(os.environ.get('AP2026Q2_DATA_DIR',str(ROOT.parent/'recovered/data')))
 train=pd.read_json(data/'train_data.jsonlines',lines=True);test=pd.read_json(data/'test_data.jsonlines',lines=True)
 assert len(train)==70000 and len(test)==30000
 y=train.condition.eq('used').to_numpy(dtype=int)
 raw=pd.concat([train,test],ignore_index=True)
 X=build_features(raw).drop(columns='title_text')
 for c in ['seller_id','official_store_id','status','international_delivery_mode']:
  X[c]=raw[c].fillna('__missing__').astype(str)
 for c in ['warranty','title','attributes']:
  X['text_'+c]=raw[c].map(lambda v:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else str(v or '__empty__')).str.lower().replace('', '__empty__')
 for c in ['initial_quantity','available_quantity','sold_quantity']:
  X['log_'+c]=np.log1p(pd.to_numeric(raw[c],errors='coerce').clip(lower=0)).astype(float)
 # All transformations above are deterministic; CatBoost fits CTRs and text dictionaries on train only.
 texts=['text_title','text_warranty','text_attributes']
 for c in X:
  if c in texts:continue
  if not pd.api.types.is_numeric_dtype(X[c]):X[c]=X[c].fillna('__missing__').astype(str)
  else:X[c]=X[c].astype(float).fillna(-1)
 dev,confirm=train_test_split(np.arange(70000),test_size=.20,stratify=y,random_state=20261021)
 fit,tune=train_test_split(dev,test_size=.125,stratify=y[dev],random_state=20261022)
 np.savez_compressed(OUT/'partitions.npz',fit=fit,tune=tune,confirm=confirm)
 sellers=raw.seller_id.astype(str).to_numpy();seen_t=np.isin(sellers[tune],sellers[fit]);seen_c=np.isin(sellers[confirm],sellers[dev])
 probs_t={};probs_c={};probs_test={};report={}
 specs={'seller_native':dict(depth=7,iterations=650,learning_rate=.065,l2_leaf_reg=20), 'seller_free_depthwise':dict(depth=8,iterations=650,learning_rate=.06,l2_leaf_reg=25,grow_policy='Depthwise',min_data_in_leaf=80)}
 for name,spec in specs.items():
  if os.environ.get('AP_ONLY_MODEL') and name!=os.environ['AP_ONLY_MODEL']:continue
  cols=[c for c in X if name!='seller_free_depthwise' or c!='seller_id'];cats=[c for c in cols if X[c].dtype=='object' and c not in texts]
  options=dict(**spec,loss_function='Logloss',thread_count=2,random_seed=20261021,verbose=200,allow_writing_files=False,one_hot_max_size=20,
   tokenizers=[{'tokenizer_id':'Space','separator_type':'ByDelimiter','delimiter':' '}],dictionaries=[{'dictionary_id':'Word','max_dictionary_size':'2500','occurrence_lower_bound':'5'}],feature_calcers=['BoW','NaiveBayes'])
  model=CatBoostClassifier(**options);say('FIT TUNE',name,len(fit),len(cols))
  model.fit(X.iloc[fit][cols],y[fit],cat_features=cats,text_features=texts,eval_set=(X.iloc[tune][cols],y[tune]),early_stopping_rounds=90)
  probs_t[name]=model.predict_proba(X.iloc[tune][cols])[:,1];best=max(100,model.tree_count_)
  np.savez_compressed(OUT/(name+'_tune.npz'),rows=tune,p=probs_t[name])
  report[name]={'tune':metrics(y[tune],probs_t[name],seen_t),'iterations':best}
  say('TUNE',name,report[name])
  say('TUNE CHECKPOINT COMPLETE');return
  options['iterations']=best
  model=CatBoostClassifier(**options);model.fit(X.iloc[dev][cols],y[dev],cat_features=cats,text_features=texts)
  probs_c[name]=model.predict_proba(X.iloc[confirm][cols])[:,1]
  report[name]['confirmation']=metrics(y[confirm],probs_c[name],seen_c);say('CONFIRM',name,report[name]['confirmation'])
  np.savez_compressed(OUT/(name+'_confirm.npz'),rows=confirm,p=probs_c[name],y=y[confirm],seen=seen_c)
  (OUT/(name+'_report.json')).write_text(json.dumps({name:report[name]},indent=2))
  if os.environ.get('AP_ONLY_MODEL') and os.environ.get('AP_SKIP_FULL_REFIT')=='1':
   report[name]['full70k_refit']='Skipped after confirmation: diagnostic challenger not selected.'
   (OUT/(name+'_report.json')).write_text(json.dumps({name:report[name]},indent=2))
   return
  model=CatBoostClassifier(**options);model.fit(X.iloc[:70000][cols],y,cat_features=cats,text_features=texts)
  probs_test[name]=model.predict_proba(X.iloc[70000:][cols])[:,1];model.save_model(str(OUT/(name+'.cbm')))
  np.savez_compressed(OUT/(name+'_test.npz'),test=probs_test[name])
  (OUT/'comparison.json').write_text(json.dumps(report,indent=2))
 if os.environ.get('AP_ONLY_MODEL'):
  np.savez_compressed(OUT/(os.environ['AP_ONLY_MODEL']+'_tune.npz'),rows=tune,p=probs_t[os.environ['AP_ONLY_MODEL']])
  (OUT/(os.environ['AP_ONLY_MODEL']+'_report.json')).write_text(json.dumps(report,indent=2))
  return
 # Select one global blend on tune; confirmation is assessed once for that selection.
 names=list(specs);grid=[]
 for w in [0,.25,.5,.75,1]:
  p=w*probs_t[names[0]]+(1-w)*probs_t[names[1]]
  grid.append((float(accuracy_score(y[tune],p>=.5)),w))
 w=max(grid)[1];pc=w*probs_c[names[0]]+(1-w)*probs_c[names[1]]
 pt=w*probs_test[names[0]]+(1-w)*probs_test[names[1]]
 report['selected_blend']={'seller_native_weight':w,'tune_grid':grid,'confirmation':metrics(y[confirm],pc,seen_c)}
 base=pd.read_csv(ROOT/'submissions/20261019/submission_15_top16_Q4_sobre_mejor11.csv')
 base_y=base.condition.eq('used').to_numpy()
 out=pd.DataFrame({'ID':np.arange(1,30001),'p_used':pt,'native_p_used':probs_test[names[0]],'seller_free_p_used':probs_test[names[1]],'seller_seen':np.isin(sellers[70000:],sellers[:70000]),'p_alternative_to_15':np.where(base_y,1-pt,pt)})
 out.to_csv(OUT/'test_predictions.csv',index=False)
 report['seconds']=time.time()-start;report['protocol']='Fresh deterministic row split: 49k fit/7k tune; refit56k/14k confirmation. Seen/unseen defined relative to each training set. Confirmation not used to select weights. Historical methods may have used these rows before.'
 (OUT/'comparison.json').write_text(json.dumps(report,indent=2));say('DONE',report)
if __name__=='__main__':main()
