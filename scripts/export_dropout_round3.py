"""Complete a selected dropout refit using parent's full OOF teacher cache."""
import time,json,gc
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier
import train_seller_dropout_round3 as d

def main():
 reportpath=d.O/'report.json';deadline=time.time()+1200
 while not reportpath.exists():
  if time.time()>deadline:return
  time.sleep(10)
 report=json.loads(reportpath.read_text());reference=report.get('deep_baseline',report['native'])
 if report['confirmation']['test_mix_accuracy']<=reference['test_mix_accuracy']:
  print('No full refit: did not improve reference',flush=True);return
 if (d.O/'test.npz').exists():print('Full test already exists',flush=True);return
 full=d.D/'full_teacher.npz'
 while not full.exists():
  d.state('await_parent_full_teacher_for_export')
  if time.time()>deadline:
   report['test_export']='Pending parent full teacher cache after wait budget';reportpath.write_text(json.dumps(report,indent=2));return
  time.sleep(15)
 X,docs,y,sellers,qty=d.base.load();del docs;gc.collect();tr=np.arange(70000);ev=np.arange(70000,100000);po,pe=d.wait_teacher('full',tr,ev);A,B=d.base.attach(X,tr,ev,po,pe,qty);choice=report['choice'];train,n=d.dropout(A,tr,choice['dropout_fraction']);cats=A.select_dtypes(include='object').columns.tolist();d.state('full_fit_after_parent_cache',n_masked=n);m=CatBoostClassifier(**d.opts(choice['iterations'],stage='full'));m.fit(train,y[tr],cat_features=cats);p=m.predict_proba(B)[:,1];assert np.isfinite(p).all() and len(p)==30000;np.savez_compressed(d.O/'test.npz',p=p);m.save_model(str(d.O/'model.cbm'));m2=CatBoostClassifier();m2.load_model(str(d.O/'model.cbm'));assert m2.tree_count_==m.tree_count_;report['test_export']='Complete: parent full70k teacher with OOF training probabilities and reload-verified CatBoost';reportpath.write_text(json.dumps(report,indent=2));d.state('full_export_complete')
if __name__=='__main__':main()
