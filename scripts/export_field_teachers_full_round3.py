"""Full refit from locked field-teacher configuration; no selection/evaluation."""
from pathlib import Path
import json,time,gc,os,joblib
import numpy as np
import train_field_teachers_round3 as s

def main():
 lock=json.loads((s.O/'selection_lock.json').read_text());X,_,y,sellers,qty=s.parent.load();raw=s.read_fields();tr=np.arange(70000);ev=np.arange(70000,100000)
 a,b=s.matrices(X,raw,y,qty,tr,ev,'full');opts=lock['catboost_params'].copy();opts['iterations']=lock['trees'];s.state('catboost_full_locked',trees=lock['trees'],nfeatures=a.shape[1]);cats=a.select_dtypes(include='object').columns.tolist();m=s.cat(opts,'full');m.fit(a,y[:70000],cat_features=cats);p=m.predict_proba(b)[:,1];assert len(p)==30000 and np.isfinite(p).all();s.save_npz(s.O/'test.npz',p=p);tmp=s.O/'model_atomic.cbm';m.save_model(str(tmp));check=s.CatBoostClassifier();check.load_model(str(tmp));assert np.max(np.abs(check.predict_proba(b.iloc[:100])[:,1]-p[:100]))<1e-10;os.replace(tmp,s.O/'model.cbm')
 context={key:X.iloc[tr].groupby(key)['log_price'].agg(['mean','std','size']) for key in ['category_id','seller_id']};context['feature_names']=list(a.columns);context['categorical_features']=cats;context['threshold']=lock['threshold'];tmp=s.O/'context_stats.tmp';joblib.dump(context,tmp,compress=3);joblib.load(tmp);os.replace(tmp,s.O/'context_stats.joblib')
 report=json.loads((s.O/'report.json').read_text());report['full_refit']='complete: locked configuration, all70000 labels, each training teacher strictly3foldOOF';report['test_rows']=30000;report['total_seconds_with_export']=report.get('seconds',0)+time.time()-s.START;(s.O/'report.json').write_text(json.dumps(report,indent=2));(s.O/'inference_manifest.json').write_text(json.dumps({'catboost':'model.cbm','title_teacher':'full_title_model.joblib','warranty_teacher':'full_warranty_model.joblib','combined_teacher':'../../training_round2/deep_teacher/full_teacher_model.joblib','context_stats':'context_stats.joblib','threshold':lock['threshold'],'features':list(a.columns),'feature_builder':'src/ap2026q2_common.py + src/ap2026q2_deep_features.py + parent.attach + fields.matrices','note':'Context maps are frozen full70k statistics. Avoid recomputing on new query batches. Other deep features are deterministic per row. No public-label locks.'},indent=2));s.state('done_full_refit',n_test=30000)
if __name__=='__main__':main()
