"""Predeclared 85% historical / 15% NB comparison on common excluded rows.
No weight search; these reused holdouts are diagnostic, not fresh validation.
"""
from pathlib import Path
import numpy as np,pandas as pd,json
from sklearn.metrics import accuracy_score,log_loss
from scipy.stats import binomtest
R=Path(__file__).resolve().parents[1];D=R/'results/training_next/nbsvm'
a=np.load(D/'confirmation.npz');b=np.load(R/'results/new_models_20261021/seller_native_confirm.npz')
n=pd.DataFrame({'row':a['rows'],'y':a['y'],'nb':a['p'],'native':b['p'],'seen_new_train':a['seen']})
h=R.parent/'recovered/mercadolibre_checkpoint_20261017/mercadolibre_recovery_20261017/prior_results/warranty/results/composition/holdout_predictions.csv'
x=n.merge(pd.read_csv(h),on='row',validate='one_to_one');assert len(x)==4147 and np.array_equal(x.y,x.y_true)
def metrics(p):
 o={'accuracy':float(accuracy_score(x.y,p>=.5)),'log_loss':float(log_loss(x.y,p))}
 for seen in [True,False]:
  m=x.seen_new_train==seen;o['seen' if seen else 'unseen']={'n':int(m.sum()),'accuracy':float(accuracy_score(x.y[m],p[m]>=.5))}
 return o
report={'n_common':len(x),'historical_source':str(h.relative_to(R.parent)),'weight_predeclared':.85,'primary_historical_column':'refit49k_matched_without_public','scope':'Intersection confirmation and historical holdout excluded from both training sets. Old holdout repeatedly reused; NB hyperparameters were chosen on tune rows that overlap old development/holdout. Diagnostic only, no confirmation weight tuning. Seller segmentation relative to NB56k, not historical49k.','models':{}}
for name,col in [('primary','refit49k_matched_without_public'),('historical_without_public','refit49k_without_public')]:
 old=x[col].to_numpy();p=.85*old+.15*x.nb.to_numpy();y=x.y.to_numpy();oc=(old>=.5)==y;nc=(p>=.5)==y;g=int((nc&~oc).sum());l=int((oc&~nc).sum());report['models'][name]={'historical':metrics(old),'blend':metrics(p),'gains':g,'losses':l,'net':g-l,'paired_p_unadjusted':float(binomtest(g,g+l,.5).pvalue) if g+l else 1.}
 x[name+'_blend']=p
report['models']['native']=metrics(x.native.to_numpy());report['models']['nb_standalone']=metrics(x.nb.to_numpy());report['models']['native_nb_fixed_blend']=metrics(.85*x.native.to_numpy()+.15*x.nb.to_numpy())
(D/'historical_matched_comparison.json').write_text(json.dumps(report,indent=2));x[['row','y','nb','native','seen_new_train','refit49k_matched_without_public','refit49k_without_public','primary_blend','historical_without_public_blend']].to_csv(D/'historical_matched_probabilities.csv',index=False);print(json.dumps(report,indent=2))
