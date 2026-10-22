from pathlib import Path
import json,numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,log_loss
from scipy.stats import binomtest
R=Path(__file__).resolve().parents[1];D=R/'results/training_next/stacker'
a=np.load(D/'confirmation.npz');h=R.parent/'recovered/mercadolibre_checkpoint_20261017/mercadolibre_recovery_20261017/prior_results/warranty/results/composition/holdout_predictions.csv'
x=pd.DataFrame({'row':a['rows'],'y':a['y'],'stacker':a['p'],'seen':a['seen']}).merge(pd.read_csv(h),on='row',validate='one_to_one');assert np.array_equal(x.y,x.y_true)
def met(p):
 return {'accuracy':float(accuracy_score(x.y,p>=.5)),'log_loss':float(log_loss(x.y,p))}
o={'n_common':len(x),'scope':'Diagnostic intersection excluded from both base training sets. Reused holdouts and tune overlap prevent a virgin confirmation claim. No coefficient selected here.','stacker':met(x.stacker),'historical':{}}
for c in ['refit49k_matched_without_public','refit49k_without_public']:
 old=(x[c]>=.5)==x.y;new=(x.stacker>=.5)==x.y;g=int((new&~old).sum());l=int((old&~new).sum());o['historical'][c]={**met(x[c]),'stacker_gains':g,'stacker_losses':l,'net':g-l,'paired_p_unadjusted':float(binomtest(g,g+l,.5).pvalue) if g+l else 1}
(D/'historical_comparison.json').write_text(json.dumps(o,indent=2));x.to_csv(D/'historical_matched.csv',index=False);print(json.dumps(o,indent=2))
