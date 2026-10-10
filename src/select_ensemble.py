from pathlib import Path
import itertools,json
import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
from optimize_leaderboard import OUT,prepare,record

def select():
 d=prepare();y=d['y'][d['v']]
 table=pd.read_csv(OUT/'metrics.csv');table=table[table.protocol=='selection_seed42']
 preds={name:np.load(OUT/(name+'_selection_seed42.npy')) for name in table.model}
 records=[]
 thresholds=np.arange(.44,.561,.005)
 def assess(names,weights):
  p=sum(w*preds[n] for n,w in zip(names,weights));a=np.array([np.mean((p>=q)==y) for q in thresholds]);i=int(np.argmax(a))
  records.append(dict(names=list(names),weights=list(map(float,weights)),threshold=float(thresholds[i]),accuracy=float(a[i])))
 for name in preds:assess([name],[1.])
 for a,b in itertools.combinations(preds,2):
  for w in np.arange(.1,.91,.1):assess([a,b],[w,1-w])
 # Only the strongest individual models enter triples.
 top=table.sort_values('accuracy',ascending=False).model.head(7).tolist()
 if (OUT/'selected_ensemble.json').exists():
  top=list(dict.fromkeys(top+json.loads((OUT/'selected_ensemble.json').read_text())['names']))
 for names in itertools.combinations(top,3):
  for a in [.2,.4,.6]:
   for b in [.2,.4,.6]:
    if a+b>=1:continue
    assess(names,[a,b,1-a-b])
 records.sort(key=lambda r:(-r['accuracy'],len(r['names'])))
 pd.DataFrame([{**r,'names':' + '.join(r['names']),'weights':json.dumps(r['weights'])} for r in records]).to_csv(OUT/'ensemble_search.csv',index=False)
 best=records[0];(OUT/'selected_ensemble.json').write_text(json.dumps(best,indent=2))
 p=sum(w*preds[n] for n,w in zip(best['names'],best['weights']))
 np.save(OUT/'selected_ensemble_selection_seed42.npy',p)
 print('SELECTED',best,flush=True)
 rows=[]
 for a,b in itertools.combinations(preds,2):
  ea=(preds[a]>=.5)!=y;eb=(preds[b]>=.5)!=y
  rows.append({'a':a,'b':b,'a_errors':int(ea.sum()),'b_errors':int(eb.sum()),'both_wrong':int((ea&eb).sum()),'a_wrong_b_right':int((ea&~eb).sum()),'a_right_b_wrong':int((~ea&eb).sum())})
 pd.DataFrame(rows).to_csv(OUT/'error_complementarity.csv',index=False)
 pd.DataFrame({'row_index':d['v'],'y_used':y,'probability_used':p,'prediction_used':(p>=best['threshold']).astype(int),'wrong':(p>=best['threshold'])!=y}).to_csv(OUT/'holdout_predictions.csv',index=False)
 return best
if __name__=='__main__':select()
