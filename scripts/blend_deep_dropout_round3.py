from pathlib import Path
import numpy as np,json
from sklearn.metrics import accuracy_score,log_loss
R=Path(__file__).resolve().parents[1];A=R/'results/training_round2/deep_teacher';B=R/'results/training_round3/dropout';O=R/'results/training_round3/deep_dropout_blend';O.mkdir(parents=True,exist_ok=True)
a=np.load(A/'tune.npz');b=np.load(B/'tune_0.2.npz') if (B/'tune_0.2.npz').exists() else None
if b is None:raise SystemExit('Find selected tune file first; no substitution')
assert np.array_equal(a['rows'],b['rows']);inputs=np.load(A/'input.npz',allow_pickle=True);y=inputs['y'][a['rows']];part=np.load(R/'results/new_models_20261021/partitions.npz');seen=np.isin(inputs['sellers'][a['rows']],inputs['sellers'][part['fit']]);p=.5*a['p']+.5*b['p'];grid=[]
for t in [.475,.49,.5,.51,.525]:
 q=(p>=t)==y;grid.append({'threshold':t,'reweighted_accuracy':float(.716*q[seen].mean()+.284*q[~seen].mean())})
t=max(grid,key=lambda q:(q['reweighted_accuracy'],-abs(q['threshold']-.5)))['threshold'];lock={'weight_deep':.5,'threshold':t,'grid':grid,'selection':'Fixed50/50, thresholdselectedtuneonly. Exploratory design after individual confirms, not independent confirmation.'};(O/'selection_lock.json').write_text(json.dumps(lock,indent=2));a=np.load(A/'confirm.npz');b=np.load(B/'confirm.npz');assert np.array_equal(a['rows'],b['rows']);p=.5*a['p']+.5*b['p'];y=a['y'];report={**lock,'confirmation':{'accuracy':float(accuracy_score(y,p>=t)),'log_loss':float(log_loss(y,p))}};np.savez_compressed(O/'confirm.npz',rows=a['rows'],p=p,y=y);(O/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if (A/'test.npz').exists() and (B/'test.npz').exists():np.savez_compressed(O/'test.npz',p=.5*np.load(A/'test.npz')['p']+.5*np.load(B/'test.npz')['p'])
