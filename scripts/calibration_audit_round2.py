from pathlib import Path
import numpy as np,pandas as pd,json
from sklearn.metrics import brier_score_loss,log_loss
R=Path(__file__).resolve().parents[1];O=R/'results/training_round2/calibration';O.mkdir(parents=True,exist_ok=True)
z=np.load(R/'results/training_round2/historical_fullaligned/confirmation.npz');rows=z['rows'];y=z['y'];probs={'historical':z['baseline'],'meta':z['p']};sellers=[]
for line in open(R.parent/'recovered/data/train_data.jsonlines'):sellers.append(str(json.loads(line)['seller_id']))
sellers=np.array(sellers);part=np.load(R/'results/new_models_20261021/partitions.npz');seen=np.isin(sellers[rows],sellers[np.r_[part['fit'],part['tune']]])
def metric(y,p):
 ix=np.minimum((p*10).astype(int),9);ece=0;bins=[]
 for k in range(10):
  m=ix==k
  if m.sum():
   gap=float(p[m].mean()-y[m].mean());ece+=m.mean()*abs(gap);bins.append({'bin':k,'n':int(m.sum()),'p_used_mean':float(p[m].mean()),'used_fraction':float(y[m].mean()),'gap':gap})
 return {'brier':float(brier_score_loss(y,p)),'logloss':float(log_loss(y,p)),'ece_10_equal_width':float(ece),'bins':bins}
report={'scope':'Exploratory calibration on reused common4147. ECE depends on bins; seller bootstrap captures within-seller grouping, not model-selection uncertainty. Seen relative newdev56k, not historical49k.','metrics':{}}
for label,p in probs.items():report['metrics'][label]={'all':metric(y,p),'seen':metric(y[seen],p[seen]),'unseen':metric(y[~seen],p[~seen])}
p=probs['meta'];b=probs['historical'];delta=((p>=.5)==y).astype(int)-((b>=.505)==y).astype(int);bd=(p-y)**2-(b-y)**2;ll=-(y*np.log(np.clip(p,1e-12,1))+(1-y)*np.log(np.clip(1-p,1e-12,1)))+(y*np.log(np.clip(b,1e-12,1))+(1-y)*np.log(np.clip(1-b,1e-12,1)))
f=pd.DataFrame({'seller':sellers[rows],'n':1,'accuracy_delta':delta,'brier_delta':bd,'logloss_delta':ll}).groupby('seller').sum();a=f.to_numpy();rng=np.random.default_rng(20261022);res=[]
for _ in range(2000):
 q=a[rng.integers(0,len(a),len(a))].sum(0);res.append(q[1:]/q[0])
res=np.array(res);report['seller_cluster_bootstrap']={'n_groups':len(f),'repetitions':2000,'accuracy_delta':float(delta.mean()),'accuracy_delta_ci95':np.quantile(res[:,0],[.025,.975]).tolist(),'brier_delta':float(bd.mean()),'brier_delta_ci95':np.quantile(res[:,1],[.025,.975]).tolist(),'logloss_delta':float(ll.mean()),'logloss_delta_ci95':np.quantile(res[:,2],[.025,.975]).tolist()};(O/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:{s:{v:m[v] for v in ['brier','logloss','ece_10_equal_width']} for s,m in d.items()} for k,d in report['metrics'].items()},indent=2));print(json.dumps(report['seller_cluster_bootstrap'],indent=2))
