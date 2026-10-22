"""Diagnostic evidence matrix; no fitted predictive model or public labels."""
from pathlib import Path
import json,numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,log_loss,brier_score_loss
R=Path(__file__).resolve().parents[1];O=R/'results/training_round3';O.mkdir(parents=True,exist_ok=True)
raw=[]
for line in open(R.parent/'recovered/data/train_data.jsonlines'):
 x=json.loads(line);raw.append({'seller':str(x['seller_id']),'category':str(x['category_id']),'quantity_one':float(x.get('initial_quantity') or 0)<=1,'listing':str(x.get('listing_type_id','')),'warranty_present':bool(x.get('warranty'))})
raw=pd.DataFrame(raw)
a=np.load(R/'results/new_models_20261021/seller_native_confirm.npz');nb=np.load(R/'results/training_next/nbsvm/confirmation.npz');st=np.load(R/'results/training_next/stacker/confirmation.npz');b=np.load(R/'results/training_round2/fresh/confirmation.npz')
assert np.array_equal(a['rows'],nb['rows']) and np.array_equal(a['rows'],st['rows'])
parts={'first':{'rows':a['rows'],'y':a['y'],'seen':a['seen'],'native':a['p'],'nb':nb['p'],'blend85_15':nb['blend'],'meta':st['p'],'meta_definition':'Five logits (native,NB,rich,retrieval,context) + interactions; C selected innerCV tune.'},'second':{'rows':b['rows'],'y':b['y'],'seen':b['seen'],'native':b['native'],'nb':b['nb'],'blend85_15':b['blend'],'meta':b['meta'],'meta_definition':'Two logits (native,NB) + predefined gates, C=.1 fixed.'}}
def metrics(y,p):return {'n':len(y),'accuracy':float(accuracy_score(y,p>=.5)),'logloss':float(log_loss(y,p)),'brier':float(brier_score_loss(y,p))}
def paired(y,p,base):
 old=(base>=.5)==y;new=(p>=.5)==y
 return {'fixes':int((new&~old).sum()),'breaks':int((old&~new).sum()),'net':int(new.sum()-old.sum())}
def bootstrap(rows,y,p,base,seed):
 d=pd.DataFrame({'seller':raw.seller.iloc[rows].to_numpy(),'n':1,'delta':((p>=.5)==y).astype(int)-((base>=.5)==y).astype(int)}).groupby('seller').sum().to_numpy();rng=np.random.default_rng(seed);out=[]
 for _ in range(1500):
  q=d[rng.integers(0,len(d),len(d))].sum(0);out.append(q[1]/q[0])
 return {'seller_groups':len(d),'repetitions':1500,'accuracy_delta':float(np.mean(((p>=.5)==y).astype(int)-((base>=.5)==y).astype(int))),'ci95':np.quantile(out,[.025,.975]).tolist(),'scope':'Cluster conditional on fitted models; excludes selection uncertainty and repeated dataset reuse.'}
report={'scope':'All evidence uses train labels. The two partitions share rows and sellers; do not pool as independent trials. Different meta definitions prevent direct replication claim for meta. Same fixed85/15 blend does replicate directional contribution.','partitions':{}}
slices=[];cats=[]
for name,z in parts.items():
 y=z['y'];rows=z['rows'];d=raw.iloc[rows].reset_index(drop=True);seen=z['seen'];out={'n':len(y),'meta_definition':z['meta_definition'],'models':{},'seller_bootstrap':{}}
 for model in ['native','nb','blend85_15','meta']:
  p=z[model];out['models'][model]={'all':metrics(y,p),'seen':metrics(y[seen],p[seen]),'unseen':metrics(y[~seen],p[~seen]),'paired_vs_native':paired(y,p,z['native'])}
 for model in ['blend85_15','meta']:out['seller_bootstrap'][model]=bootstrap(rows,y,z[model],z['native'],20261023+(model=='meta'))
 masks={'all':np.ones(len(y),bool),'new_seller':~seen,'seen_seller':seen,'quantity_one':d.quantity_one.to_numpy(),'quantity_many':~d.quantity_one.to_numpy(),'new_seller_quantity_one':~seen&d.quantity_one.to_numpy(),'seen_seller_quantity_one':seen&d.quantity_one.to_numpy(),'listing_free':d.listing.eq('free').to_numpy(),'listing_bronze':d.listing.eq('bronze').to_numpy(),'title_model_disagrees_native':(z['nb']>=.5)!=(z['native']>=.5),'used_label':y==1,'new_label':y==0}
 for seg,m in masks.items():
  for model in ['blend85_15','meta']:
   q=paired(y[m],z[model][m],z['native'][m]);slices.append({'partition':name,'segment':seg,'model':model,'n':int(m.sum()),'baseline_accuracy':float(np.mean((z['native'][m]>=.5)==y[m])),'accuracy':float(np.mean((z[model][m]>=.5)==y[m])),**q})
 for c,ix in d.groupby('category').groups.items():
  ix=np.array(list(ix));
  if len(ix)>=30:cats.append({'partition':name,'category':c,'n':len(ix),**paired(y[ix],z['meta'][ix],z['native'][ix])})
 report['partitions'][name]=out
common=np.intersect1d(a['rows'],b['rows']);report['dependence']={'overlap_confirm_rows':len(common),'overlap_sellers':len(set(raw.seller.iloc[a['rows']])&set(raw.seller.iloc[b['rows']])),'first_unique_sellers':raw.seller.iloc[a['rows']].nunique(),'second_unique_sellers':raw.seller.iloc[b['rows']].nunique(),'caution':'Overlap not the only dependence; models share most train data and history informed all designs.'}
cal=json.loads((R/'results/training_round2/calibration/report.json').read_text());report['historical_comparison_uncertainty']=cal['seller_cluster_bootstrap'];report['historical_meta_calibration']={k:{s:{m:q[m] for m in ['brier','logloss','ece_10_equal_width']} for s,q in d.items()} for k,d in cal['metrics'].items()}
report['next_experiments']=[{'name':'seller dropout on strong deep teacher','rationale':'Replicated fixed-blend gains concentrate on unseen sellers, while removing seller entirely previously harmed known sellers. Dropout can preserve known-seller value while improving fallback. Already in progress; prioritize its tune-only locked result.','validation':'Compare seller dropout versus same deep baseline, especially new-seller quantity1. Use same rows, locked test-mix weighting and final untouched-by-selection comparison.'},{'name':'field-specific text teacher with explicit negation context','rationale':'All-title NB is useful as complement but poor standalone. Used/new mentions are context-dependent; split title and warranty teachers can distinguish "nuevo modelo" from "sin uso" and counter noisy warranty text.','validation':'Cross-fit every teacher inside fit/development, then lock a small model family on tune. Do not choose regex or field weights from public flip outcomes.'},{'name':'seller-disjoint outer check of the chosen strong stack','rationale':'Row splits reproduce marginal test mixture but still permit seller lookup for known sellers. A seller-disjoint outer split can test whether new-seller gains are transferable rather than an idiosyncratic row-split artifact.','validation':'Fixed strongest architecture, no broad hyperparameter search on outer seller split; report separately and reweight only as sensitivity, not a substitute for actual test performance.'}]
(O/'robustness.json').write_text(json.dumps(report,indent=2));pd.DataFrame(slices).to_csv(O/'robustness_slices.csv',index=False);pd.DataFrame(cats).to_csv(O/'robustness_categories.csv',index=False)
print(json.dumps({'dependence':report['dependence'],'seller_bootstrap':{k:v['seller_bootstrap'] for k,v in report['partitions'].items()}},indent=2))
