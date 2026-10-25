import csv,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import beta as beta_dist
R=Path(__file__).resolve().parent
h=np.load(R/'round13/hist.npz');c=np.load(R/'round13/cat.npz')
assert np.array_equal(h['rows'],c['rows']) and np.array_equal(h['y'],c['y']) and np.array_equal(h['seen'],c['seen'])
t=np.load(R/'work20/round30_full70k_test_probabilities.npz')
ids=t['kaggle_ID'];assert np.array_equal(ids,np.arange(1,30001))
test_blend=t['p_blend_used'];test_seen=t['seller_seen'];test_hist=t['p_hist_used'];test_cat=t['p_cat_used']
def binid(p):return min(9,int(p*10))
def posterior(seen,b,p):
 n,s=cells.get((seen,b),(0,0));center=(b+.5)/10
 return (s+10*center)/(n+10),n,s
cells=defaultdict(lambda:[0,0]); pv=(h['p']+c['p'])/2
for p,y,seen in zip(pv,h['y'],h['seen']):
 a=cells[(bool(seen),binid(float(p)))];a[0]+=1;a[1]+=int(y)
cell_params={}
for seen in [False,True]:
 for b in range(10):
  n,s=cells[(seen,b)];alpha=s+10*(b+.5)/10;beta=n-s+10*(1-(b+.5)/10)
  cell_params[(seen,b)]=(alpha,beta,float(beta_dist.ppf(.05,alpha,beta)),float(beta_dist.ppf(.95,alpha,beta)))
base=[x['condition'] for x in csv.DictReader(open(R/'work20/submission_33_gate_stable_next31_sobre31.csv'))]
allrows={}
for n in [4,5,6,7,8,9]+list(range(11,42)):
 path=R/'replayed'/f'submission_{n:02d}.csv'
 if path.exists():allrows[n]=[x['condition'] for x in csv.DictReader(open(path))]
assert len(allrows)==37
vote_counts=np.sum(np.array([[v=='used' for v in a] for a in allrows.values()],dtype=np.uint8),axis=0)
scores={1:.94433,2:.94419,3:.94447,4:.94457,5:.94447,6:.94457,7:.94476,10:.94500,11:.94519,12:.94419,13:.94428,15:.94523,16:.94500,17:.94447,18:.94404,19:.94409,20:.94285,21:.94504,22:.94471,23:.94495,24:.94533,25:.94519,31:.94585,32:.94561,33:.94614,34:.94580,35:.94609,36:.94571,37:.94490,38:.94423,39:.92976,40:.94595}
for n in [1,2,3,10]:assert n not in allrows
out=R/'AP_2026_Q2_row_distributions_20261025.csv'
with out.open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['ID','prediction_33','seller_segment','p_used_model_raw','p_used_train_binned','p_used_cell_low90','p_used_cell_high90','beta_alpha','beta_beta','train_cell_n','train_cell_used','p_33_correct_model','expected_flip_gain_vs33','history_used_votes_of37','history_predictions_available','p_hist_used','p_cat_used','public_row_impact_identifiable'])
 for k,i in enumerate(ids):
  p=float(test_blend[k]);seen=bool(test_seen[k]);q,n,s=posterior(seen,binid(p),p);correct=q if base[k]=='used' else 1-q
  votes=int(vote_counts[k])
  alpha,beta,lo,hi=cell_params[(seen,binid(p))]
  w.writerow([int(i),base[k],'seen' if seen else 'new',round(p,6),round(q,6),round(lo,6),round(hi,6),round(alpha,4),round(beta,4),n,s,round(correct,6),round(1-2*correct,6),votes,37,round(float(test_hist[k]),6),round(float(test_cat[k]),6),'no'])
report={'method':'Bernoulli per-row probability from full70k Hist/Cat blend, recalibrated with 14k labeled train third split by seller-seen x probability decile using 10-count prior at decile midpoint. Historical vote counts are correlated descriptions, not independent likelihood terms. Public scores enter group ledger only; row impact not identifiable. Third split not virgin.','train_cells':[{'seen':seen,'bin':b,'n':posterior(seen,b,0)[1],'used':posterior(seen,b,0)[2],'p_used':round(posterior(seen,b,0)[0],5)} for seen in [False,True] for b in range(10)],'submission_scores':scores,'pending':[8,9,14,26,27,28,29,30,41],'exact_prediction_coverage':sorted(allrows),'row_count':30000}
for sub in [35,36,37,38,39,40,41]:
 a=allrows[sub];idx=[k for k in range(30000) if a[k]!=base[k]]
 gains=[]
 for k in idx:
  p=float(test_blend[k]);q=posterior(bool(test_seen[k]),binid(p),p)[0];g=(2*q-1) if a[k]=='used' else (1-2*q);gains.append(g)
 report.setdefault('expected_gain_diagnostics',{})[str(sub)]={'n':len(idx),'expected_gain_total':round(sum(gains),2),'expected_gain_new_sellers':round(sum(g for g,k in zip(gains,idx) if not test_seen[k]),2),'note':'model expectation only; not a public score estimate or validation against 33'}
(R/'AP_2026_Q2_uncertainty_report_20261025.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report['expected_gain_diagnostics'],indent=2));print(out.stat().st_size)
