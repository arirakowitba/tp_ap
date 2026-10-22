"""Exact finite-population cohort decoding under INDEPENDENT fixed priors.
Experimental probabilities only: no fit, pseudo-labels, submissions or expected score.
"""
from pathlib import Path
from itertools import combinations,product
from math import comb
import json,hashlib,argparse
import numpy as np
import pandas as pd
from scipy.special import gammaln
R=Path(__file__).resolve().parents[1]
O=R/'results/training_round3/cohort_posterior'

def validate_groups(groups,N,K):
 if not isinstance(N,(int,np.integer)) or not isinstance(K,(int,np.integer)) or not 0<=K<=N:raise ValueError('Require integer 0<=K<=N')
 allids=[]
 for g in groups:
  ids=np.asarray(g['ids']);p=np.asarray(g['p'],float);d=g['d']
  if len(ids)==0 or len(ids)!=len(p):raise ValueError('Empty group or unequal lengths')
  if not isinstance(d,(int,np.integer)) or isinstance(d,(bool,np.bool_)) or abs(d)>len(ids):raise ValueError('Incompatible integer cohort balance')
  if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Priors must be finite probabilities')
  if not np.issubdtype(ids.dtype,np.integer) or np.any((ids<1)|(ids>N)):raise ValueError('IDs must be integers in 1..N')
  allids.extend(ids.tolist())
 if len(allids)!=len(set(allids)):raise ValueError('Overlapping groups or repeated IDs')
 if len(allids)>N:raise ValueError('Union larger than population')
 return len(allids)

def group_dp(p):
 """Coefficient array D[m,s+n]; mask assignments already enumerated."""
 p=np.asarray(p,float);n=len(p);a=np.zeros((n+1,2*n+1));a[0,n]=1
 for i,q in enumerate(p):
  b=a.copy()  # Hidden: sum_Y Bernoulli(q)=1.
  b[1:,:-1]+=(1-q)*a[:-1,1:]  # Public and wrong: balance -1.
  b[1:,1:]+=q*a[:-1,:-1]     # Public and correct: balance +1.
  a=b
 return a

def mass_at_balance(p,d):
 n=len(p)
 if abs(d)>n:return np.zeros(n+1)
 return group_dp(p)[:,n+d]

def logchoose(n,k):
 k=np.asarray(k);out=np.full(k.shape,-np.inf,dtype=float);ok=(k>=0)&(k<=n)
 out[ok]=gammaln(n+1)-gammaln(k[ok]+1)-gammaln(n-k[ok]+1)
 return out

def decode(groups,N,K,clamp=1e-6):
 U=validate_groups(groups,N,K);ps=[np.clip(np.asarray(g['p'],float),clamp,1-clamp) for g in groups]
 masses=[mass_at_balance(p,g['d']) for p,g in zip(ps,groups)]
 prefix=[np.array([1.])]
 for a in masses:prefix.append(np.convolve(prefix[-1],a))
 suffix=[None]*(len(groups)+1);suffix[-1]=np.array([1.])
 for j in range(len(groups)-1,-1,-1):suffix[j]=np.convolve(masses[j],suffix[j+1])
 # For EACH assignment of the U mask bits, outside completions supply this weight.
 # Do not multiply by choose(U,m): the DPs already count those assignments.
 logs=logchoose(N-U,K-np.arange(U+1));finite=np.isfinite(logs)
 if not finite.any():raise ValueError('No compatible fixed-size mask')
 scale=float(logs[finite].max());weights=np.zeros(U+1);weights[finite]=np.exp(logs[finite]-scale)
 den=float(prefix[-1]@weights)
 if not np.isfinite(den) or den<=0:raise ValueError('Scores incompatible with groups and fixed public mask')
 rows=[];summary=[]
 for j,(g,p,gmass) in enumerate(zip(groups,ps,masses)):
  other=np.convolve(prefix[j],suffix[j+1]);n=len(p);d=g['d']
  def weighted(v,shift=0):
   c=np.convolve(v,other);return float(c@weights[shift:shift+len(c)])
  mg=np.array([gmass[m]*float(other@weights[m:m+len(other)])/den for m in range(n+1)])
  if abs(mg.sum()-1)>1e-10:raise AssertionError('Group public-count distribution not normalized')
  start=len(rows)
  for i,(ident,q) in enumerate(zip(g['ids'],p)):
   rest=np.delete(p,i)
   hidden=weighted(mass_at_balance(rest,d))
   public_correct_no_prior=weighted(mass_at_balance(rest,d-1),1)
   public_wrong_no_prior=weighted(mass_at_balance(rest,d+1),1)
   pc=q*public_correct_no_prior;pw=(1-q)*public_wrong_no_prior
   if abs((hidden+pc+pw)/den-1)>1e-10:raise AssertionError('Item decomposition not normalized')
   ph=hidden/den;pp=(pc+pw)/den
   correct=q*(hidden+public_correct_no_prior)/den
   # This conditional identity is a THEOREM of independence, not hidden transfer.
   ch=(q*hidden/den)/ph if ph>0 else np.nan
   cp=pc/(pc+pw) if pc+pw>0 else np.nan
   rows.append({'ID':int(ident),'group':g['name'],'observed_public_net_balance':int(d),'prior_p_alt_correct_raw':float(g['p'][i]),'prior_p_alt_correct_clamped':float(q),'posterior_p_alt_correct_marginal':correct,'posterior_p_public':pp,'posterior_p_hidden':ph,'posterior_p_alt_correct_given_hidden':ch,'posterior_p_alt_correct_given_public':cp,'hidden_conditional_minus_prior':ch-q if ph>0 else np.nan})
  rr=rows[start:]
  summary.append({'group':g['name'],'n':n,'observed_balance':int(d),'posterior_public_count_distribution':mg.tolist(),'prior_mean_p_alt_correct':float(p.mean()),'posterior_mean_p_alt_correct_marginal':float(np.mean([r['posterior_p_alt_correct_marginal'] for r in rr])),'posterior_mean_p_public':float(np.mean([r['posterior_p_public'] for r in rr]))})
 log_evidence=float(np.log(den)+scale-float(logchoose(N,np.array(K))))
 result=pd.DataFrame(rows)
 if not result.empty:
  cols=['posterior_p_alt_correct_marginal','posterior_p_public','posterior_p_hidden']
  if ((result[cols]<-1e-10)|(result[cols]>1+1e-10)).any().any():raise AssertionError('Posterior outside probability range')
  if result.hidden_conditional_minus_prior.abs().max()>1e-12:raise AssertionError('Independence conditional identity failed')
 return result,{'N':int(N),'K':int(K),'U':U,'n_groups':len(groups),'prior_clamp':clamp,'n_priors_clamped':int(sum(np.sum(np.asarray(g['p'])!=p) for g,p in zip(groups,ps))),'log_probability_of_observed_constraints':log_evidence,'groups':summary}

def brute_force(groups,N,K):
 ids=[i for g in groups for i in g['ids']];p=np.concatenate([g['p'] for g in groups]);pos={i:j for j,i in enumerate(ids)};n=len(ids);den=0.;cy=np.zeros(n);pub=np.zeros(n);hy=np.zeros(n);hid=np.zeros(n)
 for maskset in combinations(range(1,N+1),K):
  mask=set(maskset);mm=np.array([i in mask for i in ids],bool)
  for yy in product([0,1],repeat=n):
   yy=np.array(yy)
   if any(sum((2*yy[pos[i]]-1) for i in g['ids'] if i in mask)!=g['d'] for g in groups):continue
   w=float(np.prod(np.where(yy,p,1-p)))/comb(N,K);den+=w;cy+=w*yy;pub+=w*mm;hid+=w*(~mm);hy+=w*yy*(~mm)
 if not den:raise ValueError('Incompatible toy observations')
 return {'ID':ids,'posterior_p_alt_correct_marginal':cy/den,'posterior_p_public':pub/den,'posterior_p_hidden':hid/den,'posterior_p_alt_correct_given_hidden':np.divide(hy,hid,out=np.full(n,np.nan),where=hid>0)},den

def tests():
 gs=[{'name':'toyA','ids':np.array([1,2]),'p':np.array([.15,.72]),'d':1},{'name':'toyB','ids':np.array([3,4,5]),'p':np.array([.4,.9,.33]),'d':0}]
 d,s=decode(gs,8,4);b,e=brute_force(gs,8,4);err=max(float(np.max(np.abs(d[k]-b[k]))) for k in b if k!='ID');assert err<1e-10
 evidence_error=abs(np.exp(s['log_probability_of_observed_constraints'])-e);assert evidence_error<1e-10
 dp=group_dp(np.array([.1,.3,.8,.9]));assert np.allclose(dp.sum(1),[comb(4,m) for m in range(5)],rtol=1e-12,atol=1e-12)
 rejected=[]
 for name,bad,N,K in [('overlap',[gs[0],{'name':'bad','ids':np.array([2]),'p':np.array([.4]),'d':0}],8,4),('balance',[{'name':'bad','ids':np.array([1,2]),'p':np.array([.4,.5]),'d':3}],8,4),('fixed_mask_impossible',gs,8,0)]:
  try:decode(bad,N,K)
  except ValueError:rejected.append(name)
  else:raise AssertionError('Failed to reject '+name)
 return {'toy_N':8,'toy_K':4,'toy_union':5,'label_configurations':32,'mask_configurations':70,'max_absolute_posterior_error_vs_bruteforce':err,'evidence_absolute_error_vs_bruteforce':evidence_error,'dp_mask_count_identity':'passed','rejected_invalid_inputs':rejected}

def load_groups(prior_file):
 z=np.load(prior_file);p=np.asarray(z['p'],float)
 if p.shape!=(30000,) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Require ordered30000 p_used probabilities')
 base=pd.read_csv(R/'submissions/20261019/submission_15_top16_Q4_sobre_mejor11.csv').set_index('ID');assert np.array_equal(np.sort(base.index),np.arange(1,30001))
 a=pd.read_csv(R/'submissions/20261020/cohorte_16_auditoria.csv');a=a[a.full70k_seed_votes_alternative_out_of_7>=4].copy();assert len(a)==32
 e=pd.read_csv(R/'submissions/20261020/cohorte_E_auditoria.csv');top=pd.read_csv(R/'submissions/20261021/submission_25_E2_top16_auditoria.csv');assert len(top)==16
 e2=e[e.subgroup.str.startswith('E2_')].copy();assert len(e2)==32 and set(top.ID).issubset(set(e2.ID));assert not top.ID.duplicated().any()
 frames=[('A32',a,'label_in_15','label_in_16',2),('E1',e[e.subgroup.str.startswith('E1_')],'historical_control_label','historical_alternative_label',-6),('E2_top16',e2[e2.ID.isin(top.ID)],'historical_control_label','historical_alternative_label',-3),('E2_complement16',e2[~e2.ID.isin(top.ID)],'historical_control_label','historical_alternative_label',3),('E3',e[e.subgroup.str.startswith('E3_')],'historical_control_label','historical_alternative_label',-7),('E4',e[e.subgroup.str.startswith('E4_')],'historical_control_label','historical_alternative_label',-2)]
 for c in ['historical_control_label','historical_alternative_label']:
  assert np.array_equal(top.set_index('ID')[c].sort_index(),e2.set_index('ID').loc[top.ID,c].sort_index())
 groups=[];orientation=[]
 for name,f,control,alt,d in frames:
  assert len(f)==(16 if name.startswith('E2_') else 32);ids=f.ID.to_numpy(dtype=int);bl=base.loc[ids,'condition'].to_numpy();al=f[alt].to_numpy();assert np.array_equal(f[control].to_numpy(),bl)
  if not set(al).issubset({'new','used'}) or np.any(al==bl):raise ValueError('Alternative orientation does not flip base15')
  pa=np.where(al=='used',p[ids-1],1-p[ids-1]);groups.append({'name':name,'ids':ids,'p':pa,'d':d})
  orientation.extend({'ID':int(i),'alternative_label_from_existing_audit':str(v),'control_label_in15':str(b)} for i,v,b in zip(ids,al,bl))
 validate_groups(groups,30000,21000);return groups,pd.DataFrame(orientation)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prior-file',type=Path,default=R/'results/training_round3/fields/test.npz');ap.add_argument('--self-test-only',action='store_true');args=ap.parse_args();O.mkdir(parents=True,exist_ok=True);checks=tests();(O/'validation.json').write_text(json.dumps(checks,indent=2))
 if args.self_test_only:print(json.dumps(checks,indent=2));return
 gs,orientation=load_groups(args.prior_file);post,report=decode(gs,30000,21000);post=post.merge(orientation,on='ID',validate='one_to_one');assert len(post)==160;post.to_csv(O/'posterior_probabilities.csv',index=False)
 report.update({'prior_file':str(args.prior_file.relative_to(R)),'prior_file_sha256':hashlib.sha256(args.prior_file.read_bytes()).hexdigest(),'validation':checks,'max_hidden_conditional_prior_difference':float(post.hidden_conditional_minus_prior.abs().max()),'assumptions':['Y_i independent Bernoulli with fixed supplied calibrated-or-not model probabilities; independence is assumed, not established.','Public mask uniformly distributed over all subsets of exactlyK21000 out ofN30000, independent of Y before conditioning.','Groups disjoint, orientations verified against submission15; balances treated as exact observations under previous score/truncation assumptions.'],'interpretation':'Exact only within this independent-prior model. Marginal probabilities and mask membership update, but correctness CONDITIONAL ON HIDDEN remains its prior exactly. No hidden transfer, individual labels, active submission or expected score produced.','future_relaxation':'A shared latent reliability/calibration parameter could transfer evidence to hidden items, but would require a justified prior, calibration validation and sensitivity analysis. Not implemented.'});(O/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({'U':report['U'],'max_error_toy':checks['max_absolute_posterior_error_vs_bruteforce'],'max_hidden_identity_error':report['max_hidden_conditional_prior_difference'],'log_evidence':report['log_probability_of_observed_constraints']},indent=2))
if __name__=='__main__':main()
