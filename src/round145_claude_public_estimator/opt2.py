import numpy as np, pandas as pd, json, hashlib, os
from estimator2 import V, h, i4, s4, fit, predict
others=[i for i in range(len(V)) if i!=i4]; sub='/home/claude/tp_ap/submissions/claude_20261114/'
span=((V-s4)!=0).any(0)
t30=(pd.read_csv(sub+'claude_blendfix_posterior_top30_sobre_S4.csv').condition.values=='used').astype(float)
priors={'havg_plus':np.load('p_havg_plus.npy'),'blend_fix':np.load('p_blend_fix.npy')}
Fs={}
for nm,q in priors.items():
    q=np.clip(q,1e-4,1-1e-4)
    for B in [.25,.5]: Fs[(nm,B)]=fit(1/(1+np.exp(-B*np.log(q/(1-q)))),others)
def gains(F):
    mu=F['mu'].copy(); c=F['cols']; mu[c]+=F['s2'][c]*(F['Dc'].T@(F['Sinv']@F['r'])); return np.where(s4==1,-mu,mu)
def evalall(v): return {f'{k[0]}_b{k[1]}':tuple(round(x,1) for x in (predict(F,v)[0],2.7*predict(F,v)[1])) for k,F in Fs.items()}
G=gains(Fs[('blend_fix',.25)]); Gh=gains(Fs[('havg_plus',.25)])
# consensus ranking: require positive gain under both priors and in-span
ok=(G>0)&(Gh>0)&span; score=np.minimum(G,Gh); order=np.argsort(-np.where(ok,score,-1e9))
print('rows positive under both priors & in span:',int(ok.sum()))
res={}
for K in [10,20,30,50]:
    if K>ok.sum(): break
    v=s4.copy(); sel=order[:K]; v[sel]=1-v[sel]
    e=evalall(v); res[K]=(v,sel,e); print('K=%d'%K, e, 'overlap with top30', int(((v!=s4)&(t30!=s4)).sum()))
print('top30 (old) now:', evalall(t30))
np.save('opt2_order.npy',order); np.save('opt2_ok.npy',ok)
import pickle; pickle.dump({K:(r[1],r[2]) for K,r in res.items()},open('opt2_res.pkl','wb'))
