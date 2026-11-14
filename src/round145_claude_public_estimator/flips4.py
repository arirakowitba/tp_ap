import numpy as np, json, pandas as pd
from estimator4 import V, h, i4, s4 as base, fit, predict, postmean, meta
n=[m[0] for m in meta]; others=[i for i in range(len(V)) if i!=i4]; S4=V[n.index('TODAY_S4')]
span=((V-base)!=0).any(0)
pri={}
for nm,f in [('havg_plus','p_havg_plus.npy'),('blend_fix','p_blend_fix.npy')]:
    q=np.clip(np.load(f),1e-4,1-1e-4)
    for B in [.25,.5]: pri[(nm,B)]=fit(1/(1+np.exp(-B*np.log(q/(1-q)))),others)
fr=20998/30000
PW=[]  # posterior P(base wrong) per row per prior
for F in pri.values():
    mu=postmean(F); py=np.clip(.5+mu/(2*fr),0,1)  # P(y=used)
    PW.append(np.where(base==1,1-py,py))
PW=np.array(PW); pmin=PW.min(0)
ins=span&(pmin>.5)
print('in-span rows with P(base wrong)>0.5 under all 4 priors:',int(ins.sum()))
for t in [.55,.6,.65,.7]: print('  >%.2f: %d'%(t,int((span&(pmin>t)).sum())))
np.save('pmin_inspan.npy',np.where(span,pmin,0)); np.save('span4.npy',span)
# strict unprobed under the larger span
e1010=(pd.read_csv('/home/claude/wt_main/submissions/submission_optimized_ensemble.csv').condition.values=='used')
b10=np.load('p_b10.npy')>=.5; bl=np.load('p_blend_fix.npy'); bf=np.load('/home/claude/tp_ap/results/round146_claude_fixed_model/pred_b_fix_noso_b.npz')['te']
strict=~span&(base==e1010)&(base==b10)
U=strict&((bl>=.5)!=base); Fm=strict&((bf>=.5)!=base)&(np.abs(bf-.5)>=.05)&~U
U36=np.load('U36.npy'); Fs=np.load('F_strict.npy')
print('strict rows',int(strict.sum()),'U now',int(U.sum()),'(U36 kept',int((U&U36).sum()),') F now',int(Fm.sum()),'(F86 kept',int((Fm&Fs).sum()),')')
np.save('U4.npy',U); np.save('F4.npy',Fm)
