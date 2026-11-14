import numpy as np, json, sys
from estimator5 import V, h, i4, s4, fit, predict
others=[i for i in range(len(V)) if i!=i4]
bf=np.load('/home/claude/tp_ap/results/round146_claude_fixed_model/pred_b_fix_noso_b.npz')['te']
P={'b10':np.load('p_b10.npy'),'havg':np.load('p_havg.npy'),'havg_plus':np.load('p_havg_plus.npy'),'blend_fix':np.load('p_blend_fix.npy'),'b_fix':bf,
   'fin_cat5':np.load('/home/claude/data/pred_fin_cat5.npz')['te'],'mean_all':None}
P['mean_all']=np.mean([P[k] for k in ['b10','havg_plus','blend_fix','b_fix']],0)
res=[]
for nm,p in P.items():
    q=np.clip(p,1e-4,1-1e-4); lq=np.log(q/(1-q))
    for B in ([0] if nm=='b10' else [])+[.1,.25,.5,.75,1.0]:
        pr=1/(1+np.exp(-B*lq)); err=[]; zz=[]
        for k in others:
            F=fit(pr,[i for i in others if i!=k]); e,sd=predict(F,V[k]); err.append((h[k]-h[i4])-e); zz.append(err[-1]/sd)
        err=np.array(err); res.append((nm,B,float(np.mean(np.abs(err))),float(np.sqrt(np.mean(err**2))),float(np.std(zz))))
        print('%-10s b=%.2f  LOO MAE %.2f RMSE %.2f zsd %.2f'%res[-1],flush=True)
json.dump(res,open('prior_grid.json','w'))
