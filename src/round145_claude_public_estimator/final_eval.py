import numpy as np, pandas as pd, hashlib, json, os
from estimator import V, h, i4, s4, fit, predict
others=[i for i in range(len(V)) if i!=i4]
b10=np.load('/home/claude/work/r145/p_b10.npy'); bf=np.load('/home/claude/work/r146/pred_b_fix_noso_b.npz')['te']; bl=.5*b10+.5*bf
np.save('/home/claude/work/r145/p_blend_fix.npy',bl)
out='/home/claude/tp_ap/submissions/claude_20261114'; man=json.load(open(f'{out}/manifest.json'))
def write(v,name,e,sd,extra):
    txt='ID,condition\n'+''.join(f'{i+1},{"used" if v[i] else "new"}\n' for i in range(30000)); open(f'{out}/{name}','w').write(txt)
    man['candidates'][name]=dict(flips=int((v!=s4).sum()),est_delta_hits=round(float(e),1),sd_inflated=round(float(sd),1),est_public=round((h[i4]+e)/20998,5),sha256=hashlib.sha256(txt.encode()).hexdigest(),**extra)
    print('%-48s flips %4d est %+6.1f ± %5.1f -> %.5f'%(name,(v!=s4).sum(),e,sd,(h[i4]+e)/20998))
for qn,q0 in [('havg_plus',np.load('/home/claude/work/r145/p_havg_plus.npy')),('blend_fix',bl)]:
    q=np.clip(q0,1e-4,1-1e-4)
    for B in [.25,.5]:
        F=fit(1/(1+np.exp(-B*np.log(q/(1-q)))),others)
        e,sd=predict(F,(bl>=.5).astype(float)); print(f'prior {qn} b={B}: pure blend_fix vs S4 est {e:+.1f} ± {2.7*sd:.1f}')
        if qn=='blend_fix':
            mu=F['mu'].copy(); c=F['cols']; mu[c]+=F['s2'][c]*(F['Dc'].T@(F['Sinv']@F['r'])); g=np.where(s4==1,-mu,mu); o=np.argsort(-g)
            for K in [30,int((g>0).sum())]:
                v=s4.copy(); sel=o[:K]; v[sel]=1-v[sel]; e,sd=predict(F,v); 
                F2=fit(1/(1+np.exp(-.25*np.log(np.clip(np.load('/home/claude/work/r145/p_havg_plus.npy'),1e-4,1-1e-4)/(1-np.clip(np.load('/home/claude/work/r145/p_havg_plus.npy'),1e-4,1-1e-4))))),others); e2,sd2=predict(F2,v)
                print(f'   opt prior blend b={B} K={K}: est {e:+.1f} ± {2.7*sd:.1f} | under havg_plus prior {e2:+.1f} ± {2.7*sd2:.1f}')
                if B==.25: write(v,f'claude_blendfix_posterior_top{K}_sobre_S4.csv',e,2.7*sd,dict(prior='blend_fix b=0.25',est_under_havgplus_prior=round(float(e2),1),ids=[int(i+1) for i in sorted(sel)]))
vb=(bl>=.5).astype(float); write(vb,'claude_blendfix_pure_model.csv',*[x*(1 if j==0 else 2.7) for j,x in enumerate(predict(F,vb))],dict(note='0.5 B10 + 0.5 b_fix_noso_b, pure model'))
json.dump(man,open(f'{out}/manifest.json','w'),indent=1)
