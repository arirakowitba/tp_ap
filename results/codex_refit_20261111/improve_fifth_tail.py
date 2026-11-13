"""Compare whole-candidate tails exactly under named transfer assumptions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, hypergeom
from scipy.signal import fftconvolve
from scipy.special import ndtr

from select_fifth_history_v2 import (
    add_model, get_probability, groups, calibration, bounds_for, anchor, old, today, sha,
)

POP, PUB, NEED = 30000, 20998, 50


def exact(counts, rates):
    dist = np.array([1.])
    for n,p in zip(counts,rates):
        if n:
            dist = np.convolve(dist,binom.pmf(np.arange(n+1),n,p))
    k=int(sum(counts))
    pub=np.arange(NEED,k+1)
    need=(pub+NEED+1)//2
    h=hypergeom.sf(need[None,:]-1,k,np.arange(k+1)[:,None],pub[None,:])
    return float(dist @ (h @ hypergeom.pmf(pub,POP,k,PUB)))


def approx(n,p):
    mu=(n*(2*p-1)).sum()*PUB/POP
    sd=np.sqrt(max(1,(n*(1-((2*p-1)*PUB/POP)**2)*PUB/POP).sum()))
    return float(ndtr((mu-49.5)/sd))


def main():
    ext=Path('/workspace/scratch/5f1fcd4a32e7')
    history=json.loads(Path('ap_history_nov13/constraints_private.json').read_text())
    tested={frozenset(ids) for _,ids,_ in history}
    t,c,w=(add_model(key,ext) for key in ('full','cold','warm'))
    assert not set(c.id)&set(w.loc[w.fresh,'id'])
    a=anchor.eq('used').to_numpy()
    assert np.array_equal(t.ID,anchor.index)
    seen=t.seen.to_numpy(dtype=bool)
    proxy=t.proxy.ge(.5).to_numpy()==a
    retired=np.zeros(POP,bool)
    touched=np.zeros(POP,bool)
    for j,(_,ids,_) in enumerate(history):
        touched[np.array(ids)-1]=True
        if j<9:
            retired[np.array(ids)-1]=True
    paths=[today/'ap_20261113_05_final.csv',today/'ap_20261113_05_consenso_cruzado.csv',
           today/'ap_20261113_R_consenso_fuera_proxy.csv']
    extras=[]
    for path in paths:
        df=pd.read_csv(path)
        assert df.ID.equals(t.ID)
        extras.append(df.condition.eq('used').to_numpy()!=a)
    w=w.loc[w.fresh].copy()
    rows=[]
    visited=set()
    for weight in ((0,1,0),(1,0,0),(.5,.5,0),(.33,.34,.33)):
        p=get_probability(t,weight)
        for st in (.35,.40,.45,.50,.55,.60,.65):
            for nt in (.35,.40,.45,.50,.55,.60,.65):
                threshold=(st,nt)
                base=p>=np.where(seen,st,nt)
                rates=[calibration(d,weight,threshold)[0] for d in (c,w)]
                for mode in ('broad','virgin','add_previous'):
                    m=(base!=a)&~retired
                    if mode=='virgin':
                        m &= ~touched
                    elif mode=='add_previous':
                        m |= extras[0]
                    ids=frozenset(t.ID[m])
                    if not 50<=len(ids)<=550 or ids in visited or ids in tested:
                        continue
                    visited.add(ids)
                    g=groups(p,m,~a,seen)
                    # A change already in the previous candidate can be an
                    # override of this model. It belongs to the scenario-only group.
                    informed=proxy[m] & (base[m]!=a[m])
                    n=np.r_[np.bincount(g[informed],minlength=8),int((~informed).sum())]
                    model_rates=[np.r_[r,.45] for r in rates]
                    rank=min(approx(n,r) for r in model_rates)
                    rows.append((rank,weight,threshold,mode,m,n,model_rates))
    rows.sort(key=lambda r:r[0],reverse=True)
    best=[]
    for row,bounds in zip(rows[:45],bounds_for((r[4] for r in rows[:45]),history)):
        if bounds[1]<NEED:
            continue
        _,weight,threshold,mode,m,n,rates=row
        scenarios={str(q):[exact(n,np.r_[r[:-1],q]) for r in rates] for q in (.40,.45,.50,.55,.60)}
        best.append((min(scenarios["0.45"]),row,bounds,scenarios))
    assert best
    value,row,bounds,scenarios=max(best,key=lambda x:x[0])
    _,weight,threshold,mode,m,n,rates=row
    output=Path('ap_20261113_05_ampliada.csv')
    pd.DataFrame({'ID':t.ID,'condition':np.where(a^m,'used','new')}).to_csv(output,index=False,lineterminator='\n')
    saved=pd.read_csv(output)
    assert saved.ID.equals(t.ID) and np.array_equal(saved.condition.eq('used'),a^m)
    # Compare incumbent using its original matched 30 schema changes and otherwise
    # the exact same independent, uniform-public calculation and penalty grid.
    incumbent=json.loads(Path('ap_20261113_05_final_report.json').read_text())['cv_proxy']['precision']
    prior_n=np.r_[30,71]
    comparison={str(q):[exact(prior_n,[r,q]) for r in incumbent] for q in (.40,.45,.50,.55,.60)}
    report={'scope':'Exploratory selection used both reused development panels. Exact probabilities are conditional on '
                    'independent correctness given fixed cell rates and a uniform public subset. No estimate of S4-relative '
                    'off-proxy accuracy is available. The off-proxy rate is varied explicitly. Ranks are not prospective.',
            'history_constraints':len(history),'policies':len(rows),'shortlisted':45,
            'weights':weight,'thresholds_seen_new':threshold,'mode':mode,'flips':int(m.sum()),
            'proxy_calibrated':int(n[:-1].sum()),'off_proxy':int(n[-1]),'history_touched':int((m&touched).sum()),
            'bounds':bounds,'conditional_tail':scenarios,'previous_conditional_tail':comparison,
            'candidate_sha256':sha(output),'replaces_previous':value>min(comparison['0.45'])}
    Path('ap_20261113_05_ampliada_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
