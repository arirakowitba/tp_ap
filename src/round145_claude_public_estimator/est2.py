import numpy as np, sys
from estimator import *
others=[i for i in range(len(V)) if i!=i4]
q=np.clip(np.load('p_havg_plus.npy'),1e-4,1-1e-4); B=float(sys.argv[1]) if len(sys.argv)>1 else .25
P=1/(1+np.exp(-B*np.log(q/(1-q)))); F=fit(P,others)
def est(v): e,sd=predict(F,v); return e, sd*2.7
def post_mean():
    mu=F['mu'].copy(); c=F['cols']; mu[c]+= F['s2'][c]*(F['Dc'].T@(F['Sinv']@F['r'])); return mu
