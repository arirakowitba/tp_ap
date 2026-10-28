import json, sympy, time, numpy as np
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
A=json.load(open("atoms.json")); K={int(k):v for k,v in A["K"].items()}
subs=sorted(n for n in K if n!=48)
atoms=[(sorted(k),v) for k,v in A["atoms"]]; J=len(atoms)
Am=sympy.Matrix([[1 if s in k else 0 for k,_ in atoms] for s in subs])
t=time.time(); R,piv=Am.rref(); print("rank",len(piv),time.time()-t)
free=[j for j in range(J) if j not in piv]
V=[]
for f in free:
    v=[sympy.Integer(0)]*J; v[f]=sympy.Integer(1)
    for r,p in enumerate(piv): v[p]=-R[r,f]
    den=sympy.ilcm(*[x.q for x in v]); vi=np.array([int(x*den) for x in v]); vi//=np.gcd.reduce(np.abs(vi)); V.append(vi)
V=np.array(V); print(V.shape,"max",np.abs(V).max(),"mean nnz",(V!=0).sum(1).mean())
np.save("nullbasis.npy",V)
