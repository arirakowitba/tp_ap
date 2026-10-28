import json, numpy as np
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
from scipy.optimize import milp, LinearConstraint, Bounds
A=json.load(open("atoms.json")); K={int(k):v for k,v in A["K"].items()}
subs=sorted(n for n in K if n!=48)
atoms=[(sorted(k),v) for k,v in A["atoms"]]; J=len(atoms); nv=2*J
cons=[];lo=[];hi=[]
for j,(k,v) in enumerate(atoms):
    r=np.zeros(nv); r[j]=1; r[J+j]=1; cons.append(r); lo.append(0); hi.append(len(v))
for n in subs:
    r=np.zeros(nv)
    for j,(k,v) in enumerate(atoms):
        if n in k: r[j]=1; r[J+j]=-1
    cons.append(r); lo.append(K[n]); hi.append(K[n])
C=LinearConstraint(np.array(cons),lo,hi); sz=np.array([len(v) for k,v in atoms]); B=Bounds(0,np.r_[sz,sz])
out=[]
for j in range(J):
    o=np.zeros(nv); o[j]=1; o[J+j]=-1
    mn=milp(o,constraints=C,integrality=np.ones(nv),bounds=B).fun
    mx=-milp(-o,constraints=C,integrality=np.ones(nv),bounds=B).fun
    out.append((j,atoms[j][0],len(atoms[j][1]),round(mn),round(mx)))
json.dump(out,open("bounds.json","w"))
pos=[o for o in out if o[3]>0]; print("atoms guaranteed positive:",pos)
print("atoms with max>0:",sum(o[4]>0 for o in out),"rows",sum(o[2] for o in out if o[4]>0))
det=[o for o in out if o[3]==o[4]]; print("fully determined atoms:",len(det), [(o[1],o[2],o[3]) for o in det][:40])
