import subprocess, io, json, numpy as np, pandas as pd, hashlib
V=list(np.load('/home/claude/work/r145/vectors.npy')); meta=[m for m in json.load(open('/home/claude/work/r145/vectors_meta.json')) if m[2]=='OK']
for n,sc in [(4,0.94766),(1,0.94542),(2,0.94757),(3,0.94752)]:
    b=subprocess.run(['git','show','origin/antigravity/overnight-recovery-coordination-20261030:submissions/SUBMISSION_%d_TODAY_RECOMMENDED.csv'%n],capture_output=True).stdout.replace(b'\r\n',b'\n')
    d=pd.read_csv(io.BytesIO(b)); assert d.ID.tolist()==list(range(1,30001))
    V.append((d.condition.values=='used').astype(np.int8)); meta.append(['TODAY_S%d'%n,'SUBMISSION_%d_TODAY'%n,'OK',sc,round(sc*20998)]); print(n,hashlib.sha256(b).hexdigest()[:12])
V=np.array(V); np.save('/home/claude/work/r145/V.npy',V); json.dump(meta,open('/home/claude/work/r145/V_meta.json','w'))
s4=V[-4]; print('dup check S4 vs EVT-143 diffs',[ (m[0],int((V[i]!=s4).sum())) for i,m in enumerate(meta) if m[0] in ('EVT-143','EVT-152')])
print(len(V))
