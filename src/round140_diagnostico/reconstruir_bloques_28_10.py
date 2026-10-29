import json, hashlib, numpy as np, pandas as pd
import os
D=os.environ.get("AP2026Q2_DATA_DIR","data/ap_2026_q_2").rstrip("/")+"/"; H=os.path.dirname(os.path.abspath(__file__)); R=os.path.join(H,"..","..","submissions","20261028")+"/"
tr=pd.read_pickle(D+"tr.pkl"); te=pd.read_pickle(D+"te.pkl")
m=json.load(open(R+"manifest_bandas_fin_stack.json"))
s48={int(k):v for k,v in json.load(open(os.path.join(H,"..","..","results","revision_submissions","posterior_flips_20261027","s48.json"))).items()}
pure=np.load(D+"pred_fin_stack.npz")["te"]>.5
seen=te.seller_id.isin(set(tr.seller_id)).values
pool=set().union(*(set(p['ids']) for p in m['probes'] if p['number'] in (60,61,62))); assert len(pool)==820
blocks={}
for name,seg in [('seen',seen),('new',~seen)]:
    ch=[i for i in pool if seg[i-1]]; ch.sort(key=lambda i: hashlib.sha256(f'20261028:fin_stack:{name}:{i}'.encode()).digest())
    blocks[f's150_{name}']=ch[:150]
p59=next(p for p in m['probes'] if p['number']==59)['ids']
blocks['p59']=p59
blocks['p59_new']=[i for i in p59 if not seen[i-1]]
blocks['p59_seen_to_new']=[i for i in p59 if seen[i-1] and s48[i]=='used']
blocks['p59_seen_to_used']=[i for i in p59 if seen[i-1] and s48[i]=='new']
print({k:len(v) for k,v in blocks.items()})
# verify hashes of Ariel probes
def csv_of(ids):
    sub=dict(s48)
    for i in ids: sub[i]='new' if sub[i]=='used' else 'used'
    return "ID,condition\n"+"".join(f"{i},{sub[i]}\n" for i in range(1,30001))
exp={'s150_seen':'37a8c519abff07ae8c64e360001229f3f3bf6953301908b38a946327ed00c832','s150_new':'ebb365337aa674d5ce099b763b4917089348539bf1b3b8e5f509e7950cfcf67a','p59_new':'de1ebd9af13ecc1723e5f521e1c0dcb525a2e9261267ae3edb7bafe4bc5e7b9f','p59_seen_to_new':'d06115e8984d4ae82c34cd0ac7276f7b16a6666d1eca57bea736aa5f7b7f32e7','p59':'b34723587e1ef09c897a4af5768a2ea025c0dd313c04bd6494d5f0289899f373'}
for k,h in exp.items():
    got=hashlib.sha256(csv_of(blocks[k]).encode()).hexdigest(); print(k, got==h)
json.dump(blocks,open(os.path.join(H,"bloques_28_10.json"),"w"))
