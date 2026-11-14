"""Build and cache B10's enrich() features + raw extras for a (train,test) pair of JSONL files.
usage: python feats.py DATA_DIR OUT_PKL"""
import sys,os,time
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
sys.path.insert(0,"/home/claude/wt_main/src")
import numpy as np,pandas as pd
from pathlib import Path
from optimize_leaderboard import enrich
d=Path(sys.argv[1]);t0=time.time()
raw=pd.read_json(d/"train_data.jsonlines",lines=True)
y=(raw.condition=="used").astype("int8").to_numpy()
groups=raw.parent_item_id.astype("string").fillna("__ROW__"+raw.id.astype(str)).to_numpy()
def extras(r):
    e=pd.DataFrame(index=range(len(r)))
    e["seller_id"]=r.seller_id.astype(str).values
    e["title_raw"]=r.title.fillna("").astype(str).values
    e["n_pictures"]=r.pictures.map(lambda z:len(z) if isinstance(z,list) else 0).values
    e["n_variations"]=r.variations.map(lambda z:len(z) if isinstance(z,list) else 0).values
    e["n_attributes"]=r.attributes.map(lambda z:len(z) if isinstance(z,list) else 0).values
    e["n_nonmp"]=r.non_mercado_pago_payment_methods.map(lambda z:len(z) if isinstance(z,list) else 0).values
    e["nonmp_ids"]=r.non_mercado_pago_payment_methods.map(lambda z:" ".join(sorted(m.get("id","") for m in z)) if isinstance(z,list) else "").values
    return e
xa,ids=enrich(raw);ea=extras(raw);del raw
print("train enrich",time.time()-t0,xa.shape,flush=True)
test=pd.read_json(d/"test_data.jsonlines",lines=True)
xb,_=enrich(test,ids);eb=extras(test);del test
print("test enrich",time.time()-t0,xb.shape,flush=True)
pd.to_pickle(dict(xa=xa,xb=xb,ea=ea,eb=eb,y=y,groups=groups,attr_ids=ids),sys.argv[2])
print("done",time.time()-t0,flush=True)
