"""Audit AG0065 pairs against original train; quantify global-OOF contamination."""
import json,zipfile,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
with zipfile.ZipFile("ap_data_recovered/ap-2026-q-2 (1).zip") as z:
    raw=[json.loads(l) for l in z.open("train_data.jsonlines")]
s=np.asarray([str(r["seller_id"]) for r in raw]); ids=np.asarray([str(r["id"]) for r in raw]); y=np.asarray([int(r["condition"]=="used") for r in raw])
d=pd.read_csv("ag0065_pairs.csv",dtype={"seller_id":str}); byid={v:i for i,v in enumerate(ids)}
idx=np.asarray([byid[str(v)] for v in d.item_id]); assert np.array_equal(y[idx],d.y_true);assert np.array_equal(s[idx],d.seller_id)
rng=np.random.RandomState(20261144); u=np.unique(s);rng.shuffle(u); ns=len(u)
seen_pool=set(u[:int(.7*ns)]); seen_ix=np.flatnonzero(np.isin(s,list(seen_pool)));new_ix=np.flatnonzero(~np.isin(s,list(seen_pool)))
rng.shuffle(seen_ix);rng.shuffle(new_ix)
cs=int(.716*12264);cn=12264-cs;ts=int(.716*12442);tn=12442-ts
co=np.sort(np.r_[seen_ix[:cs],new_ix[:cn]]);tu=np.sort(np.r_[seen_ix[cs:cs+ts],new_ix[cn:cn+tn]]);fi=np.sort(seen_ix[cs+ts:])
assert np.array_equal(ids[co],d.item_id.astype(str))
seen=np.isin(s[co],s[fi]);assert np.array_equal(seen,d.is_seen_seller)
contam=[]
for f,(tr,va) in enumerate(GroupKFold(4).split(np.arange(len(y)),y,s)):
    contam.append({"text_fold":f,"outer_confirm_labels_used_in_text_train":int(np.isin(co,tr).sum()),
      "outer_tune_labels_used_in_text_train":int(np.isin(tu,tr).sum()),
      "outer_fit_rows_receiving_this_text_model":int(np.isin(fi,va).sum()),
      "outer_confirm_rows_receiving_this_text_model":int(np.isin(co,va).sum())})
out={"source_pairs_sha256":hashlib.sha256(Path("ag0065_pairs.csv").read_bytes()).hexdigest(),
 "rows":len(d),"raw_id_label_seller_mismatches":0,"reproduced_split":{"fit":len(fi),"tune":len(tu),"confirm":len(co),"excluded":len(y)-len(fi)-len(tu)-len(co)},
 "seen_rows":int(seen.sum()),"seen_fraction":float(seen.mean()),"global_oof_dependency":contam,"comparisons":{}}
base=(d.p_baseline_cb.to_numpy()>=.5); truth=d.y_true.to_numpy()
for name in ["p_full_cb","p_full_lgb","p_blend80","p_blend90"]:
    pred=d[name].to_numpy()>=.5;delta=(pred==truth).astype(int)-(base==truth).astype(int)
    by=pd.Series(delta).groupby(d.seller_id.to_numpy()).sum().to_numpy();r=np.random.default_rng(104)
    boot=np.asarray([by[r.integers(0,len(by),len(by))].sum() for _ in range(3000)])
    out["comparisons"][name]={"base_hits":int((base==truth).sum()),"hits":int((pred==truth).sum()),"fixes":int((delta==1).sum()),"breaks":int((delta==-1).sum()),"net":int(delta.sum()),"ci95":np.quantile(boot,[.025,.975]).tolist(),
      "seen_net":int(delta[seen].sum()),"new_net":int(delta[~seen].sum())}
Path("ag0065_dependency_audit.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
