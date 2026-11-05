"""Fixed-policy transport audit. All cohorts are development, overlap explicit."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
def contrast(d,a,b):
    y=d.y_true.to_numpy();pa=d[a[0]].to_numpy()>=a[1];pb=d[b[0]].to_numpy()>=b[1]
    delta=(pb==y).astype(int)-(pa==y).astype(int)
    by=pd.Series(delta).groupby(d.seller_id.to_numpy()).sum().to_numpy()
    rng=np.random.default_rng(104)
    boot=np.array([by[rng.integers(0,len(by),len(by))].sum() for _ in range(5000)])
    seen=d.is_seen.to_numpy().astype(bool)
    return {"base_hits":int((pa==y).sum()),"candidate_hits":int((pb==y).sum()),
      "fixes":int((delta==1).sum()),"breaks":int((delta==-1).sum()),"net":int(delta.sum()),
      "seller_ci95":np.quantile(boot,[.025,.975]).tolist(),
      "seen_net":int(delta[seen].sum()),"new_net":int(delta[~seen].sum())}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("output");ap.add_argument("directories",nargs="+");args=ap.parse_args()
    out={"limitation":"Predeclared policies evaluated on development splits; repeated item overlap documented, no public-score projection.",
         "cuts":{},"pairwise_overlap":[]}
    sets={}
    for path in args.directories:
        p=Path(path);d=pd.read_csv(p/"pairs.csv",dtype={"item_id":str,"seller_id":str})
        r=json.loads((p/"report.json").read_text())
        assert d.item_id.is_unique and set(d.y_true.unique())<={0,1}
        sets[path]=set(d.item_id)
        out["cuts"][path]={"seed":r["seed"],"n":len(d),"pure":r["pure"],"n_seen":int(d.is_seen.sum()),
          "pair_sha256":hashlib.sha256((p/"pairs.csv").read_bytes()).hexdigest(),
          "native_nb_common50":contrast(d,("p_cb_nested_nb",.5),("p_blend_native_nb",.5)),
          "native_nb_frozen_thresholds":contrast(d,("p_cb_nested_nb",.53),("p_blend_native_nb",.51)),
          "cb_lgb80_common50":contrast(d,("p_cb_nested_nb",.5),("p_blend_cb_lgb80",.5))}
    names=list(sets)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            n=len(sets[a]&sets[b]);out["pairwise_overlap"].append({"a":a,"b":b,"n_common":n,
               "fraction_a":n/len(sets[a]),"fraction_b":n/len(sets[b])})
    Path(args.output).write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
if __name__=="__main__":main()
