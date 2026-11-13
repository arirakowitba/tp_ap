"""Compare one frozen equal blend on disjoint archived development cohorts."""
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

from evaluate_structural_blends import score, sha


def main():
    base = Path("late_native_categories")
    panels = [pd.read_csv(base/f"{name}_pairs.csv", dtype={"id":str}) for name in ("cold","warm")]
    panels[1] = panels[1][panels[1].fresh]
    assert not set(panels[0].id) & set(panels[1].id)
    ids = set().union(*(set(p.id) for p in panels))
    with zipfile.ZipFile(sys.argv[1]) as z,z.open("train_data.jsonlines") as f:
        lookup = {str(r["id"]):(str(r["seller_id"]),str(r["category_id"]))
                  for r in map(json.loads,f) if str(r["id"]) in ids}
    report={"scope":"Fixed equal-weight blend on two reused, ID-disjoint validation cohorts.",
            "metrics":{},"input_sha256":{name:sha(base/f"{name}_pairs.csv") for name in ("cold","warm")}}
    for name,d,key in zip(("cold","warm_exclusive"),panels,("lgb","attributes")):
        seller,category=np.array([lookup[i] for i in d.id]).T
        b=d[key].to_numpy()
        p=.5*(d.p_native.to_numpy()+b)
        y=d.y.to_numpy()
        seen=d.seen.to_numpy(dtype=bool)
        report["metrics"][name]={key:score(y[m],p[m],b[m],seller[m],category[m],2500)
                                  for key,m in (("all",np.ones(len(y),bool)),("seen",seen),("new",~seen))}
    (base/"blend_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report["metrics"],indent=2))


if __name__=="__main__":
    main()
