"""Compare causal seller profile features against the structural LightGBM baseline."""
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

from validate_variant_inventory import METADATA, PICTURES, load, run, sha


def profiles(df, fit, val):
    query = np.r_[fit, val]
    out = np.full((len(df), 1), 0., dtype=np.float32)
    # A ratio within a listing is available when predicting that listing.
    log_price = np.log1p(df.price.clip(lower=0))
    quantity = np.log1p(df.init_qty.clip(lower=0))
    sold = np.log1p(df.sold_qty.clip(lower=0))
    features = [out]
    for keys in (["seller_id"], ["seller_id", "category_id"], ["seller_id", "brand_val"]):
        fields = [*keys, "start_time"]
        train = df.iloc[fit][fields+["y"]].assign(price=log_price.iloc[fit],
                price_sq=log_price.iloc[fit]**2, quantity=quantity.iloc[fit], sold=sold.iloc[fit],
                relist=df.automatic_relist.iloc[fit], warranty=df.warranty_str.iloc[fit].str.len().gt(0))
        summary = train.groupby(fields, as_index=False, sort=False).agg(
            n=("y","size"), y=("y","sum"), price=("price","sum"), price_sq=("price_sq","sum"),
            quantity=("quantity","sum"), sold=("sold","sum"), relist=("relist","sum"), warranty=("warranty","sum"))
        summary = summary.sort_values("start_time",kind="stable")
        columns = ["n","y","price","price_sq","quantity","sold","relist","warranty"]
        summary[columns] = summary.groupby(keys,sort=False)[columns].cumsum()
        target = df.iloc[query][fields].assign(pos=np.arange(len(query)))
        join = pd.merge_asof(target.sort_values("start_time",kind="stable"), summary, on="start_time",
                             by=keys,allow_exact_matches=False).sort_values("pos")
        n = join.n.to_numpy()
        mean = join.price.to_numpy()/n
        sd = np.sqrt(np.maximum(0,join.price_sq.to_numpy()/n - mean**2))
        current = log_price.to_numpy()[query]
        block=np.column_stack([np.log1p(np.nan_to_num(n)),(join.y.to_numpy()+2.5)/(n+5),
                current-mean,(current-mean)/np.maximum(sd,.1),sd,
                quantity.to_numpy()[query]-join.quantity.to_numpy()/n,
                sold.to_numpy()[query]-join.sold.to_numpy()/n,join.relist.to_numpy()/n,
                join.warranty.to_numpy()/n])
        full=np.full((len(df),block.shape[1]),np.nan,dtype=np.float32)
        full[query]=block
        features.append(full)
    return np.column_stack(features)


def main():
    archive=Path(sys.argv[1])
    paths=list(map(Path,sys.argv[5:]))
    assert len(paths)==8
    sources=(Path("recent_seller_history"),Path("structural_heldout_confirm"))
    assert all(sha(archive)==json.loads((s/"report.json").read_text())["archive_sha256"] for s in sources)
    with zipfile.ZipFile(archive) as z:
        df,extra=load(z,"train")
    assert len(df)==70000 and df.id.is_unique
    extra=extra[:,sorted(METADATA+PICTURES)]
    dest="late_seller_profiles"
    panels={}
    for i,s in enumerate(sources):
        r=run(df,extra,profiles,s,paths if i==0 else paths[:-1],i==0,dest)
        r["metrics"].pop("variations")
        r["metrics"].pop("fresh_variations")
        panels[s.name]=r
    report={"scope":"Reused train-only panels; exclusive subsets disjoint by ID; no public projection.",
            "runner_sha256":sha(Path(__file__)),"archive_sha256":sha(archive),"panels":panels}
    (Path(dest)/"report.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
