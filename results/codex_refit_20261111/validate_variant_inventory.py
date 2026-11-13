"""Train a challenger using nested variation inventory and attribute consistency."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from confirm_structural_models import split as earlier_split
from validate_recent_seller_history import split as warm_split
from validate_structural_disjoint import (
    METADATA, PARAMS, PICTURES, StratifiedGroupKFold, ag, causal_features, load, sha,
)
from evaluate_structural_blends import score


def nested(row):
    variants = row.get("variations") or []
    if not variants:
        return np.full(13, np.nan)
    values = [[float(v.get(key) or 0) for v in variants]
              for key in ("available_quantity", "sold_quantity", "price")]
    avail, sold, price = np.log1p(np.asarray(values).clip(0))
    pics = [set(v.get("picture_ids") or []) for v in variants]
    dims = [[(str(x.get("id") or x.get("name")), str(x.get("value_name") or x.get("value_id")))
             for x in v.get("attribute_combinations") or []] for v in variants]
    return [np.ptp(avail), np.ptp(sold), np.ptp(price), np.min(price),
            np.log1p(sum(values[0]))-np.log1p(max(0., float(row.get("available_quantity") or 0))),
            np.log1p(sum(values[1]))-np.log1p(max(0., float(row.get("sold_quantity") or 0))),
            np.mean(avail == 0), np.mean(sold == 0), len(set.union(*pics)), len(set.intersection(*pics)),
            len({k for v in dims for k, _ in v}), len({p for v in dims for p in v}),
            np.mean([bool(v.get("seller_custom_field")) for v in variants])]


def ids(paths):
    out = set()
    for path in paths:
        key = "id" if "id" in pd.read_csv(path, nrows=0) else "item"
        out.update(pd.read_csv(path, usecols=[key], dtype=str)[key])
    return out


def run(df, extra, features, source, paths, warm, output="variant_inventory"):
    manifest = json.loads((source / "report.json").read_text())
    prior = manifest["prior_pairs" if warm else "historical_panels"]
    assert [sha(p) for p in paths] == [x["sha256"] for x in prior]
    if warm:
        fit, val, seen, fresh = warm_split(df, ids(paths))
    else:
        fit, val, seen = earlier_split(df, ids(paths))
        fresh = np.ones(len(val), bool)
    digest = hashlib.sha256(np.r_[fit, val].tobytes()).hexdigest()
    assert digest == manifest["split_sha256"]
    pairs_path = source / "pairs_private.csv"
    assert sha(pairs_path) == manifest["pairs_sha256"]
    pairs = pd.read_csv(pairs_path, dtype={"id": str})
    assert np.array_equal(pairs.id, df.id.to_numpy()[val])
    assert np.array_equal(pairs.y, df.y.to_numpy()[val])
    assert np.array_equal(pairs.seen, seen)
    if callable(features):
        features = features(df, fit, val)
    tr = ag.precompute_item_intrinsic_arrays(df)
    ag.compute_causal_seller_features = causal_features([tr["start_time"][val]])
    inner = list(StratifiedGroupKFold(4, shuffle=True, random_state=61111).split(
        fit, tr["y"][fit], tr["seller_id"][fit]))
    assert all(not set(tr["seller_id"][fit[a]]) & set(tr["seller_id"][fit[b]]) for a,b in inner)
    cache = source / f"text_{digest[:16]}.npz"
    assert cache.exists()
    m = ag.build_fold_matrices(fit, [val], inner, tr, [tr], cache)
    x = np.column_stack([m["X_m1_seen_causal_T"], extra[fit], features[fit]])
    xv = np.column_stack([m["X_m1_seen_causal_evals"][0], extra[val], features[val]])
    model = LGBMClassifier(**PARAMS).fit(x, m["y_T"])
    base = pairs["attributes" if warm else "lgb"].to_numpy()
    supported = np.isfinite(features[val, 0])
    pred = np.where(supported, model.predict_proba(xv)[:,1], base)
    y, s, c = pairs.y.to_numpy(), tr["seller_id"][val], tr["category_id"][val]
    cohorts = {"all": np.ones(len(y), bool), "variations": supported, "fresh": fresh,
               "fresh_variations": fresh & supported, "fresh_seen": fresh & seen}
    out = Path(output) / source.name
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "model.txt"
    model.booster_.save_model(str(checkpoint))
    pp = out / "pairs_private.csv"
    pairs.assign(p_variant_inventory=pred, supported=supported, fresh=fresh).to_csv(pp,index=False)
    result = {"split_sha256": digest, "baseline_pairs_sha256": sha(pairs_path),
              "pairs_sha256": sha(pp), "model_sha256": sha(checkpoint),
              "disagreement_ids_sha256": hashlib.sha256("\n".join(
                  sorted(pairs.id[(pred >= .5) != (base >= .5)])).encode()).hexdigest(),
              "metrics": {k: score(y[g],pred[g],base[g],s[g],c[g],iterations=2500)
                          for k,g in cohorts.items()}}
    print(source, {k:(r["n"],r["net"],r["seller_category_q05_q95"][0])
                   for k,r in result["metrics"].items()}, flush=True)
    return result


def main():
    archive = Path(sys.argv[1])
    paths = list(map(Path,sys.argv[5:]))
    assert len(paths) == 8
    sources = (Path("recent_seller_history"),Path("structural_heldout_confirm"))
    assert all(sha(archive) == json.loads((s / "report.json").read_text())["archive_sha256"] for s in sources)
    with zipfile.ZipFile(archive) as z:
        df, extra = load(z,"train")
        with z.open("train_data.jsonlines") as f:
            features = np.asarray([nested(row) for row in map(json.loads,f)],dtype=np.float32)
    assert len(df) == len(features) == 70000
    extra = extra[:,sorted(METADATA + PICTURES)]
    out = {"scope": "Two archived development panels. The warm exclusive cohort excludes the earlier panel IDs. "
                    "Threshold fixed at .5, routed only for existing variants; outside support reuses the baseline.",
           "archive_sha256": sha(archive), "runner_sha256": sha(Path(__file__)),
           "panels": {s.name: run(df,extra,features,s,paths if i == 0 else paths[:-1],i == 0)
                      for i,s in enumerate(sources)}}
    (Path("variant_inventory") / "report.json").write_text(json.dumps(out,indent=2)+"\n")


if __name__ == "__main__":
    main()
