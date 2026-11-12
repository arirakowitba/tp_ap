"""Fixed publication-attribute challenger on IDs outside all three previous cuts."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from ablate_physical_history import METADATA, PICTURES, PARAMS
from replicate_structural_metadata import StratifiedGroupKFold, ag, causal_features, load, metrics, sha


def prior_ids(paths):
    return set().union(*(set(pd.read_csv(p, usecols=["id"], dtype=str).id) for p in paths))


def make_split(df, old):
    available = ~df.id.isin(old)
    rng = np.random.default_rng(28111206)
    keys = np.sort(df.loc[available, "seller_id"].unique())
    rng.shuffle(keys)
    cold = df.seller_id.isin(keys[:len(keys) // 4])
    # For a reproducible warm cohort, keep only training rows strictly before
    # the selected timestamp. Previously evaluated rows at or after it are omitted.
    pool = df.loc[available & ~cold, ["seller_id", "start_time"]]
    latest = pool.groupby("seller_id").start_time.max()
    first = df.groupby("seller_id").start_time.min()
    warm_keys = np.sort(latest.index[latest > first.reindex(latest.index)].to_numpy())
    rng.shuffle(warm_keys)
    limits = latest.reindex(warm_keys[:len(warm_keys) // 2])
    cutoff = df.seller_id.map(limits)
    warm = cutoff.notna() & (df.start_time >= cutoff)
    val = np.flatnonzero(available & (cold | (warm & (df.start_time == cutoff))))
    fit = np.flatnonzero(~(cold | warm))
    seen = ~cold.to_numpy()[val]
    assert not set(df.id.iloc[val]) & old
    assert not set(df.seller_id.iloc[val[~seen]]) & set(df.seller_id.iloc[fit])
    tmax = df.iloc[fit].groupby("seller_id").start_time.max()
    assert (df.seller_id.iloc[val[seen]].map(tmax) < df.start_time.iloc[val[seen]]).all()
    return fit, val, seen


def main():
    src = Path(sys.argv[1])
    old_dir = Path(sys.argv[2])
    external = Path(sys.argv[4])
    out = Path("structural_disjoint")
    out.mkdir(exist_ok=True)
    paths = [old_dir / "line3_today/pairs.csv", Path("structural_replica/pairs_private.csv"),
             external / "structural_replication.csv"]
    with zipfile.ZipFile(src) as archive:
        df, extras = load(archive, "train")
    assert len(df) == 70000 and df.id.is_unique
    old = prior_ids(paths)
    fit, val, seen = make_split(df, old)
    print("split", len(fit), len(val), int(seen.sum()), flush=True)
    tr = ag.precompute_item_intrinsic_arrays(df)
    ag.compute_causal_seller_features = causal_features([tr["start_time"][val]])
    inner = list(StratifiedGroupKFold(4, shuffle=True, random_state=61111).split(
        fit, tr["y"][fit], tr["seller_id"][fit]))
    assert all(not set(tr["seller_id"][fit[a]]) & set(tr["seller_id"][fit[b]]) for a, b in inner)
    digest = hashlib.sha256(np.r_[fit, val].tobytes()).hexdigest()
    cache = out / ("text_" + digest[:16] + ".npz")
    m = ag.build_fold_matrices(fit, [val], inner, tr, [tr], cache)
    print("text cache ready", flush=True)
    xf, xv = m["X_m1_seen_causal_T"], m["X_m1_seen_causal_evals"][0]
    extra = extras[:, sorted(METADATA + PICTURES)]
    pred = {}
    for key, a, b in (("control", xf, xv), ("attributes", np.column_stack([xf, extra[fit]]),
                                           np.column_stack([xv, extra[val]]))):
        model = LGBMClassifier(**PARAMS).fit(a, m["y_T"])
        pred[key] = model.predict_proba(b)[:, 1]
        model.booster_.save_model(str(out / f"{key}.txt"))
        print(key, "ready", flush=True)
    hp = {"num_leaves": 55, "colsample_bytree": .7, "min_child_samples": 40}
    for regime in ("unseen", "seen_causal"):
        key = f"X_m1_{regime}"
        pred[regime] = ag.fit_predict_lgb_bagged(
            m[key + "_T"], m["y_T"], m[key + "_evals"], hp, [42, 2026, 777], 900)[0]
        print(regime, "reference ready", flush=True)
    pred["line3"] = np.where(seen, pred["seen_causal"], pred["unseen"])
    y = tr["y"][val]
    frame = df.iloc[val][["id", "y"]].assign(seen=seen, p_control=pred["control"],
                                            p_attributes=pred["attributes"], p_line3=pred["line3"])
    pairpath = out / "pairs_private.csv"
    frame.to_csv(pairpath, index=False)
    report = {"fit_rows": len(fit), "validation_rows": len(val), "omitted_rows": len(df)-len(fit)-len(val),
              "past_validation_ids": len(old), "holdout_overlap": 0, "archive_sha256": sha(src),
              "split_sha256": digest, "runner_sha256": sha(Path(__file__)),
              "earlier_pairs_sha256": [sha(p) for p in paths], "pairs_sha256": sha(pairpath),
              "attributes_model_sha256": sha(out / "attributes.txt")}
    for base in ("control", "line3"):
        report[f"vs_{base}"] = {k: metrics(y[s], pred["attributes"][s], pred[base][s], tr["seller_id"][val][s])
                                 for k, s in (("all", np.ones(len(y), bool)), ("seen", seen), ("new", ~seen))}
        report[f"vs_{base}"]["disagreement_ids_sha256"] = hashlib.sha256(
            "\n".join(sorted(frame.id[(pred["attributes"] >= .5) != (pred[base] >= .5)])).encode()).hexdigest()
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
