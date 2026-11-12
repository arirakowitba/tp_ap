"""Paired train-only test of recent seller history on a warm-heavy cut."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from evaluate_structural_blends import score
from validate_structural_disjoint import (
    METADATA, PARAMS, PICTURES, StratifiedGroupKFold, ag, causal_features, load, sha,
)

DAY_MS = 86_400_000


def split(df, old):
    rng = np.random.default_rng(28111215)
    sellers = np.sort(df.seller_id.unique())
    rng.shuffle(sellers)
    cold = df.seller_id.isin(sellers[:len(sellers) // 8])
    ordered = df.loc[~cold].sort_values("start_time")
    times = ordered[["seller_id", "start_time"]].drop_duplicates()
    position = times.groupby("seller_id").cumcount()
    distinct = times.groupby("seller_id").start_time.transform("size")
    threshold = np.maximum(2, np.floor(.60 * distinct))
    cutoff = times.loc[position.eq(threshold)].set_index("seller_id").start_time
    warm = df.start_time.ge(df.seller_id.map(cutoff))
    fit = np.flatnonzero(~(cold | warm))
    val = np.flatnonzero(warm | cold)
    seen = warm.to_numpy()[val]
    fresh = ~df.id.iloc[val].isin(old).to_numpy()
    assert not set(df.seller_id.iloc[val[~seen]]) & set(df.seller_id.iloc[fit])
    latest = df.iloc[fit].groupby("seller_id").start_time.max()
    assert df.seller_id.iloc[val[seen]].map(latest).lt(df.start_time.iloc[val[seen]]).all()
    return fit, val, seen, fresh


def history(df, fit, query):
    columns = []
    recent_seller_mean = None
    for keys in (["seller_id"], ["seller_id", "category_id"]):
        fields = [*keys, "start_time"]
        table = (df.iloc[fit].groupby(fields, as_index=False, sort=False)
                 .agg(n=("y", "size"), used=("y", "sum")).sort_values("start_time"))
        values = ["n", "used"]
        table[values] = table.groupby(keys, sort=False)[values].cumsum()
        target = df.iloc[query][fields].assign(pos=np.arange(len(query)))

        def before(days):
            left = target.assign(start_time=target.start_time - days * DAY_MS).sort_values("start_time")
            joined = pd.merge_asof(left, table, on="start_time", by=keys, allow_exact_matches=False)
            return joined.sort_values("pos")[values].fillna(0).to_numpy()

        present = before(0)
        for window in (90, 365):
            n, used = (present - before(window)).T
            assert np.all((used >= 0) & (n >= used))
            prior = .5 if recent_seller_mean is None else recent_seller_mean
            p = (used + 5 * prior) / (n + 5)
            columns.extend([np.log1p(n), np.where(n > 0, p, np.nan)])
            if keys == ["seller_id"] and window == 365:
                seller_prior = p
        if keys == ["seller_id"]:
            recent_seller_mean = seller_prior
    return np.column_stack(columns).astype(np.float32)


def main():
    src = Path(sys.argv[1])
    prior_paths = list(map(Path, sys.argv[5:]))
    assert prior_paths
    old = set()
    for p in prior_paths:
        key = "id" if "id" in pd.read_csv(p, nrows=0) else "item"
        old.update(pd.read_csv(p, dtype=str, usecols=[key])[key])
    with zipfile.ZipFile(src) as z:
        df, extras = load(z, "train")
    assert len(df) == 70_000 and df.id.is_unique and old <= set(df.id)
    fit, val, seen, fresh = split(df, old)
    assert (fresh & seen).sum() and (fresh & ~seen).sum()
    fingerprint = hashlib.sha256(np.r_[fit, val].tobytes()).hexdigest()
    out = Path("recent_seller_history")
    out.mkdir(exist_ok=True)
    report = {
        "train_rows": len(fit), "validation_rows": len(val), "seen_rows": int(seen.sum()),
        "fresh_rows": int(fresh.sum()), "fresh_seen_rows": int((fresh & seen).sum()),
        "archive_sha256": sha(src), "split_sha256": fingerprint, "runner_sha256": sha(Path(__file__)),
        "prior_pairs": [{"file": p.parent.name + "/" + p.name, "sha256": sha(p)} for p in prior_paths],
        "protocol": "Fixed LightGBM settings and threshold; validate latest listings causally for warm sellers; "
                    "cold sellers disjoint; exclusive cohort excludes archived local validation IDs.",
    }
    (out / "split_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print("split", len(fit), len(val), seen.sum(), fresh.sum(), (fresh & seen).sum(), flush=True)
    tr = ag.precompute_item_intrinsic_arrays(df)
    ag.compute_causal_seller_features = causal_features([tr["start_time"][val]])
    inner = list(StratifiedGroupKFold(4, shuffle=True, random_state=61111).split(
        fit, tr["y"][fit], tr["seller_id"][fit]))
    assert all(not set(tr["seller_id"][fit[a]]) & set(tr["seller_id"][fit[b]]) for a, b in inner)
    m = ag.build_fold_matrices(fit, [val], inner, tr, [tr], out / f"text_{fingerprint[:16]}.npz")
    hist = history(df, fit, np.r_[fit, val])
    extra = extras[:, sorted(METADATA + PICTURES)]
    x = np.column_stack([m["X_m1_seen_causal_T"], extra[fit]])
    xv = np.column_stack([m["X_m1_seen_causal_evals"][0], extra[val]])
    pred = {}
    for key, a, b in (("attributes", x, xv),
                      ("recent_history", np.column_stack([x, hist[:len(fit)]]),
                       np.column_stack([xv, hist[len(fit):]]))):
        model = LGBMClassifier(**PARAMS).fit(a, m["y_T"])
        pred[key] = model.predict_proba(b)[:, 1]
        model.booster_.save_model(str(out / f"{key}.txt"))
        print(key, "trained", flush=True)
    y = tr["y"][val]
    seller, category = tr["seller_id"][val], tr["category_id"][val]
    challenger, base = pred["recent_history"], pred["attributes"]
    pairs = df.iloc[val][["id", "y"]].assign(seen=seen, fresh=fresh, **pred)
    pp = out / "pairs_private.csv"
    pairs.to_csv(pp, index=False)
    cohorts = {"all": np.ones(len(y), bool), "seen": seen, "new": ~seen, "fresh": fresh,
               "fresh_seen": fresh & seen, "fresh_new": fresh & ~seen}
    report["metrics"] = {k: score(y[s], challenger[s], base[s], seller[s], category[s], iterations=2500)
                         for k, s in cohorts.items()}
    flip = (challenger >= .5) != (base >= .5)
    report["disagreement_ids_sha256"] = hashlib.sha256(
        "\n".join(sorted(pairs.id[flip])).encode()).hexdigest()
    report["pairs_sha256"] = sha(pp)
    report["model_sha256"] = {k: sha(out / f"{k}.txt") for k in pred}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["metrics"], flush=True)


if __name__ == "__main__":
    main()
