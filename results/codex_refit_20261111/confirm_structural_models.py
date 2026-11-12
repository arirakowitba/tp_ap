"""Compare frozen structural candidates on IDs outside archived local validation."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from ap_quinta_catboost import FROZEN_SETTINGS
from evaluate_structural_blends import score
from validate_structural_disjoint import (
    METADATA, PICTURES, PARAMS, StratifiedGroupKFold, ag, causal_features, load, sha,
)


def split(df, excluded):
    rng = np.random.default_rng(28111212)
    available = ~df.id.isin(excluded)
    keys = np.sort(df.loc[available, "seller_id"].unique())
    rng.shuffle(keys)
    cold = df.seller_id.isin(keys[:len(keys) // 3])
    remaining = df.loc[available & ~cold, ["seller_id", "start_time"]]
    latest = remaining.groupby("seller_id").start_time.max()
    earliest = df.groupby("seller_id").start_time.min()
    warm_keys = np.sort(latest.index[latest.gt(earliest.reindex(latest.index))])
    rng.shuffle(warm_keys)
    cutoff = df.seller_id.map(latest.reindex(warm_keys[:len(warm_keys) // 2]))
    warm = df.start_time.ge(cutoff)
    fit = np.flatnonzero(~(cold | warm))
    val = np.flatnonzero(available & (cold | df.start_time.eq(cutoff)))
    seen = ~cold.to_numpy()[val]
    assert not set(df.id.iloc[val]) & excluded
    assert not set(df.seller_id.iloc[val[~seen]]) & set(df.seller_id.iloc[fit])
    latest_fit = df.iloc[fit].groupby("seller_id").start_time.max()
    assert df.seller_id.iloc[val[seen]].map(latest_fit).lt(df.start_time.iloc[val[seen]]).all()
    return fit, val, seen


def main():
    src = Path(sys.argv[1])
    prior = [Path(p) for p in sys.argv[5:]]
    assert prior
    with zipfile.ZipFile(src) as archive:
        df, extras = load(archive, "train")
    assert len(df) == 70_000 and df.id.is_unique
    excluded = set()
    archive_id_sets = {}
    for p in prior:
        key = "id" if "id" in pd.read_csv(p, nrows=0).columns else "item"
        ids = set(pd.read_csv(p, usecols=[key], dtype=str)[key])
        assert ids <= set(df.id), p
        excluded.update(ids)
        archive_id_sets[str(p)] = len(ids)
    fit, val, seen = split(df, excluded)
    assert len(val) > 1000 and seen.any() and (~seen).any()
    digest = hashlib.sha256(np.r_[fit, val].tobytes()).hexdigest()
    out = Path("structural_heldout_confirm")
    out.mkdir(exist_ok=True)
    report = {
        "fit_rows": len(fit), "validation_rows": len(val), "seen_rows": int(seen.sum()),
        "omitted_rows": len(df)-len(fit)-len(val), "excluded_ids": len(excluded),
        "previous_validation_id_overlap": 0,
        "archive_sha256": sha(src), "split_sha256": digest, "runner_sha256": sha(Path(__file__)),
        "historical_panels": [{"file": p.parent.name + "/" + p.name, "sha256": sha(p),
                               "rows": archive_id_sets[str(p)]} for p in prior],
        "protocol": "fixed configs and .5 threshold; CatBoost or equal blend, seller-group inner OOF; separate warm causal cohort",
    }
    (out / "split_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print("fit / evaluation / seen / excluded", len(fit), len(val), seen.sum(), len(excluded), flush=True)
    tr = ag.precompute_item_intrinsic_arrays(df)
    ag.compute_causal_seller_features = causal_features([tr["start_time"][val]])
    inner = list(StratifiedGroupKFold(4, shuffle=True, random_state=61111).split(
        fit, tr["y"][fit], tr["seller_id"][fit]))
    assert all(not set(tr["seller_id"][fit[a]]) & set(tr["seller_id"][fit[b]]) for a, b in inner)
    matrices = ag.build_fold_matrices(fit, [val], inner, tr, [tr], out / f"text_{digest[:16]}.npz")
    print("OOF matrices ready", flush=True)
    extra = extras[:, sorted(METADATA + PICTURES)]
    x = np.column_stack([matrices["X_m1_seen_causal_T"], extra[fit]])
    xv = np.column_stack([matrices["X_m1_seen_causal_evals"][0], extra[val]])
    y_fit = matrices["y_T"]
    lgb = LGBMClassifier(**PARAMS).fit(x, y_fit)
    lgb.booster_.save_model(str(out / "attributes.txt"))
    cat = CatBoostClassifier(**FROZEN_SETTINGS).fit(x, y_fit)
    cat.save_model(str(out / "attributes.cbm"))
    print("attribute models ready", flush=True)
    line3 = {}
    hp = {"num_leaves": 55, "colsample_bytree": .7, "min_child_samples": 40}
    for regime in ("unseen", "seen_causal"):
        key = f"X_m1_{regime}"
        line3[regime] = ag.fit_predict_lgb_bagged(
            matrices[key + "_T"], y_fit, matrices[key + "_evals"], hp, [42, 2026, 777], 900)[0]
    y = tr["y"][val]
    p_lgb, p_cat = lgb.predict_proba(xv)[:, 1], cat.predict_proba(xv)[:, 1]
    refs = {"lgb": p_lgb, "line3": np.where(seen, line3["seen_causal"], line3["unseen"])}
    candidates = {"cat": p_cat, "equal": .5 * (p_cat + p_lgb)}
    pairs = df.iloc[val][["id", "y"]].assign(seen=seen, **refs, **candidates)
    pairs_path = out / "pairs_private.csv"
    pairs.to_csv(pairs_path, index=False)
    seller, category = tr["seller_id"][val], tr["category_id"][val]
    cohorts = {"all": np.ones(len(y), bool), "new": ~seen, "seen": seen}
    report["metrics"] = {}
    for name, pred in candidates.items():
        report["metrics"][name] = {}
        for baseline, base in refs.items():
            stats = {k: score(y[s], pred[s], base[s], seller[s], category[s], iterations=2500)
                     for k, s in cohorts.items()}
            flip = (pred >= .5) != (base >= .5)
            stats["disagreement_ids_sha256"] = hashlib.sha256(
                "\n".join(sorted(pairs.id[flip])).encode()).hexdigest()
            report["metrics"][name][baseline] = stats
    report["pairs_sha256"] = sha(pairs_path)
    report["checkpoints_sha256"] = {name: sha(out / name) for name in ("attributes.txt", "attributes.cbm")}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, results in report["metrics"].items():
        print(name, {b: groups["all"] for b, groups in results.items()}, flush=True)


if __name__ == "__main__":
    main()
