"""Paired factorial ablation with fixed learner and two seller-group development cuts."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from replicate_structural_metadata import (
    StratifiedGroupKFold, add_history, ag, causal_features, load, metrics, sha,
)

PARAMS = dict(n_estimators=900, learning_rate=.035, num_leaves=48, min_child_samples=35,
              colsample_bytree=.75, reg_lambda=1, random_state=119, n_jobs=6, verbosity=-1)
METADATA = [0, 1, 2, 3, 4, 5, 6, 18, 19]
PICTURES = [10, 11, 12, 13, 14, 15, 16, 17]
ARMS = {
    "physical_and_listing": ("publication",),
    "additional_seller_history": ("history",),
    "both_no_new_clocks": ("publication", "history"),
}


def run_cut(df, extras, arrays, name, paired, cache, other_ids, out):
    val = pd.Index(df.id).get_indexer(paired.id)
    assert paired.id.is_unique and (val >= 0).all()
    assert np.array_equal(df.y.to_numpy()[val], paired.y)
    fit = np.setdiff1d(np.arange(len(df)), val)
    assert not set(val) & set(fit) and cache.is_file()
    seen = paired.seen.to_numpy(dtype=bool)
    seller = arrays["seller_id"][val]
    assert np.array_equal(seen, np.isin(seller, arrays["seller_id"][fit]))
    ag.compute_causal_seller_features = causal_features([arrays["start_time"][val]])
    inner = list(StratifiedGroupKFold(4, shuffle=True, random_state=61111).split(
        fit, arrays["y"][fit], arrays["seller_id"][fit]))
    assert all(not set(arrays["seller_id"][fit[a]]) & set(arrays["seller_id"][fit[b]]) for a, b in inner)
    m = ag.build_fold_matrices(fit, [val], inner, arrays, [arrays], cache)
    order = np.r_[fit, val]
    extra = {"publication": extras[order][:, sorted(METADATA + PICTURES)],
             "history": add_history(df, fit, order)}
    if "p_control" not in paired:
        control = LGBMClassifier(**PARAMS).fit(m["X_m1_seen_causal_T"], m["y_T"])
        paired["p_control"] = control.predict_proba(m["X_m1_seen_causal_evals"][0])[:, 1]
    if "both_no_new_clocks" in paired:
        arms = {k: v for k, v in ARMS.items() if k != "both_no_new_clocks"}
    else:
        arms = ARMS
    for arm, groups in arms.items():
        block = np.column_stack([extra[g] for g in groups])
        model = LGBMClassifier(**PARAMS)
        model.fit(np.column_stack([m["X_m1_seen_causal_T"], block[:len(fit)]]), m["y_T"])
        paired[arm] = model.predict_proba(np.column_stack(
            [m["X_m1_seen_causal_evals"][0], block[len(fit):]]))[:, 1]
        print(name, arm, "done", flush=True)
    y = paired.y.to_numpy()
    report = {"fit_n": len(fit), "val_n": len(val), "cache_sha256": sha(cache),
              "overlap_with_other_cut": int(paired.id.isin(other_ids).sum()), "arms": {}}
    for arm in ARMS:
        p = paired[arm].to_numpy()
        result = {}
        for baseline in ("p_control", "p_line3"):
            b = paired[baseline].to_numpy()
            masks = {"all": np.ones(len(y), dtype=bool), "seen": seen, "new": ~seen,
                     "exclusive_to_cut": ~paired.id.isin(other_ids).to_numpy()}
            result[baseline] = {k: metrics(y[s], p[s], b[s], seller[s]) for k, s in masks.items()}
            result[baseline]["disagreement_ids_sha256"] = hashlib.sha256(
                "\n".join(sorted(paired.id[(p >= .5) != (b >= .5)])).encode()).hexdigest()
        report["arms"][arm] = result
    path = out / f"{name}_pairs_private.csv"
    paired.to_csv(path, index=False)
    report["pairs_sha256"] = sha(path)
    (out / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(name, {k: v["p_line3"]["all"]["net"] for k, v in report["arms"].items()}, flush=True)


def main():
    archive = Path(sys.argv[1])
    old_dir = Path(sys.argv[2])
    out = Path("structural_physical_history")
    out.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        df, extras = load(z, "train")
    assert len(df) == 70000 and df.id.is_unique and extras.shape[1] == 20
    arrays = ag.precompute_item_intrinsic_arrays(df)
    earlier = pd.read_csv(old_dir / "line3_today/pairs.csv").rename(columns={"m1": "p_line3"})
    later_path = Path("structural_clock_ablation/pairs_private.csv")
    assert sha(later_path) == json.loads(Path("structural_clock_ablation/report.json").read_text())["pairs_sha256"]
    later = pd.read_csv(later_path).rename(columns={"attributes_history": "both_no_new_clocks"})
    earlier_cache = old_dir / "line3_today" / f"text_{len(df)-len(earlier)}.npz"
    later_cache, = Path("structural_replica").glob("text_*.npz")
    if not (out / "original.json").exists():
        run_cut(df, extras, arrays, "original", earlier, earlier_cache, later.id, out)
    if not (out / "replica.json").exists():
        run_cut(df, extras, arrays, "replica", later, later_cache, earlier.id, out)


if __name__ == "__main__":
    main()
