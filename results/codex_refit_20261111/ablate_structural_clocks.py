"""Frozen two-factor ablation on the CX-0172 seller-group development split."""
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


def main():
    archive = Path(sys.argv[1])
    prior = Path("structural_replica")
    out = Path("structural_clock_ablation")
    out.mkdir(exist_ok=True)
    paired = pd.read_csv(prior / "pairs_private.csv")
    parent = json.loads((prior / "report.json").read_text())
    assert sha(prior / "pairs_private.csv") == parent["pairs_sha256"]
    assert sha(archive) == parent["archive_sha256"]
    with zipfile.ZipFile(archive) as z:
        df, extras = load(z, "train")
    assert len(df) == 70000 and df.id.is_unique and paired.id.is_unique
    val = pd.Index(df.id).get_indexer(paired.id)
    assert (val >= 0).all() and np.array_equal(df.y.to_numpy()[val], paired.y)
    fit = np.setdiff1d(np.arange(len(df)), val)
    import hashlib
    cache_key = hashlib.sha256(np.r_[fit, val].tobytes()).hexdigest()[:16]
    cache = prior / f"text_{cache_key}.npz"
    assert cache.is_file()
    tr = ag.precompute_item_intrinsic_arrays(df)
    ag.compute_causal_seller_features = causal_features([tr["start_time"][val]])
    split = list(StratifiedGroupKFold(4, shuffle=True, random_state=61111).split(
        fit, tr["y"][fit], tr["seller_id"][fit]))
    m = ag.build_fold_matrices(fit, [val], split, tr, [tr], cache)
    x, xv = m["X_m1_seen_causal_T"], m["X_m1_seen_causal_evals"][0]
    hist = add_history(df, fit, np.r_[fit, val])
    t = [7, 8, 9]  # creation minus start, update minus creation, creation time-of-day
    other = [i for i in range(extras.shape[1]) if i not in t]
    assert extras.shape[1] == 19
    added = {
        "clocks_only": (extras[fit][:, t], extras[val][:, t]),
        "attributes_history": (np.column_stack([extras[fit][:, other], hist[:len(fit)]]),
                               np.column_stack([extras[val][:, other], hist[len(fit):]])),
    }
    params = dict(n_estimators=900, learning_rate=.035, num_leaves=48, min_child_samples=35,
                  colsample_bytree=.75, reg_lambda=1, random_state=119, n_jobs=6, verbosity=-1)
    seen = paired.seen.to_numpy(dtype=bool)
    seller = tr["seller_id"][val]
    y = paired.y.to_numpy()
    report = {"source_archive_sha256": sha(archive), "runner_sha256": sha(Path(__file__)),
              "parent_pairs_sha256": parent["pairs_sha256"], "cache_sha256": sha(cache),
              "scope": "new structural clocks only; base still contains duration and causal seller history",
              "development_reused": True, "fit_rows": len(fit), "holdout_rows": len(val), "arms": {}}
    for arm, (ax, av) in added.items():
        model = LGBMClassifier(**params)
        model.fit(np.column_stack([x, ax]), m["y_T"])
        p = model.predict_proba(np.column_stack([xv, av]))[:, 1]
        paired[arm] = p
        path = out / (arm + ".txt")
        model.booster_.save_model(str(path))
        arm_report = {"model_sha256": sha(path)}
        for col in ("p_control", "p_structural", "p_line3"):
            base = paired[col].to_numpy()
            arm_report[col] = {name: metrics(y[mask], p[mask], base[mask], seller[mask])
                               for name, mask in [("all", np.ones(len(y), bool)),
                                                  ("seen", seen), ("new", ~seen)]}
            arm_report[col]["disagreements_id_sha256"] = hashlib.sha256(
                "\n".join(sorted(paired.id[(p >= .5) != (base >= .5)])).encode()).hexdigest()
        report["arms"][arm] = arm_report
        print(arm, {k: arm_report[k]["all"]["net"] for k in ("p_control", "p_structural", "p_line3")}, flush=True)
    pairs = out / "pairs_private.csv"
    paired.to_csv(pairs, index=False)
    report["pairs_sha256"] = sha(pairs)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
