"""Second seller-group split, paired metadata ablation with a frozen Line3 reference."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

STRUCTURAL = Path(sys.argv[4])
sys.path.insert(0, str(STRUCTURAL))
from train_structural import add_history, ag, causal_features, load, metrics, StratifiedGroupKFold


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    archive = Path(sys.argv[1])
    out = Path("structural_replica")
    out.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        df, extras = load(z, "train")
    assert len(df) == 70000 and df.id.is_unique
    cold_fit, cold_val = next(StratifiedGroupKFold(5, shuffle=True, random_state=281113).split(
        df, df.y, df.seller_id))
    warm_df = df.iloc[cold_fit]
    grouped = warm_df.groupby("seller_id").start_time
    last, first = grouped.transform("max"), grouped.transform("min")
    warm_val = warm_df.index[(warm_df.start_time == last) & (first < last)]
    val = np.sort(np.r_[cold_val, warm_val])
    fit = np.setdiff1d(np.arange(len(df)), val)
    seen = ~np.isin(val, cold_val)
    assert not set(df.seller_id.iloc[cold_val]) & set(df.seller_id.iloc[fit])
    assert len(np.unique(val)) == len(val)
    assert np.isin(df.seller_id.iloc[val[seen]], df.seller_id.iloc[fit]).all()
    tr = ag.precompute_item_intrinsic_arrays(df)
    ag.compute_causal_seller_features = causal_features([tr["start_time"][val]])
    inner = list(StratifiedGroupKFold(4, shuffle=True, random_state=61111).split(
        fit, tr["y"][fit], tr["seller_id"][fit]))
    assert all(not set(tr["seller_id"][fit[a]]) & set(tr["seller_id"][fit[b]]) for a, b in inner)
    cache = out / ("text_" + hashlib.sha256(np.r_[fit, val].tobytes()).hexdigest()[:16] + ".npz")
    m = ag.build_fold_matrices(fit, [val], inner, tr, [tr], cache)
    print("replication matrices ready", len(fit), len(val), flush=True)
    x, xv = m["X_m1_seen_causal_T"], m["X_m1_seen_causal_evals"][0]
    hist = add_history(df, fit, np.r_[fit, val])
    ext, ev = np.column_stack([extras[fit], hist[:len(fit)]]), np.column_stack([extras[val], hist[len(fit):]])
    params = dict(n_estimators=900, learning_rate=.035, num_leaves=48, min_child_samples=35,
                  colsample_bytree=.75, reg_lambda=1, random_state=119, n_jobs=6, verbosity=-1)
    predictions = {}
    for name, (fx, vx) in {"controlled": (x, xv),
                           "structural": (np.column_stack([x, ext]), np.column_stack([xv, ev]))}.items():
        model = LGBMClassifier(**params)
        model.fit(fx, m["y_T"])
        predictions[name] = model.predict_proba(vx)[:, 1]
        model.booster_.save_model(str(out / (name + ".txt")))
        print(name, "done", flush=True)
    hp = {"num_leaves": 55, "colsample_bytree": .7, "min_child_samples": 40}
    for regime in ("unseen", "seen_causal"):
        key = f"X_m1_{regime}"
        predictions[regime] = ag.fit_predict_lgb_bagged(
            m[key + "_T"], m["y_T"], m[key + "_evals"], hp, [42, 2026, 777], 900)[0]
    line3 = np.where(seen, predictions["seen_causal"], predictions["unseen"])
    y = tr["y"][val]
    old = pd.read_csv(Path(sys.argv[2]) / "line3_today/pairs.csv", usecols=["row"]).row.to_numpy()
    report = {"fit_n": len(fit), "val_n": len(val), "seen_n": int(seen.sum()),
              "old_holdout_overlap": int(np.isin(val, old).sum()),
              "archive_sha256": sha(archive), "code_sha256": sha(Path(__file__)),
              "structural_source_sha256": sha(STRUCTURAL / "train_structural.py")}
    pred = predictions["structural"]
    frame = df.iloc[val][["id", "y"]].assign(seen=seen, p_control=predictions["controlled"],
                                            p_line3=line3, p_structural=pred)
    frame.to_csv(out / "pairs_private.csv", index=False)
    report["pairs_sha256"] = sha(out / "pairs_private.csv")
    report["disagreement_id_sha256"] = hashlib.sha256("\n".join(sorted(
        frame.id[(pred >= .5) != (line3 >= .5)])).encode()).hexdigest()
    report["model_sha256"] = sha(out / "structural.txt")
    for name, base in [("vs_control", predictions["controlled"]), ("vs_line3", line3)]:
        report[name] = {}
        for key, mask in [("all", np.ones(len(val), bool)), ("seen", seen), ("new", ~seen)]:
            report[name][key] = metrics(y[mask], pred[mask], base[mask], tr["seller_id"][val][mask])
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
