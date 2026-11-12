"""Train a frozen CatBoost attribute alternative on the disjoint seller-group cut."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from ap_quinta_catboost import FROZEN_SETTINGS
from evaluate_structural_blends import score
from validate_structural_disjoint import (
    METADATA, PICTURES, StratifiedGroupKFold, ag, causal_features, load, make_split, prior_ids, sha,
)


def main():
    src, old_dir, ext = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[4])
    source = Path("structural_disjoint")
    manifest = json.loads((source / "report.json").read_text())
    assert manifest["archive_sha256"] == sha(src)
    old_paths = [old_dir / "line3_today/pairs.csv", Path("structural_replica/pairs_private.csv"),
                 ext / "structural_replication.csv"]
    assert [sha(p) for p in old_paths] == manifest["earlier_pairs_sha256"]
    with zipfile.ZipFile(src) as archive:
        df, extras = load(archive, "train")
    fit, val, seen = make_split(df, prior_ids(old_paths))
    split_sha = hashlib.sha256(np.r_[fit, val].tobytes()).hexdigest()
    assert split_sha == manifest["split_sha256"]
    pairpath = source / "pairs_private.csv"
    assert sha(pairpath) == manifest["pairs_sha256"]
    pairs = pd.read_csv(pairpath, dtype={"id": str})
    assert np.array_equal(pairs.id, df.id.to_numpy()[val])
    assert np.array_equal(pairs.y, df.y.to_numpy()[val]) and np.array_equal(pairs.seen, seen)
    tr = ag.precompute_item_intrinsic_arrays(df)
    ag.compute_causal_seller_features = causal_features([tr["start_time"][val]])
    inner = list(StratifiedGroupKFold(4, shuffle=True, random_state=61111).split(
        fit, tr["y"][fit], tr["seller_id"][fit]))
    assert all(not set(tr["seller_id"][fit[a]]) & set(tr["seller_id"][fit[b]]) for a, b in inner)
    cache = source / ("text_" + split_sha[:16] + ".npz")
    assert cache.is_file()
    m = ag.build_fold_matrices(fit, [val], inner, tr, [tr], cache)
    extra = extras[:, sorted(METADATA + PICTURES)]
    x = np.column_stack([m["X_m1_seen_causal_T"], extra[fit]])
    xv = np.column_stack([m["X_m1_seen_causal_evals"][0], extra[val]])
    model = CatBoostClassifier(**FROZEN_SETTINGS).fit(x, m["y_T"])
    p = model.predict_proba(xv)[:, 1]
    out = Path("structural_catboost_disjoint")
    out.mkdir(exist_ok=True)
    model.save_model(str(out / "model.cbm"))
    candidates = {"cat_attributes": p, "equal_with_lgb": .5*(p+pairs.p_attributes.to_numpy())}
    report = {"fit_rows": len(fit), "validation_rows": len(val), "holdout_overlap": 0,
              "development_panel_reused": True, "archive_sha256": sha(src), "split_sha256": split_sha,
              "source_pairs_sha256": sha(pairpath), "runner_sha256": sha(Path(__file__)),
              "model_sha256": sha(out / "model.cbm"), "metrics": {}}
    y, sellers, cats = pairs.y.to_numpy(), tr["seller_id"][val], tr["category_id"][val]
    groups = {"all": np.ones(len(y), bool), "seen": seen, "new": ~seen}
    for name, p in candidates.items():
        pairs["p_" + name] = p
        report["metrics"][name] = {}
        for b in ("p_attributes", "p_line3"):
            base = pairs[b].to_numpy()
            rs = {k: score(y[s], p[s], base[s], sellers[s], cats[s]) for k, s in groups.items()}
            rs["disagreement_ids_sha256"] = hashlib.sha256(
                "\n".join(sorted(pairs.id[(p >= .5) != (base >= .5)])).encode()).hexdigest()
            report["metrics"][name][b] = rs
    pairs.to_csv(out / "pairs_private.csv", index=False)
    report["pairs_sha256"] = sha(out / "pairs_private.csv")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for n, rr in report["metrics"].items():
        print(n, {b: (m["all"]["net"], m["all"]["logloss"], m["all"]["seller_q05_q95"])
                  for b, m in rr.items()}, flush=True)


if __name__ == "__main__":
    main()
