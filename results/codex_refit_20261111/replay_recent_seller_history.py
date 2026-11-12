"""Replay the same recent-history challenger on the frozen CX-0177 cut."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from confirm_structural_models import split
from validate_recent_seller_history import (
    METADATA, PARAMS, PICTURES, StratifiedGroupKFold, ag, causal_features, history, load, score, sha,
)


def main():
    archive = Path(sys.argv[1])
    source = Path("structural_heldout_confirm")
    manifest = json.loads((source / "report.json").read_text())
    prior = list(map(Path, sys.argv[5:]))
    assert [sha(p) for p in prior] == [p["sha256"] for p in manifest["historical_panels"]]
    assert sha(archive) == manifest["archive_sha256"]
    excluded = set()
    for p in prior:
        key = "id" if "id" in pd.read_csv(p, nrows=0) else "item"
        excluded.update(pd.read_csv(p, dtype=str, usecols=[key])[key])
    with zipfile.ZipFile(archive) as z:
        df, extras = load(z, "train")
    fit, val, seen = split(df, excluded)
    split_sha = hashlib.sha256(np.r_[fit, val].tobytes()).hexdigest()
    assert split_sha == manifest["split_sha256"]
    reference = source / "pairs_private.csv"
    assert sha(reference) == manifest["pairs_sha256"]
    pairs = pd.read_csv(reference, dtype={"id": str})
    assert np.array_equal(pairs.id, df.id.to_numpy()[val])
    assert np.array_equal(pairs.y, df.y.to_numpy()[val])
    assert np.array_equal(pairs.seen, seen)
    tr = ag.precompute_item_intrinsic_arrays(df)
    ag.compute_causal_seller_features = causal_features([tr["start_time"][val]])
    inner = list(StratifiedGroupKFold(4, shuffle=True, random_state=61111).split(
        fit, tr["y"][fit], tr["seller_id"][fit]))
    cache = source / f"text_{split_sha[:16]}.npz"
    assert cache.exists()
    m = ag.build_fold_matrices(fit, [val], inner, tr, [tr], cache)
    hist = history(df, fit, np.r_[fit, val])
    extra = extras[:, sorted(METADATA + PICTURES)]
    x = np.column_stack([m["X_m1_seen_causal_T"], extra[fit], hist[:len(fit)]])
    xv = np.column_stack([m["X_m1_seen_causal_evals"][0], extra[val], hist[len(fit):]])
    model = LGBMClassifier(**PARAMS).fit(x, m["y_T"])
    pred = model.predict_proba(xv)[:, 1]
    out = Path("recent_seller_history_replay")
    out.mkdir(exist_ok=True)
    checkpoint = out / "recent_history.txt"
    model.booster_.save_model(str(checkpoint))
    pairs["recent_history"] = pred
    pp = out / "pairs_private.csv"
    pairs.to_csv(pp, index=False)
    seller, cat, y, base = tr["seller_id"][val], tr["category_id"][val], pairs.y.to_numpy(), pairs.lgb.to_numpy()
    stats = {k: score(y[s], pred[s], base[s], seller[s], cat[s], iterations=2500)
             for k, s in {"all": np.ones(len(y), bool), "seen": seen, "new": ~seen}.items()}
    flip = (pred >= .5) != (base >= .5)
    report = {
        "scope": "Reused CX-0177 evaluation. No public projection.",
        "fit_rows": len(fit), "validation_rows": len(val), "split_sha256": split_sha,
        "pairs_sha256": sha(pp), "reference_pairs_sha256": sha(reference),
        "runner_sha256": sha(Path(__file__)), "model_sha256": sha(checkpoint),
        "disagreement_ids_sha256": hashlib.sha256("\n".join(sorted(pairs.id[flip])).encode()).hexdigest(),
        "metrics": stats,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(stats, flush=True)


if __name__ == "__main__":
    main()
