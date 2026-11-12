"""Stress paired validation under a range of seen-seller weights."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    archive, pairs_path = map(Path, sys.argv[1:3])
    pairs = pd.read_csv(pairs_path, dtype={"id": str})
    assert pairs.id.is_unique
    ids = set(pairs.id)
    with zipfile.ZipFile(archive) as z, z.open("train_data.jsonlines") as f:
        groups = {str(row["id"]): (str(row["seller_id"]), str(row["category_id"]))
                  for row in map(json.loads, f) if str(row["id"]) in ids}
    assert len(groups) == len(ids)
    seller, category = zip(*(groups[item] for item in pairs.id))
    s, c = pd.factorize(seller)[0], pd.factorize(category)[0]
    seen = pairs.seen.to_numpy(dtype=bool)
    group = seen.astype(int)
    sizes = np.bincount(group)
    y = pairs.y.to_numpy()
    mix = np.array([.0, .25, .5, .7, .9, 1.])
    rng = np.random.default_rng(28111223)
    report = {"archive_sha256": sha(archive), "source_pairs_sha256": sha(pairs_path),
              "scenario_seen_share": mix.tolist(),
              "scope": "paired validation accuracy in percentage points at hypothetical mixtures; not public forecasts",
              "results": {}}
    for name in ("cat", "equal"):
        for baseline in ("lgb", "line3"):
            p, b = (pairs[key].to_numpy() >= .5 for key in (name, baseline))
            delta = (p == y).astype(int) - (b == y).astype(int)
            flip = delta != 0
            boot = np.empty((2500, 2))
            for i in range(len(boot)):
                w = rng.poisson(1, s.max() + 1)[s] * rng.poisson(1, c.max() + 1)[c]
                counts = np.bincount(group, weights=w, minlength=2)
                boot[i] = np.bincount(group[flip], weights=delta[flip]*w[flip], minlength=2) / counts
            rates = np.bincount(group[flip], weights=delta[flip], minlength=2) / sizes
            weights = np.vstack((1-mix, mix))
            direction = {}
            for is_seen in (False, True):
                for start_used in (False, True):
                    d = delta[flip & (seen == is_seen) & (b == start_used)]
                    direction[f"{'seen' if is_seen else 'new'}_{'used_to_new' if start_used else 'new_to_used'}"] = {
                        "fixes": int((d > 0).sum()), "breaks": int((d < 0).sum())}
            report["results"][f"{name}_vs_{baseline}"] = {
                "gain_pp": (100*rates @ weights).tolist(),
                "cluster_q05_gain_pp": (100*np.quantile(boot @ weights, .05, axis=0)).tolist(),
                "cluster_q95_gain_pp": (100*np.quantile(boot @ weights, .95, axis=0)).tolist(),
                "direction": direction}
    pairs_path.with_name("mix_sensitivity.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
