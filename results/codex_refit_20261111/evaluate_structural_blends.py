"""Freeze blend weights on older panels and evaluate on a disjoint validation panel."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

BASE = "p_line3"
OUT = Path("structural_blend_evaluation")


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path, attributes):
    x = pd.read_csv(path, usecols=["id", "y", "seen", BASE, attributes], dtype={"id": str})
    assert x.id.is_unique and x.y.isin([0, 1]).all()
    assert x[[BASE, attributes]].apply(lambda c: c.between(0, 1).all()).all()
    return x.rename(columns={attributes: "p_attributes"})


def choose_weight(train, target=None):
    if target is not None:
        train = train[train.seen == target]
    return min(np.arange(0, 1.001, .025), key=lambda w:
               (log_loss(train.y, (1-w)*train.p_line3+w*train.p_attributes), w))


def score(y, p, base, seller, category, iterations=1400):
    a, b = (p >= .5) == y, (base >= .5) == y
    delta = a.astype(int) - b
    u, cats = pd.factorize(seller)[0], pd.factorize(category)[0]
    rng = np.random.default_rng(7712)
    chosen = delta != 0
    sims = np.empty((2, iterations))
    for j in range(iterations):
        s = rng.poisson(1, u.max() + 1)[u[chosen]]
        c = rng.poisson(1, cats.max() + 1)[cats[chosen]]
        sims[:, j] = np.dot(delta[chosen], s), np.dot(delta[chosen], s*c)
    return {"n": len(y), "hits": int(a.sum()), "baseline_hits": int(b.sum()),
            "net": int(delta.sum()), "fixes": int((delta > 0).sum()), "breaks": int((delta < 0).sum()),
            "logloss": log_loss(y, p), "baseline_logloss": log_loss(y, base),
            "seller_q05_q95": np.quantile(sims[0], [.05, .95]).tolist(),
            "seller_category_q05_q95": np.quantile(sims[1], [.05, .95]).tolist()}


def main():
    archive = Path(sys.argv[1])
    old = [Path("structural_physical_history/original_pairs_private.csv"),
           Path("structural_physical_history/replica_pairs_private.csv")]
    val_path = Path("structural_disjoint/pairs_private.csv")
    develop0, develop1 = [read(p, "physical_and_listing") for p in old]
    duplicate = develop0[["id", "y"]].merge(develop1[["id", "y"]], on="id", validate="one_to_one")
    assert duplicate.y_x.eq(duplicate.y_y).all()
    develop = pd.concat([develop0, develop1]).drop_duplicates("id", keep="first")
    val = read(val_path, "p_attributes")
    assert not set(val.id) & set(develop.id)
    global_weight = float(choose_weight(develop))
    new_weight, seen_weight = (float(choose_weight(develop, v)) for v in (False, True))
    # Fixed shrinkage toward the global solution limits overfitting the route.
    by_regime = {False: .5*(new_weight + global_weight), True: .5*(seen_weight + global_weight)}
    policies = {"equal": .5, "global": global_weight, "routed": val.seen.map(by_regime).to_numpy()}
    needed = set(val.id)
    cats = {}
    with zipfile.ZipFile(archive) as z, z.open("train_data.jsonlines") as f:
        for row in map(json.loads, f):
            item = str(row["id"])
            if item in needed:
                cats[item] = (str(row["seller_id"]), str(row["category_id"]), row["condition"])
    assert set(cats) == needed
    seller, category, condition = np.array([cats[item] for item in val.id]).T
    y = val.y.to_numpy()
    assert np.array_equal(condition == "used", y.astype(bool))
    groups = {"all": np.ones(len(val), bool), "seen": val.seen.to_numpy(), "new": ~val.seen.to_numpy()}
    report = {"development_unique_rows": len(develop), "duplicate_rows_removed": len(duplicate),
              "validation_rows": len(val), "validation_overlap": 0,
              "weights_for_attributes": {"equal": .5, "global": global_weight,
                                         "routed_seen": by_regime[True], "routed_new": by_regime[False]},
              "fit_protocol": "weights frozen by development logloss, threshold fixed .5; archived disjoint panel reused",
              "source_hashes": {p.name: sha(p) for p in [*old, val_path, archive]},
              "runner_sha256": sha(Path(__file__)), "metrics": {}}
    for name, w in policies.items():
        pred = (1-w)*val.p_line3.to_numpy()+w*val.p_attributes.to_numpy()
        report["metrics"][name] = {}
        val["p_" + name] = pred
        for column in (BASE, "p_attributes"):
            base = val[column].to_numpy()
            r = {k: score(y[s], pred[s], base[s], seller[s], category[s]) for k, s in groups.items()}
            r["disagreement_ids_sha256"] = hashlib.sha256(
                "\n".join(sorted(val.id[(pred >= .5) != (base >= .5)])).encode()).hexdigest()
            report["metrics"][name][column] = r
    OUT.mkdir(exist_ok=True)
    pairs = OUT / "pairs_private.csv"
    val.to_csv(pairs, index=False)
    report["pairs_sha256"] = sha(pairs)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for n, rr in report["metrics"].items():
        print(n, {b: (m["all"]["net"], m["all"]["logloss"], m["all"]["seller_q05_q95"])
                  for b, m in rr.items()})
    print(report["weights_for_attributes"])


if __name__ == "__main__":
    main()
