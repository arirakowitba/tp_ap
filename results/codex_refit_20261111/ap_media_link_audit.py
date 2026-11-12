"""Evaluate exact listing/media links using labels from the fit partition only."""
import hashlib
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import log_loss

OLD = Path("/workspace/scratch/1a8e5d0ea219")
ARCHIVE = Path(sys.argv[1])
OUT = Path("media_link_result")
OUT.mkdir(exist_ok=True)


def parse(row):
    pictures = row.get("pictures") or []
    # Normalize the same asset across sizes and URL hosts using its picture ID.
    photos = {str(x["id"]) for x in pictures if isinstance(x, dict) and x.get("id")}
    for key in ("thumbnail", "secure_thumbnail"):
        url = str(row.get(key) or "")
        match = re.search(r"/(?:D_[A-Z]+_)?(\d+-ML[A-Z]\d+_\d+)", url)
        if match:
            photos.add(match.group(1))
    return (str(row["id"]), str(row["seller_id"]), int(row.get("start_time") or 0),
            int(row.get("condition") == "used"), tuple(sorted(photos)),
            str(row.get("parent_item_id") or ""), str(row.get("catalog_product_id") or ""))


def neighbors(records, train, target):
    stores = [defaultdict(list) for _ in range(3)]
    for i in train:
        item, seller, stamp, y, photos, parent, catalog = records[i]
        for j, keys in enumerate([photos, (parent,) if parent else (), (catalog,) if catalog else ()]):
            for key in keys:
                stores[j][key].append((i, seller, stamp, y))
    assert not set(train) & set(target)
    stats = np.zeros((len(target), 3, 2), dtype=int)
    for position, i in enumerate(target):
        item, seller, stamp, y, photos, parent, catalog = records[i]
        for j, keys in enumerate([photos, (parent,) if parent else (), (catalog,) if catalog else ()]):
            # Keep one fit observation per seller. For the same seller use only
            # items strictly older than the target publication.
            related = {}
            for key in keys:
                for src, s, date, label in stores[j].get(key, ()):
                    if s == seller and date >= stamp:
                        continue
                    if s not in related or (date, src) > related[s][:2]:
                        related[s] = date, src, label
            stats[position, j] = (len(related), sum(r[2] for r in related.values()))
    return stats


def apply(base, stats, rule):
    channel, strength, weight = rule
    n, u = stats[:, channel].T
    eligible = (n >= strength) & ((u == 0) | (u == n))
    probability = (u + .5) / (n + 1)
    p = base.copy()
    p[eligible] = expit(logit(np.clip(base[eligible], 1e-6, 1 - 1e-6)) +
                        weight * logit(probability[eligible]))
    return p, eligible


def metrics(y, sellers, base, pred, coverage):
    delta = ((pred >= .5) == y).astype(int) - ((base >= .5) == y)
    blocks = pd.Series(delta).groupby(sellers).sum().to_numpy()
    rng = np.random.default_rng(26)
    draws = [rng.choice(blocks, len(blocks), replace=True).sum() for _ in range(1000)]
    return {"n": len(y), "coverage": int(coverage.sum()), "net": int(delta.sum()),
            "fixes": int((delta == 1).sum()), "breaks": int((delta == -1).sum()),
            "q05": float(np.quantile(draws, .05)), "q95": float(np.quantile(draws, .95)),
            "base_logloss": float(log_loss(y, base)), "logloss": float(log_loss(y, pred))}


with zipfile.ZipFile(ARCHIVE) as archive:
    tr = [parse(json.loads(row)) for row in archive.open("train_data.jsonlines")]
assert len(tr) == 70000 and len({r[0] for r in tr}) == len(tr)
pairs = pd.read_csv(OLD / "line3_today/pairs.csv")
val = pairs.row.to_numpy()
assert len(set(val)) == len(val)
assert all(tr[i][0] == v and tr[i][3] == y for i, v, y in zip(val, pairs.id, pairs.y, strict=True))
fit = np.setdiff1d(np.arange(len(tr)), val)
stats = neighbors(tr, fit, val)
y, sellers, base = pairs.y.to_numpy(), pairs.seller_id.to_numpy(), pairs.m1.to_numpy()
tune = pd.util.hash_pandas_object(pairs.seller_id, index=False).to_numpy() % 2 == 0
rules = [(channel, n, w) for channel in range(3) for n in (1, 2, 3) for w in (.25, .5, 1.)]
grid = []
for rule in rules:
    p, c = apply(base[tune], stats[tune], rule)
    m = metrics(y[tune], sellers[tune], base[tune], p, c)
    grid.append((rule, m))
# Compare all to the option of preserving the paired baseline.
rule, best = max(grid, key=lambda x: (x[1]["q05"], x[1]["net"],
                                     x[1]["base_logloss"] - x[1]["logloss"]))
confirm = ~tune
pred, cov = apply(base[confirm], stats[confirm], rule)
segments = {}
for value in (False, True):
    mask = pairs.seen.to_numpy()[confirm] == value
    segments[str(value)] = metrics(y[confirm][mask], sellers[confirm][mask],
                                  base[confirm][mask], pred[mask], cov[mask])
delta = ((pred >= .5) == y[confirm]).astype(int) - ((base[confirm] >= .5) == y[confirm])
report = {"n_train_source": len(tr), "n_fit": len(fit), "n_holdout_unique": len(val),
          "paired_baseline": "Line 3 M1, seller-group OOF text and causal seller history",
          "selected_rule": rule, "rule_channel": ["picture", "parent", "catalog"][rule[0]],
          "tune": best, "confirm_development": metrics(y[confirm], sellers[confirm], base[confirm], pred, cov),
          "confirm_seen": segments,
          "coverage_by_channel": (stats[:, :, 0] > 0).sum(axis=0).tolist(),
          "confirm_disagreements_id_sha256": hashlib.sha256(
              "\n".join(sorted(pairs.id.to_numpy()[confirm][delta != 0])).encode()).hexdigest(),
          "pairs_sha256": hashlib.sha256((OLD / "line3_today/pairs.csv").read_bytes()).hexdigest(),
          "tune_grid": grid}
private = pairs.loc[confirm, ["id", "seen", "y", "m1"]].assign(p_link=pred, changed=delta != 0)
private.to_csv(OUT / "pairs_private.csv", index=False)
report["confirm_pairs_sha256"] = hashlib.sha256((OUT / "pairs_private.csv").read_bytes()).hexdigest()
(OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "tune_grid"}, indent=2))
