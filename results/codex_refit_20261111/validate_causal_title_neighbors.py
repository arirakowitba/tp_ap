"""Train on strictly earlier title neighbors in a seller's own inventory."""
import json
from pathlib import Path
import re
import sys
import unicodedata
import zipfile

import numpy as np

from validate_variant_inventory import METADATA, PICTURES, load, run, sha


def token_set(title):
    plain = unicodedata.normalize("NFKD", title.casefold())
    return frozenset(re.findall(r"[a-z0-9]{2,}", plain.encode("ascii", "ignore").decode()))


def title_neighbors(df, fit, val):
    titles = [token_set(x) for x in df.title]
    time = df.start_time.to_numpy()
    price = np.log1p(df.price.to_numpy().clip(0))
    y = df.y.to_numpy()
    keys = list(zip(df.seller_id, df.category_id))
    histories = {}
    for i in fit[np.argsort(time[fit], kind="stable")]:
        histories.setdefault(keys[i], []).append(i)
    histories = {key: (np.asarray(rows), time[rows]) for key, rows in histories.items()}
    out = np.full((len(df), 7), np.nan, dtype=np.float32)
    for i in np.r_[fit, val]:
        if not titles[i] or keys[i] not in histories:
            continue
        rows, times = histories[keys[i]]
        stop = times.searchsorted(time[i], side="left")
        pool = rows[max(0, stop-40):stop]
        sims = np.array([len(titles[i] & titles[j])/len(titles[i] | titles[j]) for j in pool])
        match = np.flatnonzero(sims >= .25)
        if not len(match):
            continue
        top = match[np.argsort(-sims[match], kind="stable")[:5]]
        neighbors, weights = pool[top], sims[top]**2
        assert (time[neighbors] < time[i]).all()
        out[i] = (sims[top[0]], np.log1p(len(match)), y[neighbors[0]],
                  np.dot(y[neighbors], weights)/weights.sum(),
                  price[i]-price[neighbors[0]], np.log1p((time[i]-time[neighbors[0]])/86400000),
                  np.sum(sims[match] == 1))
    return out


def main():
    archive = Path(sys.argv[1])
    paths = list(map(Path, sys.argv[5:]))
    assert len(paths) == 8
    sources = (Path("recent_seller_history"), Path("structural_heldout_confirm"))
    for source in sources:
        assert sha(archive) == json.loads((source / "report.json").read_text())["archive_sha256"]
    with zipfile.ZipFile(archive) as z:
        df, extra = load(z, "train")
    assert len(df) == 70000 and df.id.is_unique
    extra = extra[:, sorted(METADATA + PICTURES)]
    destination = "causal_title_neighbors"
    report = {
        "scope": "Archived local development panels. Frozen rule: seller/category, last 40 strictly earlier fit records, "
                 "Jaccard >= .25, up to 5 weighted neighbors, fallback to baseline without support. No public projection.",
        "archive_sha256": sha(archive), "runner_sha256": sha(Path(__file__)),
        "shared_runner_sha256": sha(Path(__file__).with_name("validate_variant_inventory.py")),
        "panels": {source.name: run(df, extra, title_neighbors, source, paths if i == 0 else paths[:-1],
                                   i == 0, destination) for i, source in enumerate(sources)},
    }
    # The historical runner named its generic support metrics after variants.
    for panel in report["panels"].values():
        for key in ("variations", "fresh_variations"):
            panel["metrics"][key.replace("variations", "title_neighbors")] = panel["metrics"].pop(key)
    (Path(destination) / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
