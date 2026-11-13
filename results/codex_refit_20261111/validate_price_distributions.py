"""Paired ablation of seller-cross-fitted conditional price distributions."""
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

from validate_variant_inventory import METADATA, PICTURES, StratifiedGroupKFold, load, run, sha


def price_distributions(df, fit, val):
    log_price = np.log1p(df.price.to_numpy().clip(0))
    y = df.y.to_numpy()
    groupings = [list(zip(*(df[k] for k in keys))) for keys in (
        ("currency_id", "category_id"),
        ("currency_id", "category_id", "brand_val"),
        ("currency_id", "category_id", "item_cond_attr"),
    )]
    out = np.full((len(df), 19), np.nan, dtype=np.float32)
    # Column zero lets the paired runner score the entire cohort.
    out[:, 0] = log_price

    def fill(train, query):
        for j, groups in enumerate(groupings):
            tables = {}
            for i in train:
                tables.setdefault((groups[i], y[i]), []).append(log_price[i])
            stats = {key: (p := np.sort(values), np.median(p)) for key, values in tables.items()}
            for i in query:
                for label in (0, 1):
                    column = 1 + 6*j + 3*label
                    table = stats.get((groups[i], label))
                    out[i, column] = 0 if table is None else np.log1p(len(table[0]))
                    if table is None or len(table[0]) < 12:
                        continue
                    p, median = table
                    l = p.searchsorted(log_price[i], side="left")
                    r = p.searchsorted(log_price[i], side="right")
                    out[i, column+1:column+3] = ((l+r+1)/(2*(len(p)+1)), log_price[i]-median)

    splitter = StratifiedGroupKFold(4, shuffle=True, random_state=61111)
    for tr, ev in splitter.split(fit, y[fit], df.seller_id.iloc[fit]):
        assert not set(df.seller_id.iloc[fit[tr]]) & set(df.seller_id.iloc[fit[ev]])
        fill(fit[tr], fit[ev])
    fill(fit, val)
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
    destination = "price_distributions"
    report = {
        "scope": "Train-only archived development panels. Label-conditioned price references are fit seller-group OOF "
                 "on train and frozen on evaluation. Published condition attribute is an input; hidden labels are never inputs.",
        "archive_sha256": sha(archive), "runner_sha256": sha(Path(__file__)),
        "shared_runner_sha256": sha(Path(__file__).with_name("validate_variant_inventory.py")),
        "panels": {source.name: run(df, extra, price_distributions, source, paths if i == 0 else paths[:-1],
                                   i == 0, destination) for i, source in enumerate(sources)},
    }
    for panel in report["panels"].values():
        panel["metrics"].pop("variations")
        panel["metrics"].pop("fresh_variations")
    (Path(destination) / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
