"""One paired ablation of causal seller-category/brand histories."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss

OLD = Path("/workspace/scratch/1a8e5d0ea219")
sys.path.insert(0, str(OLD))
from refit_ap_line3 import ag, causal_features, parser
from ap_quinta_catboost import FROZEN_SETTINGS


def add_history(frame, fit, query):
    blocks = []
    all_price = np.log1p(frame.price.to_numpy().clip(0))
    for keys in [("seller_id", "category_id"), ("seller_id", "brand_val"),
                 ("seller_id", "listing_type_id")]:
        history = {}
        table = frame.iloc[fit].assign(log_price=all_price[fit])
        for key, rows in table.groupby(list(keys), sort=False):
            bins = rows.groupby("start_time").agg(n=("y", "size"), used=("y", "sum"),
                                                  price=("log_price", "sum"))
            history[key] = (bins.index.to_numpy(), np.r_[0, bins.n.cumsum()],
                            np.r_[0, bins.used.cumsum()], np.r_[0, bins.price.cumsum()])
        cols = np.full((len(query), 3), np.nan, dtype=np.float32)
        cols[:, 0] = 0
        for j, (key, stamp, price) in enumerate(zip(
                frame.iloc[query][list(keys)].itertuples(index=False, name=None),
                frame.start_time.iloc[query], all_price[query], strict=True)):
            if key in history:
                times, n, used, prices = history[key]
                i = np.searchsorted(times, stamp, side="left")
                cols[j, 0] = np.log1p(n[i])
                if n[i]:
                    cols[j, 1] = (used[i] + 2) / (n[i] + 4)
                    cols[j, 2] = price - prices[i] / n[i]
        blocks.append(cols)
    return np.column_stack(blocks)


def metrics(y, p, base, sellers):
    a = (p >= .5) == y
    b = (base >= .5) == y
    d = a.astype(int) - b
    blocks = pd.Series(d).groupby(sellers).sum().to_numpy()
    rng = np.random.default_rng(818)
    boots = [rng.choice(blocks, len(blocks)).sum() for _ in range(1200)]
    return dict(n=len(y), hits=int(a.sum()), baseline_hits=int(b.sum()), net=int(d.sum()),
                fixes=int((d > 0).sum()), breaks=int((d < 0).sum()),
                logloss=float(log_loss(y, p)), baseline_logloss=float(log_loss(y, base)),
                seller_q05=float(np.quantile(boots, .05)), seller_q95=float(np.quantile(boots, .95)))


def main():
    archive = Path(sys.argv[1])
    with zipfile.ZipFile(archive) as z, z.open("train_data.jsonlines") as f:
        df = pd.DataFrame(parser.parse_item_record(json.loads(row), True, i) for i, row in enumerate(f))
    assert len(df) == 70000 and df.id.is_unique
    pairs = pd.read_csv(OLD / "line3_today/pairs.csv")
    val = pairs.row.to_numpy()
    fit = np.setdiff1d(np.arange(len(df)), val)
    assert df.id.iloc[val].reset_index(drop=True).equals(pairs.id)
    tr = ag.precompute_item_intrinsic_arrays(df)
    ag.compute_causal_seller_features = causal_features([tr["start_time"][val]])
    inner = list(StratifiedGroupKFold(4, shuffle=True, random_state=61111).split(
        fit, tr["y"][fit], tr["seller_id"][fit]))
    mats = ag.build_fold_matrices(fit, [val], inner, tr, [tr],
                                  OLD / "line3_today" / f"text_{len(fit)}.npz")
    hist = add_history(df, fit, np.r_[fit, val])
    print("history ready", flush=True)
    model = CatBoostClassifier(**FROZEN_SETTINGS)
    model.fit(np.column_stack([mats["X_m1_seen_causal_T"], hist[:len(fit)]]), mats["y_T"])
    p = model.predict_proba(np.column_stack([mats["X_m1_seen_causal_evals"][0], hist[len(fit):]]))[:, 1]
    reference_path = OLD / "entrega_ap/ap_quinta_catboost_probabilities.npz"
    with np.load(reference_path) as z:
        base = z["val"]
    y = pairs.y.to_numpy()
    assert len(base) == len(p) == len(y)
    seen = pairs.seen.to_numpy()
    sellers = pairs.seller_id.to_numpy()
    report = {"source_rows": len(df), "fit_rows": len(fit), "confirm_development_rows": len(val),
              "baseline": "same split, cached text OOF + same CatBoost parameters, without new history",
              "history_columns": ["seller x category", "seller x brand", "seller x listing type"],
              "all": metrics(y, p, base, sellers),
              "seen": metrics(y[seen], p[seen], base[seen], sellers[seen]),
              "new": metrics(y[~seen], p[~seen], base[~seen], sellers[~seen]),
              "vs_line3": metrics(y, p, pairs.m1.to_numpy(), sellers)}
    out = Path("seller_category_result")
    out.mkdir(exist_ok=True)
    checkpoint = out / "model.cbm"
    model.save_model(str(checkpoint))
    pp = out / "pairs_private.csv"
    pairs[["id", "seen", "y"]].assign(p_control=base, p_enriched=p).to_csv(pp, index=False)
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    report.update(pairs_sha256=sha(pp), model_sha256=sha(checkpoint),
                  archive_sha256=sha(archive), script_sha256=sha(Path(__file__)),
                  baseline_npz_sha256=sha(reference_path), 
                  disagreements_id_sha256=hashlib.sha256(
                      "\n".join(sorted(pairs.id[(p >= .5) != (base >= .5)])).encode()).hexdigest())
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
