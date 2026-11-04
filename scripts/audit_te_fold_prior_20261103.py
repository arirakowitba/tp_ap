"""Quantify label leakage from using the full-fit mean inside OOF TE folds.

This audit does not train a model. It compares the historical implementation,
whose smoothing prior is y[fit].mean(), with a fold-clean implementation whose
prior is y[fit[train_fold]].mean(). Tune/confirm maps are unchanged.
"""
import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def split_testlike(sellers, seed):
    rng = np.random.default_rng(seed)
    unique = np.unique(sellers)
    rng.shuffle(unique)
    k = int(.05 * len(unique))
    tune_new, confirm_new = set(unique[:k]), set(unique[k:2 * k])
    grouped = {}
    for i, seller in enumerate(sellers):
        grouped.setdefault(seller, []).append(i)
    eligible = []
    for seller in unique[2 * k:]:
        ix = np.asarray(grouped[seller], dtype=int)
        rng.shuffle(ix)
        eligible.extend(ix[1:])
    eligible = np.asarray(eligible, dtype=int)
    rng.shuffle(eligible)
    n_seen = min(9000, len(eligible) // 2)
    tune = np.r_[np.flatnonzero(np.isin(sellers, list(tune_new))), eligible[:n_seen]]
    confirm = np.r_[np.flatnonzero(np.isin(sellers, list(confirm_new))), eligible[n_seen:2*n_seen]]
    fit = np.setdiff1d(np.arange(len(sellers)), np.r_[tune, confirm])
    return fit, tune, confirm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_path", type=Path,
                    help="Competition ZIP or extracted train_data.jsonlines")
    ap.add_argument("--seed", type=int, default=20261144)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.data_path.suffix == ".zip":
        with zipfile.ZipFile(args.data_path) as zf:
            rows = [json.loads(line) for line in zf.open("train_data.jsonlines")]
    else:
        with args.data_path.open("r", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
    y = np.asarray([int(r["condition"] == "used") for r in rows], dtype=np.int8)
    sellers = np.asarray([str(r["seller_id"]) for r in rows])
    cat = pd.DataFrame({
        "category_id": [str(r.get("category_id") or "") for r in rows],
        "listing_type_id": [str(r.get("listing_type_id") or "") for r in rows],
        "state": [str(((r.get("seller_address") or {}).get("state") or {}).get("id") or "") for r in rows],
    })
    fit, tune, confirm = split_testlike(sellers, args.seed)
    global_prior = float(y[fit].mean())
    gkf = GroupKFold(n_splits=4)
    result = {"seed": args.seed, "sizes": [len(fit), len(tune), len(confirm)],
              "global_fit_prior": global_prior, "columns": {}}
    for col in cat:
        old = np.full(len(fit), global_prior, dtype=np.float64)
        clean = np.full(len(fit), global_prior, dtype=np.float64)
        fold_priors = []
        for tr, va in gkf.split(fit, y[fit], sellers[fit]):
            tr_rows, va_rows = fit[tr], fit[va]
            fold_prior = float(y[tr_rows].mean())
            fold_priors.append(fold_prior)
            stats = pd.DataFrame({"cat": cat.iloc[tr_rows][col].to_numpy(),
                                  "y": y[tr_rows]}).groupby("cat").y.agg(["count", "mean"])
            old_map = (stats["count"] * stats["mean"] + 10. * global_prior) / (stats["count"] + 10.)
            clean_map = (stats["count"] * stats["mean"] + 10. * fold_prior) / (stats["count"] + 10.)
            vals = cat.iloc[va_rows][col]
            old[va] = vals.map(old_map).fillna(global_prior)
            clean[va] = vals.map(clean_map).fillna(fold_prior)
        diff = np.abs(old - clean)
        result["columns"][col] = {
            "fold_train_priors": fold_priors,
            "rows_changed": int((diff > 1e-12).sum()),
            "mean_abs_change": float(diff.mean()),
            "p95_abs_change": float(np.quantile(diff, .95)),
            "max_abs_change": float(diff.max()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
