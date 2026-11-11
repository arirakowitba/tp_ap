"""Refit T98 Line 3 with timestamp-bounded seller histories and fixed settings."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from train_ap_today import hash_file

sys.path.insert(0, str(Path(__file__).resolve().parent / "ag_source" / "scripts"))
import prepare_t98_shared_splits_and_cache as parser
import t98_line3_lightgbm_enriched as ag


def causal_features(eval_times):
    def compute(sellers, y, stamps, rows, strength, prior, eval_sellers):
        observed = pd.DataFrame({"seller": sellers, "stamp": stamps, "y": y})
        histories = {}
        for seller, frame in observed.groupby("seller", sort=False):
            block = frame.groupby("stamp").y.agg(["count", "sum"])
            histories[seller] = (block.index.to_numpy(), np.r_[0, block["count"].cumsum()],
                                 np.r_[0, block["sum"].cumsum()])

        def lookup(keys, times):
            means = np.full(len(keys), prior, dtype=np.float32)
            counts = np.zeros(len(keys), dtype=np.float32)
            for i, (seller, stamp) in enumerate(zip(keys, times, strict=True)):
                if seller in histories:
                    times0, ns, ys = histories[seller]
                    j = np.searchsorted(times0, stamp, side="left")
                    means[i] = (ys[j] + strength * prior) / (ns[j] + strength)
                    counts[i] = np.log1p(ns[j])
            return means, counts

        fit_mean, fit_count = lookup(sellers, stamps)
        eval_history = [lookup(s, t) for s, t in zip(eval_sellers, eval_times, strict=True)]
        return fit_mean, fit_count, [m for m, _ in eval_history], [c for _, c in eval_history]
    return compute


def model_pass(train, target, fit, indices, dest, seeds):
    ag.compute_causal_seller_features = causal_features([target["start_time"][indices]])
    split = list(StratifiedGroupKFold(4, shuffle=True, random_state=61111).split(
        fit, train["y"][fit], train["seller_id"][fit]))
    cache = dest / ("text_" + str(len(fit)) + ".npz")
    mats = ag.build_fold_matrices(fit, [indices], split, train, [target], cache)
    print("Line 3 matrices ready", len(fit), flush=True)
    hp = {"num_leaves": 55, "colsample_bytree": .7, "min_child_samples": 40}
    base_hp = {"num_leaves": 55, "colsample_bytree": .8, "min_child_samples": 20}
    cols = {}
    for prefix, params in [("m0", base_hp), ("m1", hp)]:
        for regime in ("unseen", "seen_causal"):
            key = f"X_{prefix}_{regime}"
            cols[prefix + "_" + regime] = ag.fit_predict_lgb_bagged(
                mats[key + "_T"], mats["y_T"], mats[key + "_evals"], params, seeds, 900)[0]
            print("Line 3 fit done", prefix, regime, len(fit), flush=True)
    routed = np.isin(target["seller_id"][indices], train["seller_id"][fit])
    for name in ("m0", "m1"):
        cols[name] = np.where(routed, cols[name + "_seen_causal"], cols[name + "_unseen"])
    cols["seen"] = routed
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 2026, 777])
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    with zipfile.ZipFile(args.archive) as archive:
        def read(kind, expected):
            with archive.open(kind + "_data.jsonlines") as f:
                frame = pd.DataFrame(parser.parse_item_record(json.loads(l), kind == "train", i)
                                     for i, l in enumerate(f))
            assert len(frame) == expected and frame.id.is_unique
            return frame
        tr = read("train", 70000)
        te = read("test", 30000)
    assert not set(tr.id) & set(te.id)
    train, test = map(ag.precompute_item_intrinsic_arrays, (tr, te))
    all_train = np.arange(len(tr))
    rng = np.random.default_rng(20261111)
    sellers = np.unique(tr.seller_id)
    rng.shuffle(sellers)
    cold = tr.seller_id.isin(sellers[:len(sellers) // 5])
    eligible = tr.loc[~cold]
    last = eligible.groupby("seller_id").start_time.transform("max")
    first = eligible.groupby("seller_id").start_time.transform("min")
    warm = eligible.index[(last == eligible.start_time) & (last > first)]
    val = np.sort(np.r_[np.flatnonzero(cold), warm])
    fit = np.setdiff1d(all_train, val)
    assert len(np.unique(val)) == len(val)
    assert not set(tr.seller_id[cold]) & set(tr.seller_id.iloc[fit])
    p = model_pass(train, train, fit, val, args.out, args.seeds)
    pairs = tr.iloc[val][["row", "id", "seller_id", "category_id", "y"]].assign(**p)
    pairs.to_csv(args.out / "pairs.csv", index=False)
    print(pairs.groupby("seen").apply(lambda d: pd.Series({"n": len(d), "baseline": ((d.m0 >= .5) == d.y).sum(),
                                                            "enriched": ((d.m1 >= .5) == d.y).sum()})), flush=True)
    p = model_pass(train, test, all_train, np.arange(len(te)), args.out, args.seeds)
    out = pd.DataFrame({"ID": np.arange(1, len(te) + 1), "item": te.id, **p})
    out.to_csv(args.out / "test_probs.csv", index=False)
    report = {"archive_sha256": hash_file(args.archive), "runner_sha256": hash_file(Path(__file__)),
              "ag_line3_sha256": hash_file(Path(ag.__file__)), "seeds": args.seeds,
              "pairs_sha256": hash_file(args.out / "pairs.csv"),
              "test_probs_sha256": hash_file(args.out / "test_probs.csv")}
    (args.out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
