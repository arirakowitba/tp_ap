#!/usr/bin/env python3
"""Leakage-safe semantic NB-SVM challenger on the frozen AG-0094 outer splits.

The script never reads test data, public scores, submissions, solver outputs, or pseudo-labels.
It selects one of three predeclared regularization values on tune, freezes it, then evaluates
confirm once. Baseline probabilities must come from the matching AG-0094 two-stage pair files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from scripts.train_codex_raw_nested_text_histgb_20261104 import split, digest

CS = (0.35, 0.75, 1.50)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_text(row: dict) -> str:
    attrs = " ".join(
        str(a.get("value_name") or a.get("name") or "")
        for a in (row.get("attributes") or [])
        if isinstance(a, dict)
    )
    tags = " ".join(map(str, row.get("tags") or []))
    return " ".join(
        [
            str(row.get("title") or ""),
            "WARRANTY", str(row.get("warranty") or ""),
            "ATTR", attrs,
            "TAGS", tags,
        ]
    )


def nb_ratio(x: sparse.csr_matrix, y: np.ndarray) -> np.ndarray:
    pos = np.asarray(x[y == 1].sum(axis=0)).ravel() + 1.0
    neg = np.asarray(x[y == 0].sum(axis=0)).ravel() + 1.0
    pos /= pos.sum()
    neg /= neg.sum()
    return np.log(pos / neg).astype(np.float32)


def paired(y, p, p_base, sellers, seen) -> dict:
    pred = p >= 0.5
    base = p_base >= 0.5
    good = pred == y
    good_base = base == y
    delta = good.astype(int) - good_base.astype(int)
    fixes = int(np.sum(good & ~good_base))
    breaks = int(np.sum(~good & good_base))

    uniq = np.unique(sellers)
    by_seller = np.asarray([delta[sellers == g].sum() for g in uniq], dtype=float)
    rng = np.random.default_rng(20261107)
    boots = np.empty(2000, dtype=float)
    for b in range(len(boots)):
        boots[b] = by_seller[rng.integers(0, len(by_seller), len(by_seller))].sum()

    return {
        "n": int(len(y)),
        "hits": int(good.sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "logloss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
        "baseline_hits": int(good_base.sum()),
        "net": int(delta.sum()),
        "fixes": fixes,
        "breaks": breaks,
        "seen_net": int(delta[seen].sum()),
        "new_net": int(delta[~seen].sum()),
        "seller_bootstrap_ci95": [float(x) for x in np.quantile(boots, [0.025, 0.975])],
    }


def load_baseline(path: Path, ids_expected: np.ndarray) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"item_id", "seller_id", "y_true", "p_cb_nested_nb", "p_blend_native_nb"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df["item_id"] = df["item_id"].astype(str)
    if set(df["item_id"]) != set(ids_expected.astype(str)) or len(df) != len(ids_expected):
        raise ValueError(f"{path}: IDs do not match frozen outer split")
    return df.set_index("item_id").loc[ids_expected.astype(str)].reset_index()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip", type=Path)
    ap.add_argument("baseline_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pure", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    with zipfile.ZipFile(args.zip) as zf:
        rows = [json.loads(line) for line in zf.open("train_data.jsonlines")]
    if len(rows) != 70000:
        raise ValueError(f"expected 70000 train rows, got {len(rows)}")

    y = np.asarray([int(r["condition"] == "used") for r in rows], dtype=np.int8)
    sellers = np.asarray([str(r["seller_id"]) for r in rows])
    ids = np.asarray([str(r["id"]) for r in rows])
    if len(set(ids)) != len(ids):
        raise ValueError("train item IDs are not unique")

    fi, tu, co = split(sellers, args.seed, args.pure)
    tune_base = load_baseline(args.baseline_dir / "tune_pairs.csv", ids[tu])
    conf_base = load_baseline(args.baseline_dir / "pairs.csv", ids[co])
    if not np.array_equal(tune_base["y_true"].to_numpy(), y[tu]):
        raise ValueError("tune labels disagree with source train")
    if not np.array_equal(conf_base["y_true"].to_numpy(), y[co]):
        raise ValueError("confirm labels disagree with source train")

    text = np.asarray([build_text(r) for r in rows], dtype=object)
    word = HashingVectorizer(
        n_features=2**18, alternate_sign=False, binary=True, norm=None,
        ngram_range=(1, 2), lowercase=True, dtype=np.float32,
    )
    char = HashingVectorizer(
        n_features=2**18, alternate_sign=False, binary=True, norm=None,
        analyzer="char_wb", ngram_range=(3, 5), lowercase=True, dtype=np.float32,
    )
    counts = sparse.hstack([word.transform(text), char.transform(text)], format="csr")
    tfidf = TfidfTransformer(sublinear_tf=True)
    x_fit = tfidf.fit_transform(counts[fi])
    x_tune = tfidf.transform(counts[tu])
    ratio = nb_ratio(x_fit, y[fi])
    x_fit = x_fit.multiply(ratio).tocsr()
    x_tune = x_tune.multiply(ratio).tocsr()

    tune_rows = []
    tune_probs = {}
    for c in CS:
        model = LogisticRegression(
            C=c, solver="liblinear", max_iter=350, random_state=args.seed,
            class_weight=None,
        )
        model.fit(x_fit, y[fi])
        p = model.predict_proba(x_tune)[:, 1]
        tune_probs[c] = p
        tune_rows.append(
            {
                "C": c,
                "hits": int(((p >= 0.5) == y[tu]).sum()),
                "accuracy": float(accuracy_score(y[tu], p >= 0.5)),
                "logloss": float(log_loss(y[tu], np.clip(p, 1e-7, 1 - 1e-7))),
            }
        )

    grid = pd.DataFrame(tune_rows).sort_values(
        ["hits", "logloss", "C"], ascending=[False, True, True]
    )
    selected_c = float(grid.iloc[0]["C"])
    # Selection is now frozen. Confirm features and predictions are created only after this line.
    x_conf = tfidf.transform(counts[co]).multiply(ratio).tocsr()
    del counts
    selected = LogisticRegression(
        C=selected_c, solver="liblinear", max_iter=350,
        random_state=args.seed, class_weight=None,
    )
    selected.fit(x_fit, y[fi])
    p_conf = selected.predict_proba(x_conf)[:, 1]
    p_tune = tune_probs[selected_c]

    seen_tu = np.isin(sellers[tu], sellers[fi])
    seen_co = np.isin(sellers[co], sellers[fi])
    baselines = {
        "cb_nested_nb": ("p_cb_nested_nb",),
        "blend_native_nb": ("p_blend_native_nb",),
    }
    metrics = {"tune": {}, "confirm": {}}
    for name, (col,) in baselines.items():
        metrics["tune"][name] = paired(
            y[tu], p_tune, tune_base[col].to_numpy(), sellers[tu], seen_tu
        )
        metrics["confirm"][name] = paired(
            y[co], p_conf, conf_base[col].to_numpy(), sellers[co], seen_co
        )

    grid.to_csv(args.out / "tune_grid.csv", index=False)
    pd.DataFrame(
        {
            "item_id": ids[tu], "seller_id": sellers[tu], "is_seen": seen_tu.astype(int),
            "y_true": y[tu], "p_semantic_nbsvm": p_tune,
            "p_cb_nested_nb": tune_base["p_cb_nested_nb"],
            "p_blend_native_nb": tune_base["p_blend_native_nb"],
        }
    ).to_csv(args.out / "tune_pairs.csv", index=False)
    pd.DataFrame(
        {
            "item_id": ids[co], "seller_id": sellers[co], "is_seen": seen_co.astype(int),
            "y_true": y[co], "p_semantic_nbsvm": p_conf,
            "p_cb_nested_nb": conf_base["p_cb_nested_nb"],
            "p_blend_native_nb": conf_base["p_blend_native_nb"],
        }
    ).to_csv(args.out / "pairs.csv", index=False)

    report = {
        "method": "train_only_semantic_nbsvm",
        "seed": args.seed,
        "pure": args.pure,
        "predeclared_C": list(CS),
        "selection_rule": "max tune hits, then min tune logloss, then min C",
        "selected_C": selected_c,
        "split_sizes": {"fit": len(fi), "tune": len(tu), "confirm": len(co)},
        "split_hashes": {
            "fit": digest(ids[fi]), "tune": digest(ids[tu]), "confirm": digest(ids[co])
        },
        "baseline_source": str(args.baseline_dir),
        "baseline_tune_pairs_sha256": sha256(args.baseline_dir / "tune_pairs.csv"),
        "baseline_confirm_pairs_sha256": sha256(args.baseline_dir / "pairs.csv"),
        "metrics": metrics,
        "code_sha256": sha256(Path(__file__)),
        "tune_grid_sha256": sha256(args.out / "tune_grid.csv"),
        "tune_pairs_sha256": sha256(args.out / "tune_pairs.csv"),
        "confirm_pairs_sha256": sha256(args.out / "pairs.csv"),
        "duration_seconds": time.time() - started,
        "forbidden_inputs": [],
    }
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
