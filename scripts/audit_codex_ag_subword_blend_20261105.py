"""Honest validation of AG's 80k TF-IDF subword LR as a blend component.

Vocabulary and supervised LR are fit on outer-fit only. Blend weight and
threshold are selected on tune only, then reported once on confirm against
the frozen strong depth6 native+NB prediction pairs.
"""
import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from train_raw_nested_text_histgb import split


def digest(values):
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def paired(y, candidate, baseline):
    dc = (candidate == y).astype(np.int8) - (baseline == y).astype(np.int8)
    return {
        "hits": int((candidate == y).sum()),
        "net": int(dc.sum()),
        "fixes": int((dc == 1).sum()),
        "breaks": int((dc == -1).sum()),
        "disagreements": int((candidate != baseline).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("strong_dir")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=20261104)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with zipfile.ZipFile(args.zip) as z:
        rows = [json.loads(line) for line in z.open("train_data.jsonlines")]
    y = np.asarray([int(r["condition"] == "used") for r in rows])
    sellers = np.asarray([str(r["seller_id"]) for r in rows])
    item_ids = np.asarray([str(r["id"]) for r in rows])
    text = np.asarray([
        str(r.get("title") or "") + " " + " ".join(map(str, r.get("tags") or []))
        for r in rows
    ], dtype=object)
    fi, tu, co = split(sellers, args.seed, False)

    word = TfidfVectorizer(
        ngram_range=(1, 2), max_features=35000, min_df=3, sublinear_tf=True
    )
    char = TfidfVectorizer(
        ngram_range=(3, 5), analyzer="char_wb", max_features=45000,
        min_df=5, sublinear_tf=True
    )
    xw_fit = word.fit_transform(text[fi])
    xc_fit = char.fit_transform(text[fi])
    x_fit = hstack([xw_fit, xc_fit], format="csr")
    x_tu = hstack([word.transform(text[tu]), char.transform(text[tu])], format="csr")
    x_co = hstack([word.transform(text[co]), char.transform(text[co])], format="csr")

    lr = LogisticRegression(
        C=1.5, max_iter=1000, solver="saga", random_state=args.seed, tol=1e-4
    )
    lr.fit(x_fit, y[fi])
    p_tu_lr = lr.predict_proba(x_tu)[:, 1]
    p_co_lr = lr.predict_proba(x_co)[:, 1]

    strong = Path(args.strong_dir)
    tp = pd.read_csv(strong / "tune_pairs.csv").set_index("item_id").loc[item_ids[tu]]
    cp = pd.read_csv(strong / "pairs.csv").set_index("item_id").loc[item_ids[co]]
    assert np.array_equal(tp.y_true.to_numpy(), y[tu])
    assert np.array_equal(cp.y_true.to_numpy(), y[co])
    p_tu_base = tp.p_blend_native_nb.to_numpy()
    p_co_base = cp.p_blend_native_nb.to_numpy()

    weights = np.arange(0.0, 0.401, 0.05)
    thresholds = np.arange(0.45, 0.561, 0.01)
    grid = []
    for w in weights:
        p = (1 - w) * p_tu_base + w * p_tu_lr
        for th in thresholds:
            grid.append((accuracy_score(y[tu], p >= th), -log_loss(y[tu], p), w, th))
    _, _, weight, threshold = max(grid)
    p_tu = (1 - weight) * p_tu_base + weight * p_tu_lr
    p_co = (1 - weight) * p_co_base + weight * p_co_lr

    base_th = max(
        thresholds,
        key=lambda th: accuracy_score(y[tu], p_tu_base >= th),
    )
    pred_base = p_co_base >= base_th
    pred_candidate = p_co >= threshold
    seen = np.isin(sellers[co], sellers[fi])
    result = {
        "seed": args.seed,
        "split_sizes": {"fit": len(fi), "tune": len(tu), "confirm": len(co)},
        "split_hashes": {
            "fit": digest(item_ids[fi]), "tune": digest(item_ids[tu]),
            "confirm": digest(item_ids[co]),
        },
        "text_recipe": {
            "word_features": len(word.vocabulary_), "char_features": len(char.vocabulary_),
            "lr_C": 1.5, "fit_only": True,
        },
        "standalone": {
            "tune_accuracy_050": float(accuracy_score(y[tu], p_tu_lr >= 0.5)),
            "confirm_accuracy_050": float(accuracy_score(y[co], p_co_lr >= 0.5)),
            "confirm_logloss": float(log_loss(y[co], p_co_lr)),
        },
        "selection": {
            "weight_lr": float(weight), "threshold": float(threshold),
            "baseline_threshold": float(base_th),
            "tune_accuracy": float(accuracy_score(y[tu], p_tu >= threshold)),
            "baseline_tune_accuracy": float(accuracy_score(y[tu], p_tu_base >= base_th)),
        },
        "confirm": {
            "candidate_accuracy": float(accuracy_score(y[co], pred_candidate)),
            "baseline_accuracy": float(accuracy_score(y[co], pred_base)),
            "candidate_logloss": float(log_loss(y[co], p_co)),
            "baseline_logloss": float(log_loss(y[co], p_co_base)),
            "paired": paired(y[co], pred_candidate, pred_base),
            "seen_net": int(((pred_candidate[seen] == y[co][seen]).astype(int) -
                             (pred_base[seen] == y[co][seen]).astype(int)).sum()),
            "new_net": int(((pred_candidate[~seen] == y[co][~seen]).astype(int) -
                            (pred_base[~seen] == y[co][~seen]).astype(int)).sum()),
        },
        "duration_seconds": time.time() - t0,
        "limitation": "Reused development split; one tune-selected blend evaluated on confirm.",
    }
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    pd.DataFrame({
        "item_id": item_ids[co], "seller_id": sellers[co], "is_seen": seen.astype(int),
        "y_true": y[co], "p_strong": p_co_base, "p_subword_lr": p_co_lr,
        "p_candidate": p_co,
    }).to_csv(out / "pairs.csv", index=False)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
