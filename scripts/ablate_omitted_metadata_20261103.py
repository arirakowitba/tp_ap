"""Fresh paired ablation of omitted listing metadata vs strong historical baseline."""
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ablate_attr_value_vs_historical_text import make_te, text_oof
from scripts.train_deep_text_20261101 import split_testlike
from src.revision_submissions.deep_catboost_nbsvm_ablation_20261029 import extract_deep_features

SEED = int(os.environ.get("AP_SEED", "20261134"))
ZIP = Path(os.environ["AP_ZIP"])
OUT = ROOT / "results/omitted_metadata"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    with zipfile.ZipFile(ZIP) as z:
        rows = [json.loads(line) for line in z.open("train_data.jsonlines")]
        test = [json.loads(line) for line in z.open("test_data.jsonlines")]
    assert len(rows) == 70000 and len(test) == 30000
    for key in ("status", "descriptions", "secure_thumbnail"):
        assert all(key in row for row in test)
    y = np.array([r["condition"] == "used" for r in rows], dtype=np.int8)
    seller = np.array([str(r["seller_id"]) for r in rows])
    item_id = np.array([str(r["id"]) for r in rows])
    fit, tune, confirm = split_testlike(seller, SEED)
    base, cat, titles, _, combos = extract_deep_features(rows)
    make_te(base, cat, y, fit, tune, confirm, seller)
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=120000,
                           sublinear_tf=True, strip_accents="unicode", lowercase=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                           max_features=150000, sublinear_tf=True)
    xf = sp.hstack([word.fit_transform([combos[i] for i in fit]),
                    char.fit_transform([titles[i] for i in fit])], format="csr")
    xt = sp.hstack([word.transform([combos[i] for i in tune]),
                    char.transform([titles[i] for i in tune])], format="csr")
    xv = sp.hstack([word.transform([combos[i] for i in confirm]),
                    char.transform([titles[i] for i in confirm])], format="csr")
    oof, pt, pv = text_oof(xf, xt, xv, y, fit, seller, 3.)
    base["txt_sparse_p"] = np.nan
    for ix, p in ((fit, oof), (tune, pt), (confirm, pv)):
        base.loc[ix, "txt_sparse_p"] = p
    base["seller_id_cat"] = cat.seller_id.astype(str)
    cand = base.copy()
    cand["status"] = [str(r.get("status") or "") for r in rows]
    n_desc = np.array([len(r.get("descriptions") or []) for r in rows], dtype=np.float32)
    cand["has_description"] = (n_desc > 0).astype(np.float32)
    cand["n_descriptions"] = n_desc
    cand["secure_thumb_missing"] = np.array([not bool(r.get("secure_thumbnail"))
                                              for r in rows], dtype=np.float32)
    cats = ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode",
            "seller_id_cat"]
    probs, iterations = {}, {}
    for name, frame, columns in (("historical", base, cats),
                                  ("metadata", cand, cats + ["status"])):
        cb = CatBoostClassifier(iterations=900, depth=6, learning_rate=.07,
                                l2_leaf_reg=6., random_seed=SEED, thread_count=7,
                                verbose=False, loss_function="Logloss")
        cb.fit(frame.iloc[fit], y[fit], cat_features=columns,
               eval_set=(frame.iloc[tune], y[tune]),
               early_stopping_rounds=100, use_best_model=True)
        probs[name] = (cb.predict_proba(frame.iloc[tune])[:, 1],
                       cb.predict_proba(frame.iloc[confirm])[:, 1])
        iterations[name] = cb.best_iteration_
        print(name, cb.best_iteration_, flush=True)
    grid = np.arange(.44, .561, .005)
    thresholds = {name: float(max(grid, key=lambda t: accuracy_score(y[tune], p[0] >= t)))
                  for name, p in probs.items()}
    pred = {name: p[1] >= thresholds[name] for name, p in probs.items()}
    delta = (pred["metadata"] == y[confirm]).astype(int) - \
            (pred["historical"] == y[confirm]).astype(int)
    seen = np.isin(seller[confirm], seller[fit])
    pairs = pd.DataFrame({"item_id": item_id[confirm], "seller_id": seller[confirm],
                          "is_seen": seen.astype(int), "y_true": y[confirm],
                          "p_historical": probs["historical"][1],
                          "p_metadata": probs["metadata"][1],
                          "pred_historical": pred["historical"].astype(int),
                          "pred_metadata": pred["metadata"].astype(int), "delta": delta})
    path = OUT / f"paired_{SEED}.csv"
    pairs.to_csv(path, index=False)
    by_seller = pairs.groupby("seller_id").delta.sum().to_numpy()
    rng = np.random.default_rng(SEED)
    boot = np.array([by_seller[rng.integers(len(by_seller), size=len(by_seller))].sum()
                     for _ in range(3000)])
    report = {"seed": SEED, "fit_tune_confirm": [len(fit), len(tune), len(confirm)],
              "input_sha256": hashlib.sha256(ZIP.read_bytes()).hexdigest(),
              "confirm_ids_sha256": hashlib.sha256(item_id[confirm].astype("U").tobytes()).hexdigest(),
              "best_iterations": iterations, "tune_thresholds": thresholds,
              "accuracy": {k: float(accuracy_score(y[confirm], v)) for k, v in pred.items()},
              "logloss": {k: float(log_loss(y[confirm], p[1])) for k, p in probs.items()},
              "net": int(delta.sum()),
              "fixes_breaks": [int((delta == 1).sum()), int((delta == -1).sum())],
              "common05_net": int(((probs["metadata"][1] >= .5) == y[confirm]).sum() -
                                  ((probs["historical"][1] >= .5) == y[confirm]).sum()),
              "seen_new_net": {str(i): int(delta[seen == bool(i)].sum()) for i in (0, 1)},
              "seller_bootstrap_ci95": np.quantile(boot, [.025, .975]).tolist(),
              "paired_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT / f"report_{SEED}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
