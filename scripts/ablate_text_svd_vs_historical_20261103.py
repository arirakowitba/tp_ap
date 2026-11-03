"""Add train-only latent word-text components to strong text+seller CatBoost.

Seller-group fit/tune/confirm, matched to the historical baseline by item ID.
SVD is unsupervised and fitted exclusively on outer fit; supervised sparse text
probabilities remain seller-fold OOF on fit. No Kaggle labels or test input.
"""
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
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ablate_attr_value_vs_historical_text import make_te, text_oof
from scripts.train_deep_text_20261101 import split_testlike
from src.revision_submissions.deep_catboost_nbsvm_ablation_20261029 import extract_deep_features

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20261109
COMPONENTS = int(sys.argv[2]) if len(sys.argv) > 2 else 32
ZIP = Path(os.environ["AP_ZIP"])
OUT = ROOT / "results" / "text_svd_ablation"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    with zipfile.ZipFile(ZIP) as z:
        rows = [json.loads(line) for line in z.open("train_data.jsonlines")]
    assert len(rows) == 70000
    y = np.array([r["condition"] == "used" for r in rows], dtype=np.int8)
    sellers = np.array([str(r["seller_id"]) for r in rows])
    ids = np.array([str(r["id"]) for r in rows])
    fit, tune, confirm = split_testlike(sellers, SEED)
    assert not (set(ids[fit]) & set(ids[tune]) or set(ids[fit]) & set(ids[confirm])
                or set(ids[tune]) & set(ids[confirm]))
    a, cat, titles, _, combos = extract_deep_features(rows)
    make_te(a, cat, y, fit, tune, confirm, sellers)
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=120000,
                           sublinear_tf=True, strip_accents="unicode", lowercase=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                           max_features=150000, sublinear_tf=True)
    xw_fit = word.fit_transform([combos[i] for i in fit])
    xw_tune = word.transform([combos[i] for i in tune])
    xw_confirm = word.transform([combos[i] for i in confirm])
    xc_fit = char.fit_transform([titles[i] for i in fit])
    xc_tune = char.transform([titles[i] for i in tune])
    xc_confirm = char.transform([titles[i] for i in confirm])
    xf = sp.hstack([xw_fit, xc_fit], format="csr")
    xt = sp.hstack([xw_tune, xc_tune], format="csr")
    xv = sp.hstack([xw_confirm, xc_confirm], format="csr")
    oof, p_tune_text, p_confirm_text = text_oof(xf, xt, xv, y, fit, sellers, 3.)
    a["txt_sparse_p"] = np.nan
    for idx, pred in zip((fit, tune, confirm), (oof, p_tune_text, p_confirm_text)):
        a.loc[idx, "txt_sparse_p"] = pred
    a["seller_id_cat"] = cat.seller_id.astype(str)
    b = a.copy()
    svd = TruncatedSVD(n_components=COMPONENTS, n_iter=5, random_state=SEED)
    z_fit = svd.fit_transform(xw_fit)
    z_tune = svd.transform(xw_tune)
    z_confirm = svd.transform(xw_confirm)
    for j in range(COMPONENTS):
        col = f"text_svd_{j:02d}"
        b[col] = np.nan
        b.loc[fit, col] = z_fit[:, j].astype(np.float32)
        b.loc[tune, col] = z_tune[:, j].astype(np.float32)
        b.loc[confirm, col] = z_confirm[:, j].astype(np.float32)
    cats = ["category_id", "listing_type_id", "buying_mode", "state",
            "ship_mode", "seller_id_cat"]
    model = CatBoostClassifier(iterations=900, depth=6, learning_rate=.07,
                               l2_leaf_reg=6., random_seed=SEED, thread_count=7,
                               verbose=False, loss_function="Logloss")
    model.fit(b.iloc[fit], y[fit], cat_features=cats,
              eval_set=(b.iloc[tune], y[tune]), early_stopping_rounds=100,
              use_best_model=True)
    pt = model.predict_proba(b.iloc[tune])[:, 1]
    pc = model.predict_proba(b.iloc[confirm])[:, 1]
    grid = np.arange(.44, .561, .005)
    threshold = max(grid, key=lambda t: accuracy_score(y[tune], pt >= t))
    pred = (pc >= threshold).astype(int)
    prior_path = ROOT / "results" / "attr_ablation" / f"paired_{SEED}_seller.csv"
    if not prior_path.exists():
        prior_path = ROOT / "results" / "attr_ablation" / f"paired_{SEED}_deep_vs_hist_seller.csv"
    prior = pd.read_csv(prior_path).set_index("item_id")
    assert len(prior) == len(confirm) and set(prior.index) == set(ids[confirm])
    prior = prior.loc[ids[confirm]].reset_index()
    y_col = "y_true_h" if "y_true_h" in prior else "y_true"
    seller_col = "seller_id_h" if "seller_id_h" in prior else "seller_id"
    assert np.array_equal(prior[y_col].to_numpy(), y[confirm])
    assert np.array_equal(prior[seller_col].astype(str).to_numpy(), sellers[confirm])
    baseline_p = prior.p_historical.to_numpy(dtype=float)
    baseline_pred = prior.pred_historical.to_numpy(dtype=int)
    delta = (pred == y[confirm]).astype(int) - (baseline_pred == y[confirm]).astype(int)
    seen = np.isin(sellers[confirm], sellers[fit]).astype(int)
    pairs = pd.DataFrame({"item_id": ids[confirm], "seller_id": sellers[confirm],
                          "is_seen": seen, "y_true": y[confirm],
                          "p_historical": baseline_p, "p_plus_svd": pc,
                          "pred_historical": baseline_pred, "pred_plus_svd": pred,
                          "delta_plus_svd": delta})
    path = OUT / f"paired_{SEED}_svd{COMPONENTS}.csv"
    pairs.to_csv(path, index=False)
    by = pairs.groupby("seller_id").delta_plus_svd.sum().to_numpy()
    rng = np.random.default_rng(SEED)
    boot = np.array([by[rng.integers(0, len(by), len(by))].sum() for _ in range(3000)])
    report = {"seed": SEED, "components": COMPONENTS,
              "fit_tune_confirm": [len(fit), len(tune), len(confirm)],
              "train_zip_sha256": hashlib.sha256(ZIP.read_bytes()).hexdigest(),
              "prior_pairs_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
              "confirm_ids_sha256": hashlib.sha256(ids[confirm].astype("U").tobytes()).hexdigest(),
              "svd_explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
              "best_iteration": model.best_iteration_, "tune_threshold": float(threshold),
              "baseline_confirm_acc": float(accuracy_score(y[confirm], baseline_pred)),
              "candidate_confirm_acc": float(accuracy_score(y[confirm], pred)),
              "baseline_logloss": float(log_loss(y[confirm], baseline_p)),
              "candidate_logloss": float(log_loss(y[confirm], pc)),
              "net": int(delta.sum()), "fixes": int((delta == 1).sum()),
              "breaks": int((delta == -1).sum()),
              "common05_net": int(((pc >= .5) == y[confirm]).sum()
                                  - ((baseline_p >= .5) == y[confirm]).sum()),
              "seller_bootstrap_ci95": np.quantile(boot, [.025, .975]).tolist(),
              "seen_new": {str(k): {"n": int((seen == k).sum()),
                                        "net": int(delta[seen == k].sum())} for k in (0, 1)},
              "paired_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "caveat": "Both confirm splits were reused; pre-split unsupervised aggregates are shared with baseline."}
    (OUT / f"report_{SEED}_svd{COMPONENTS}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
