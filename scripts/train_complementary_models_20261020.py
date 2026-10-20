"""Grouped CV for complementary text/structured models, 20 October 2026.

The public leaderboard provides ONLY cohort-level feedback. It is used below
for diagnostics, never as a label for any test row or a training feature.
All fitted vectorizers and SVDs are scoped to each outer training fold.
"""

from pathlib import Path
import gc
import json
import re
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.sparse import hstack, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[3]
STAGE = Path(__file__).resolve().parents[1]
OUT = STAGE / "results" / "new_models_20261020"
OUT.mkdir(parents=True, exist_ok=True)
threadpool_limits(limits=4)


def say(*parts):
    print(*parts, flush=True)


def fixed_metrics(y, p):
    return {"accuracy_05": float(accuracy_score(y, p >= .5)),
            "log_loss": float(log_loss(y, p))}


def load_data():
    cache = ROOT / "notebook_work" / "ap_split_build" / "validation_cache"
    rich = ROOT / "experiment_outputs_posterior_aware_20261017" / "features_cache"
    parts = [pd.read_pickle(rich / (name + ".pkl"))
             for name in ("train", "tune", "hold", "test")]
    split = np.load(ROOT / "notebook_work" / "ap_split_build" / "attempt1_split.npz",
                    allow_pickle=True)
    base = np.load(ROOT / "experiment_outputs_posterior_transfer_20261017" /
                   "base_predictions.npz")
    dev, hold, tune = split["fit"].astype(int), split["valid"].astype(int), base["tune_rows"]
    inner = dev[~np.isin(dev, tune)]
    order = np.r_[inner, tune, hold]
    assert [len(x) for x in parts] == [39200, 9800, 21000, 30000]
    assert np.array_equal(np.sort(order), np.arange(70000))
    train = pd.concat(parts[:3], ignore_index=True).iloc[np.argsort(order)].reset_index(drop=True)
    test = parts[3].reset_index(drop=True)
    y = (pd.read_pickle(cache / "target.pkl").astype(str).to_numpy() == "used").astype(int)
    groups = np.load(ROOT / "experiment_outputs_complementary_architectures_20261017"
                     / "groups_all.npy", allow_pickle=True)
    raw = pd.read_pickle(ROOT / "experiment_outputs_warranty_text_20261017" / "raw_text.pkl")
    assert train.shape[0] == len(y) == len(groups) == 70000
    assert test.shape[0] == 30000 and len(raw) == 100000
    assert list(train.columns) == list(test.columns)
    return pd.concat([train, test], ignore_index=True), y, groups, raw


def make_structured(features, raw, train_rows, valid_rows, test_rows,
                    word_components, char_components, random_seed):
    """All statistics depending on a fit use the *outer training rows*."""
    chosen = np.r_[train_rows, valid_rows, test_rows]
    base = features.iloc[chosen].copy().reset_index(drop=True)
    cats = [c for c in base.select_dtypes(include=["object", "string", "category"]) if c != "title_text"]
    base = base.drop(columns="title_text", errors="ignore")
    base["seller_id"] = raw.seller.iloc[chosen].astype(str).to_numpy()
    base["category_id"] = raw.category.iloc[chosen].astype(str).to_numpy()
    cats += ["seller_id"]
    cats = list(dict.fromkeys(cats))
    corpus = (raw.title + " attributes " + raw.attributes
              + " warranty " + raw.warranty).iloc[chosen].reset_index(drop=True)
    for field in ("title", "warranty", "attributes"):
        values = raw[field].iloc[chosen].astype(str).reset_index(drop=True)
        for token, expr in (("used", r"\b(?:usad[oa]s?|seminuev[oa]s?|reacondicionad[oa]s?)\b"),
                            ("new", r"\b(?:nuev[oa]s?|sellad[oa]s?|sin uso)\b")):
            base[f"lex_{field}_{token}"] = values.str.count(expr).to_numpy(dtype=np.float32)
    for name, parts in (("word", word_components), ("char", char_components)):
        for k in range(parts.shape[1]):
            base[f"tfidf_{name}_svd_{k:02d}"] = parts[:, k].astype(np.float32)
    number_cols = base.select_dtypes(include=np.number).columns
    base[number_cols] = base[number_cols].replace([np.inf, -np.inf], np.nan)
    for c in cats:
        base[c] = pd.Categorical(base[c].fillna("__missing__").astype(str))
    ntr = len(train_rows); nva = len(valid_rows)
    # Let LightGBM handle NaNs; categorical vocabulary is unlabeled.
    return base.iloc[:ntr], base.iloc[ntr:ntr+nva], base.iloc[ntr+nva:], cats


def run():
    started = time.time()
    features, y, groups, raw = load_data()
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True,
                                     random_state=20261200).split(np.zeros(len(y)), y, groups))
    olddir = ROOT / "experiment_outputs_hour_20261017_2200" / "rotated_cv"
    old = (np.load(olddir / "RichDepthwise_oof_predictions.npz")["oof"]
           + np.load(olddir / "RichPlain_oof_predictions.npz")["oof"]) / 2
    oof = {"text_linear": np.full(len(y), np.nan),
           "svd_lgbm": np.full(len(y), np.nan)}
    test_preds = {name: np.zeros(30000) for name in oof}
    reports = []
    text = raw.title + " attributes " + raw.attributes + " warranty " + raw.warranty
    alltest = np.arange(70000, 100000)
    for fold, (tr, va) in enumerate(folds):
        assert not set(groups[tr]) & set(groups[va])
        say("FOLD", fold, "rows", len(tr), len(va), "elapsed", round(time.time()-started, 1))
        saved = {name: OUT / f"{name}_fold{fold}.npz" for name in oof}
        if all(path.exists() for path in saved.values()):
            metrics = {}
            for name, path in saved.items():
                prior = np.load(path)
                assert np.array_equal(prior["valid_rows"], va)
                assert len(prior["test"]) == 30000
                oof[name][va] = prior["valid"]
                test_preds[name] += prior["test"] / 3
                metrics[name] = fixed_metrics(y[va], prior["valid"])
            reports.append({"fold": fold, "rows_train": len(tr),
                            "rows_valid": len(va), "group_overlap": 0, **metrics})
            say("RESUMED", fold, metrics, "elapsed", round(time.time()-started, 1))
            continue
        wordvec = TfidfVectorizer(ngram_range=(1, 2), max_features=65000,
                                  min_df=3, sublinear_tf=True, dtype=np.float32)
        charvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                  max_features=50000, min_df=5, sublinear_tf=True,
                                  dtype=np.float32)
        sparse_data = []
        for vec in (wordvec, charvec):
            fit = vec.fit_transform(text.iloc[tr])
            valid = vec.transform(text.iloc[va])
            hold = vec.transform(text.iloc[alltest])
            sparse_data.append((fit, valid, hold))
        tf_word, tf_char = sparse_data
        numeric_cols = features.select_dtypes(include=np.number).columns.tolist()
        imp = SimpleImputer(strategy="median", add_indicator=True)
        scaler = StandardScaler(with_mean=False)
        train_num = scaler.fit_transform(imp.fit_transform(features.iloc[tr][numeric_cols]))
        valid_num = scaler.transform(imp.transform(features.iloc[va][numeric_cols]))
        test_num = scaler.transform(imp.transform(features.iloc[alltest][numeric_cols]))
        cat_data = pd.DataFrame({"seller": raw.seller.astype(str),
                                 "category": raw.category.astype(str),
                                 "listing": features.listing_type_id.astype(str)})
        oh = OneHotEncoder(handle_unknown="ignore", min_frequency=12, dtype=np.float32)
        tcat = oh.fit_transform(cat_data.iloc[tr]); vcat = oh.transform(cat_data.iloc[va]); hcat = oh.transform(cat_data.iloc[alltest])
        Xtr = hstack((tf_word[0], tf_char[0], tcat, csr_matrix(train_num)), format="csr")
        Xva = hstack((tf_word[1], tf_char[1], vcat, csr_matrix(valid_num)), format="csr")
        Xte = hstack((tf_word[2], tf_char[2], hcat, csr_matrix(test_num)), format="csr")
        line = LogisticRegression(C=0.85, solver="liblinear", max_iter=180,
                                  random_state=20261020 + fold)
        line.fit(Xtr, y[tr])
        p_line = line.predict_proba(Xva)[:, 1]
        t_line = line.predict_proba(Xte)[:, 1]
        oof["text_linear"][va] = p_line
        test_preds["text_linear"] += t_line / 3
        say("TEXT_LINEAR", fold, fixed_metrics(y[va], p_line), "elapsed", round(time.time()-started, 1))
        del Xtr, Xva, Xte, line, tcat, vcat, hcat, train_num, valid_num, test_num
        gc.collect()

        svds = []
        for prefix, (fit, valid, hold) in (("word", tf_word), ("char", tf_char)):
            reducer = TruncatedSVD(n_components=24, n_iter=3, random_state=20261020 + fold)
            a = reducer.fit_transform(fit)
            b = reducer.transform(valid)
            c = reducer.transform(hold)
            svds.append(np.vstack((a, b, c)))
            say("SVD", prefix, fold, "variance", round(float(reducer.explained_variance_ratio_.sum()), 4))
            del reducer
        svdtr, svdva, svdtest, cats = make_structured(
            features, raw, tr, va, alltest, svds[0], svds[1], 20261020 + fold)
        model = LGBMClassifier(n_estimators=620, learning_rate=.044, num_leaves=23,
                               max_depth=-1, min_child_samples=75, colsample_bytree=.87,
                               subsample=.82, subsample_freq=1, reg_lambda=12,
                               reg_alpha=1.2, max_bin=127, cat_smooth=35,
                               random_state=20261020 + fold, n_jobs=4, verbosity=-1)
        model.fit(svdtr, y[tr], categorical_feature=cats)
        pv = model.predict_proba(svdva)[:, 1]
        pt = model.predict_proba(svdtest)[:, 1]
        oof["svd_lgbm"][va] = pv
        test_preds["svd_lgbm"] += pt / 3
        say("SVD_LGBM", fold, fixed_metrics(y[va], pv), "elapsed", round(time.time()-started, 1))
        reports.append({"fold": fold, "rows_train": len(tr), "rows_valid": len(va),
                        "group_overlap": 0, "text_linear": fixed_metrics(y[va], p_line),
                        "svd_lgbm": fixed_metrics(y[va], pv)})
        for name, vals in (("text_linear", (p_line, t_line)),
                           ("svd_lgbm", (pv, pt))):
            np.savez_compressed(OUT / f"{name}_fold{fold}.npz", valid_rows=va,
                                valid=vals[0], test=vals[1])
        del tf_word, tf_char, sparse_data, svds, svdtr, svdva, svdtest, model
        gc.collect()

    assert all(np.isfinite(pred).all() for pred in oof.values())
    candidate = {"old_catboost_average": old}
    candidate.update(oof)
    for alpha in (.25, .50, .75):
        candidate[f"new_blend_lgbm_{alpha:.2f}"] = alpha * oof["svd_lgbm"] + (1-alpha) * oof["text_linear"]
    for alpha in (.10, .20, .30):
        candidate[f"old_plus_{alpha:.2f}_new"] = (1-alpha) * old + alpha * (.65*oof["svd_lgbm"] + .35*oof["text_linear"])
    summary = {name: {**fixed_metrics(y, prob),
                      "prediction_changes_vs_old": int(np.count_nonzero((prob>=.5) != (old>=.5)))}
               for name, prob in candidate.items()}
    cases = pd.read_csv(STAGE / "submissions" / "20261019" / "probes_11_13_case_audit.csv")
    diagnostics = {}
    for cohort in ("P_strong_cv", "Q_broad_new"):
        q = cases.loc[cases.question_cohort.eq(cohort)]
        idx = q.ID.to_numpy(int) - 1
        want_used = q.historical_alternative_label.eq("used").to_numpy()
        observed = 4 if cohort == "P_strong_cv" else -19
        diagnostics[cohort] = {"n": len(q), "observed_public_net": observed,
                                "test_mean_probability_alt": {
                                    name: float(np.where(want_used, prob[idx], 1-prob[idx]).mean())
                                    for name, prob in {"old_catboost_average":
                                        (np.load(olddir / "RichDepthwise_oof_predictions.npz")["test"]
                                         + np.load(olddir / "RichPlain_oof_predictions.npz")["test"]) / 2,
                                        **test_preds}.items()}}
    result = {"date_argentina": "2026-10-20", "n_train": 70000, "n_test": 30000,
              "folds": reports, "metrics": summary, "posterior_cohort_diagnostics": diagnostics,
              "scope": "All supervised fits confined to outer train folds; public group feedback diagnostic only.",
              "limitation": "The train labels and CV split have been used in past experiments: not a fresh blind test.",
              "seconds": time.time() - started}
    for name in oof:
        np.savez_compressed(OUT / f"{name}_grouped_cv_predictions.npz",
                            oof=oof[name], test=test_preds[name])
    (OUT / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
    pd.DataFrame.from_dict(summary, orient="index").to_csv(OUT / "comparison.csv")
    say("COMPLETE", json.dumps({"seconds": round(result["seconds"],1),
                                  "metrics": summary, "cohorts": diagnostics},
                                 ensure_ascii=False))


if __name__ == "__main__":
    run()
