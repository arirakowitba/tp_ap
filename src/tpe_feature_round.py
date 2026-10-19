"""Grouped TPE tuning, leakage-safe target features, and image-metadata ablation.

The seed-42 development matrix is used for grouped cross-validation only. A
single finalist is checked on the already-frozen seed-731 confirmation split;
the public Kaggle score is never inferred from local validation.
"""
from pathlib import Path
import argparse
import json
import re
import time

import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold

from competition_pipeline_v2 import groups_for, svm_probability
from image_metadata_round import summarize_pictures
from optimize_leaderboard import enrich, split
from optimize_stack import group_rates

ROOT = Path(__file__).resolve().parents[1]


def load_rows(path):
    """Load only the nested data needed for text fitting and image features."""
    import json

    rows, labels, ids = [], [], []
    image_rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rows.append(row)
            labels.append(int(row["condition"] == "used"))
            ids.append(str(row.get("id")))
            image_rows.append(summarize_pictures(row.get("pictures")))
    raw = pd.DataFrame(rows)
    return raw, np.asarray(labels, dtype=np.int8), groups_for(raw), pd.DataFrame(image_rows)


def target_rate_pair(train_keys, y, groups, valid_keys, alpha=5.0, seed=2026):
    """OOF smoothed group rates for train and a full-train rate for validation."""
    train_keys = pd.Series(train_keys).fillna("__MISSING__").astype(str).reset_index(drop=True)
    valid_keys = pd.Series(valid_keys).fillna("__MISSING__").astype(str).reset_index(drop=True)
    y = np.asarray(y, dtype=np.int8)
    groups = np.asarray(groups).astype(str)
    prior = float(y.mean())
    rate = np.empty(len(y), dtype=np.float32)
    count = np.empty(len(y), dtype=np.float32)
    folds = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    for it, iv in folds.split(np.zeros(len(y)), y, groups):
        stats = pd.DataFrame({"key": train_keys.iloc[it].to_numpy(), "y": y[it]}).groupby("key").y.agg(["sum", "count"])
        n = train_keys.iloc[iv].map(stats["count"]).fillna(0).to_numpy(dtype=np.float32)
        s = train_keys.iloc[iv].map(stats["sum"]).fillna(0).to_numpy(dtype=np.float32)
        rate[iv] = (s + alpha * float(y[it].mean())) / (n + alpha)
        count[iv] = np.log1p(n)
    stats = pd.DataFrame({"key": train_keys.to_numpy(), "y": y}).groupby("key").y.agg(["sum", "count"])
    n = valid_keys.map(stats["count"]).fillna(0).to_numpy(dtype=np.float32)
    s = valid_keys.map(stats["sum"]).fillna(0).to_numpy(dtype=np.float32)
    return rate, count, (s + alpha * prior) / (n + alpha), np.log1p(n)


def image_pair(image, train_rows, valid_rows):
    return (image.iloc[train_rows].reset_index(drop=True),
            image.iloc[valid_rows].reset_index(drop=True))


def build_cv_folds(raw, y, groups, image, cache, output):
    """Build each outer fold once; nested SVM and target rates stay within fold."""
    t, _ = cache["t"], cache["v"]
    base = cache["tr"].reset_index(drop=True)
    fold_dir = output / "fold_cache"
    fold_dir.mkdir(parents=True, exist_ok=True)
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=2026)
                 .split(np.zeros(len(t)), y[t], groups[t]))
    text_frame, _ = enrich(raw.iloc[t].reset_index(drop=True))
    stale = [c for c in base.columns if c.endswith("_oof_rate") or c.endswith("_oof_count")
             or c == "svm_oof_probability"]
    clean = base.drop(columns=stale)

    built = []
    for fold, (it, iv) in enumerate(folds):
        path = fold_dir / f"seed42_fold{fold}.pkl"
        if path.exists():
            built.append(pd.read_pickle(path))
            continue
        started = time.time()
        xa = clean.iloc[it].reset_index(drop=True).copy()
        xb = clean.iloc[iv].reset_index(drop=True).copy()
        xa, xb = group_rates(xa, xb, y[t[it]], groups[t[it]])
        ia, ib = image_pair(image, t[it], t[iv])
        xa = pd.concat([xa, ia], axis=1)
        xb = pd.concat([xb, ib], axis=1)

        # Exact-title label signal is useful for repeated listings. It is
        # cross-fitted inside the outer training fold and never sees fold labels.
        ra, ca, rb, cb = target_rate_pair(text_frame.title_text.iloc[it], y[t[it]], groups[t[it]],
                                           text_frame.title_text.iloc[iv], alpha=5, seed=4000 + fold)
        xa["title_exact_oof_rate"], xa["title_exact_oof_count"] = ra, ca
        xb["title_exact_oof_rate"], xb["title_exact_oof_count"] = rb, cb

        # Rebuild the text classifier meta-feature with nested OOF predictions.
        # This prevents labels from the outer validation fold entering the SVM
        # probability features used to train this fold's LightGBM.
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=6000 + fold)
        svm_train = np.zeros(len(it), dtype=np.float32)
        for subtrain, subvalid in inner.split(np.zeros(len(it)), y[t[it]], groups[t[it]]):
            svm_train[subvalid] = svm_probability(text_frame.iloc[it[subtrain]], y[t[it[subtrain]]],
                                                          text_frame.iloc[it[subvalid]])
        svm_valid = svm_probability(text_frame.iloc[it], y[t[it]], text_frame.iloc[iv])
        xa["svm_oof_probability"] = svm_train
        xb["svm_oof_probability"] = svm_valid
        frame = {"train": xa, "valid": xb, "train_rows": it, "valid_rows": iv,
                 "best_iteration": None}
        pd.to_pickle(frame, path)
        built.append(frame)
        print("FOLD_READY", fold, "train", xa.shape, "valid", xb.shape,
              "seconds", round(time.time() - started), flush=True)
    del text_frame
    return built


def metric_row(name, y, p, threshold=0.5, **extra):
    return {"name": name, "accuracy": float(accuracy_score(y, p >= threshold)),
            "logloss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
            "threshold": threshold, **extra}


def suggest_params(trial):
    return {
        "n_estimators": 2400,
        "learning_rate": trial.suggest_float("learning_rate", 0.008, 0.075, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 128),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 6, 8, 10, 12]),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 180),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.2, 60.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 20.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "subsample_freq": 1,
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        "cat_smooth": trial.suggest_float("cat_smooth", 2.0, 100.0, log=True),
        "cat_l2": trial.suggest_float("cat_l2", 1.0, 50.0, log=True),
        "max_cat_threshold": trial.suggest_categorical("max_cat_threshold", [8, 16, 32, 64]),
        "n_jobs": 2,
        "verbosity": -1,
        "random_state": 2026,
    }


def evaluate_cv(params, folds, y_dev, feature_names=None, collect_importance=False):
    oof = np.zeros(len(y_dev), dtype=np.float32)
    gains, iterations = [], []
    for k, fold in enumerate(folds):
        tr, va = fold["train"], fold["valid"]
        if feature_names is not None:
            tr, va = tr[feature_names], va[feature_names]
        model = LGBMClassifier(**params)
        model.fit(tr, y_dev[fold["train_rows"]],
                  eval_set=[(va, y_dev[fold["valid_rows"]])],
                  eval_metric="binary_logloss",
                  callbacks=[early_stopping(120, verbose=False)])
        p = model.predict_proba(va)[:, 1]
        oof[fold["valid_rows"]] = p
        iterations.append(int(model.best_iteration_ or params["n_estimators"]))
        if collect_importance:
            gains.append(pd.Series(model.booster_.feature_importance("gain"), index=tr.columns))
        print("CV_FOLD", k, "accuracy", round(accuracy_score(y_dev[fold["valid_rows"]], p >= .5), 6),
              "best_iter", iterations[-1], flush=True)
        del model
    result = metric_row("cv", y_dev, oof, .5,
                        fold_best_iterations=iterations)
    importance = pd.concat(gains, axis=1).fillna(0).mean(axis=1).sort_values(ascending=False) if gains else None
    return result, oof, importance


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    raw, y, groups, image = load_rows(Path(args.data_dir) / "train_data.jsonlines")
    selection = pd.read_pickle(args.cache)
    t42, v42 = selection["t"], selection["v"]
    if len(t42) != 56000 or len(v42) != 14000:
        raise ValueError("The frozen seed-42 matrix must contain 56k/14k rows")
    rows_file = Path(args.cache).parent / "selection_rows.csv"
    frozen = pd.read_csv(rows_file)
    if not np.array_equal(frozen.row_index.to_numpy(), v42) or not np.array_equal(frozen.y_used.to_numpy(), y[v42]):
        raise ValueError("The seed-42 cache does not match its frozen row manifest")
    folds = build_cv_folds(raw, y, groups, image, selection, out)
    y_dev = y[t42]

    base_config = json.loads((ROOT / "results/round_v2/lgb_context_svm_tpe_12.json").read_text())
    baseline_params = dict(base_config["params"])
    baseline_params.update(n_estimators=2400, n_jobs=2, verbosity=-1, random_state=2026)
    baseline, _, _ = evaluate_cv(baseline_params, folds, y_dev)
    baseline["name"] = "previous_tpe_params"
    records = [baseline]
    (out / "tpe_cv_baseline.json").write_text(json.dumps(baseline, indent=2))
    print("BASELINE_CV", json.dumps(baseline), flush=True)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=20261013, n_startup_trials=6),
                                pruner=optuna.pruners.NopPruner())
    state = {"best": -1.0, "params": None, "oof": None, "importance": None, "row": None}

    def objective(trial):
        params = suggest_params(trial)
        row, oof, importance = evaluate_cv(params, folds, y_dev, collect_importance=True)
        trial.set_user_attr("logloss", row["logloss"])
        trial.set_user_attr("fold_best_iterations", row["fold_best_iterations"])
        if row["accuracy"] > state["best"]:
            state.update(best=row["accuracy"], params=params, oof=oof, importance=importance, row=row)
        print("TPE_RESULT", trial.number, row["accuracy"], row["logloss"], flush=True)
        return row["accuracy"]

    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    study.trials_dataframe().to_csv(out / "tpe_feature_trials.csv", index=False)
    (out / "tpe_feature_best_params.json").write_text(json.dumps(state["params"], indent=2))
    state["importance"].rename("mean_gain").to_csv(out / "tpe_feature_importance.csv")
    np.save(out / "tpe_feature_best_oof.npy", state["oof"])

    ranked = state["importance"].index.tolist()
    all_cols = list(folds[0]["train"].columns)
    # Rank cutoffs directly test whether low-importance columns add noise.
    candidate_sets = {
        "all_with_image_metadata": all_cols,
        "positive_gain": [c for c in ranked if state["importance"][c] > 0],
        "top_300": ranked[:300],
        "top_200": ranked[:200],
        "top_120": ranked[:120],
    }
    subset_records = []
    for name, cols in candidate_sets.items():
        row, oof, _ = evaluate_cv(state["params"], folds, y_dev, feature_names=cols)
        row.update(name=name, feature_count=len(cols))
        subset_records.append(row)
        np.save(out / f"{name}_oof.npy", oof)
        print("FEATURE_ABLATION", json.dumps(row), flush=True)
    pd.DataFrame(subset_records).to_csv(out / "tpe_feature_ablation.csv", index=False)
    best_set = max(subset_records, key=lambda r: (r["accuracy"], -r["feature_count"]))
    chosen_cols = candidate_sets[best_set["name"]]

    # One seed-731 confirmation, with an iteration count fixed by development CV.
    confirm_path = out.parent / "round_v5" / "confirmation" / "matrices.pkl"
    if not confirm_path.exists():
        confirm_path = Path(args.confirmation_cache)
    confirm = pd.read_pickle(confirm_path)
    t731, v731 = split(y, groups, 731, 2)
    trc, vac = confirm["tr"].copy(), confirm["va"].copy()
    if len(trc) != len(t731) or len(vac) != len(v731):
        raise ValueError("Seed-731 confirmation cache is not aligned")
    trc = pd.concat([trc.reset_index(drop=True), image.iloc[t731].reset_index(drop=True)], axis=1)
    vac = pd.concat([vac.reset_index(drop=True), image.iloc[v731].reset_index(drop=True)], axis=1)
    # Exact-title target rates are cross-fitted only inside the confirmation train set.
    title_keys = raw.title.fillna("").astype(str).str.lower()
    exact_a, exact_n, exact_b, exact_bn = target_rate_pair(
        title_keys.iloc[t731], y[t731], groups[t731], title_keys.iloc[v731], alpha=5, seed=2026)
    trc["title_exact_oof_rate"], trc["title_exact_oof_count"] = exact_a, exact_n
    vac["title_exact_oof_rate"], vac["title_exact_oof_count"] = exact_b, exact_bn
    names = [c for c in chosen_cols if c in trc.columns and c in vac.columns]
    iterations = int(np.median(state["row"]["fold_best_iterations"]))
    params = dict(state["params"])
    params["n_estimators"] = max(100, iterations)
    model = LGBMClassifier(**params)
    model.fit(trc[names], y[t731])
    p = model.predict_proba(vac[names])[:, 1]
    confirmation = metric_row("tpe_feature_candidate", y[v731], p, .5,
                               feature_count=len(names), n_estimators=iterations,
                               cv_accuracy=best_set["accuracy"], feature_set=best_set["name"])
    baseline_state = json.loads((out.parent / "round_v5" / "round3_selected.json").read_text())
    prior = np.load(out.parent / "round_v5" / "confirmation" / "prior_ensemble.npy")
    confirmation["prior_accuracy_seed731"] = float(accuracy_score(y[v731], prior >= .5))
    confirmation["promote_over_prior"] = confirmation["accuracy"] > confirmation["prior_accuracy_seed731"]
    confirmation["previous_confirmation_accuracy"] = baseline_state["prior_confirmation"]["accuracy"]
    (out / "tpe_feature_confirmation.json").write_text(json.dumps(confirmation, indent=2))
    np.save(out / "tpe_feature_confirmation.npy", p)
    pd.DataFrame({"row_index": v731, "y_used": y[v731], "probability": p,
                  "prior_probability": prior}).to_csv(out / "tpe_feature_confirmation_predictions.csv", index=False)
    print("CONFIRMATION", json.dumps(confirmation), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default=str(ROOT / "results/round_v6"))
    parser.add_argument("--cache", default=str(ROOT / "results/round_v5/cache_base/matrices.pkl"))
    parser.add_argument("--confirmation-cache", default=str(ROOT / "results/round_v5/confirmation/matrices.pkl"))
    parser.add_argument("--trials", type=int, default=18)
    run(parser.parse_args())
