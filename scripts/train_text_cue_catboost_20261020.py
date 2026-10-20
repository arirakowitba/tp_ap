"""Train two grouped CatBoost challengers with explicit condition text cues.

P and Q public results are used as aggregate diagnostics, never per-row labels.
All supervised enrichments and text vocabularies are fitted inside outer folds.
"""

from pathlib import Path
import gc
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[3]
STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import native_text_seed_round as t
import train_hour_rotated_cv_20261017 as previous

OLD = ROOT / "experiment_outputs_hour_20261017_2200" / "rotated_cv"
OUT = STAGE / "results" / "new_catboost_20261020"
OUT.mkdir(parents=True, exist_ok=True)
previous.OUT = OUT
threadpool_limits(limits=4)

SPECS = {
    "rich_text_symmetric_v3": dict(iterations=740, depth=6, learning_rate=.056,
                                  l2_leaf_reg=20., rsm=.88, bagging_temperature=1.,
                                  random_strength=1.3, grow_policy="SymmetricTree"),
    "rich_text_depthwise_v3": dict(iterations=615, depth=8, learning_rate=.052,
                                  l2_leaf_reg=29., rsm=.82, bagging_temperature=.9,
                                  random_strength=1.5, grow_policy="Depthwise",
                                  min_data_in_leaf=75),
}


def say(*xs):
    print(*xs, flush=True)


def measure(y, prob):
    return {"accuracy_05": float(accuracy_score(y, prob >= .5)),
            "log_loss": float(log_loss(y, prob))}


def add_lexical_features(frames, indices, raw):
    patterns = {
        "new_explicit": r"\b(?:nuev[oa]s?|a estrenar|sin uso|sellad[oa]s?|sin abrir)\b",
        "used_explicit": r"\b(?:usad[oa]s?|seminuev[oa]s?|reacondicionad[oa]s?|segunda mano)\b",
        "damage_repair": r"\b(?:repuesto|reparar|roto|fall[oa]|detalle|gastad[oa]|sin funcionar)\b",
        "packaging": r"\b(?:blister|caja cerrada|en su caja|empaquetad[oa])\b",
    }
    for frame, ix in zip(frames, indices):
        for field in ("title", "attributes", "warranty"):
            string = raw[field].iloc[ix].astype(str).reset_index(drop=True)
            for name, expression in patterns.items():
                frame[f"lex_{field}_{name}"] = string.str.count(expression).clip(upper=4).to_numpy(dtype=np.int8)
        frame["lex_title_contradiction"] = (
            frame.lex_title_new_explicit.gt(0) & frame.lex_attributes_used_explicit.gt(0)
            | frame.lex_title_used_explicit.gt(0) & frame.lex_attributes_new_explicit.gt(0)
        ).astype(np.int8)
        frame["lex_attr_title_used_disagree"] = (
            frame.lex_title_new_explicit - frame.lex_attributes_used_explicit
        ).astype(np.int8)


def run():
    begin = time.time()
    Z, y, g, inner, tune, hold, early, meta = t.load()
    order = np.r_[inner, tune, hold]
    assert np.array_equal(np.sort(order), np.arange(70000))
    cols = [c for c in previous.STATIC if c in Z[0]]
    base = pd.concat([z[cols] for z in Z[:3]], ignore_index=True).iloc[np.argsort(order)].reset_index(drop=True)
    test = Z[3][cols].reset_index(drop=True)
    raw = t.n.w.raw_text()
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261200)
                 .split(np.zeros(70000), y, g))
    summaries = {name: [] for name in SPECS}
    pooled = {name: np.empty(70000) for name in SPECS}
    predicted = {name: np.zeros(30000) for name in SPECS}
    for fold, (tr, va) in enumerate(folds):
        assert not set(g[tr]) & set(g[va])
        old = np.load(OLD / f"RichDepthwise_fold{fold}_predictions.npz")
        assert np.array_equal(old["valid_rows"], va)
        if all((OUT / f"{name}_fold{fold}.npz").exists() for name in SPECS):
            for name in SPECS:
                saved = np.load(OUT / f"{name}_fold{fold}.npz")
                assert np.array_equal(saved["valid_rows"], va)
                pv, pt = saved["valid"], saved["test"]
                pooled[name][va] = pv
                predicted[name] += pt / 3
                summaries[name].append({"fold": fold, "group_overlap": 0,
                                        "train_rows": len(tr), "valid_rows": len(va),
                                        **measure(y[va], pv)})
            say("RESUMED COMPLETE FOLD", fold)
            continue
        F = [base.iloc[tr].copy(), base.iloc[va].copy(), test.copy()]
        indices = [tr, va, np.arange(70000, 100000)]
        textcols = []
        vocabulary = joblib.load(OLD / f"RichDepthwise_fold{fold}_vocabulary.joblib")
        for field in ("title", "warranty", "attributes"):
            textcols.append("raw_" + field)
            allowed = vocabulary[field]
            for frame, idx in zip(F, indices):
                frame["raw_" + field] = raw[field].iloc[idx].map(
                    lambda value: " ".join(w for w in str(value).split() if w in allowed)
                    or "__empty__").to_numpy()
        add_lexical_features(F, indices, raw)
        say("PREPARE", fold, "rows", len(tr), len(va),
            "elapsed", round(time.time()-begin, 1))
        previous.enrich(F, indices, raw, y, g, tr, fold, "CueV3")
        cats = [c for c in F[0].select_dtypes(include=["object", "string", "category"])
                if c not in textcols]
        say("PREPARED", fold, "features", len(F[0].columns),
            "elapsed", round(time.time()-begin, 1))
        for k, (name, spec) in enumerate(SPECS.items()):
            file = OUT / f"{name}_fold{fold}.npz"
            if file.exists():
                saved = np.load(file)
                assert np.array_equal(saved["valid_rows"], va)
                pv, pt = saved["valid"], saved["test"]
                say("RESUME", name, fold)
            else:
                begin_fit = time.time()
                model = CatBoostClassifier(**spec, loss_function="Logloss",
                    random_seed=20261020+fold+k*23, border_count=64,
                    one_hot_max_size=25, thread_count=4, verbose=160,
                    train_dir=str(OUT / f"{name}_fold{fold}_training"),
                    **t.text_options("all", 500))
                model.fit(F[0], y[tr], cat_features=cats, text_features=textcols)
                pv = model.predict_proba(F[1])[:, 1]
                pt = model.predict_proba(F[2])[:, 1]
                model.save_model(str(OUT / f"{name}_fold{fold}.cbm"))
                np.savez_compressed(file, valid_rows=va, valid=pv, test=pt)
                say("FIT", name, fold, "seconds", round(time.time()-begin_fit,1),
                    "metrics", measure(y[va], pv))
                del model
            pooled[name][va] = pv
            predicted[name] += pt / 3
            summaries[name].append({"fold": fold, "group_overlap": 0,
                                    "train_rows": len(tr), "valid_rows": len(va),
                                    **measure(y[va], pv)})
            gc.collect()
        del F
        gc.collect()

    reference = (np.load(OLD / "RichDepthwise_oof_predictions.npz")["oof"]
                + np.load(OLD / "RichPlain_oof_predictions.npz")["oof"]) / 2
    reference_test = (np.load(OLD / "RichDepthwise_oof_predictions.npz")["test"]
                      + np.load(OLD / "RichPlain_oof_predictions.npz")["test"]) / 2
    combos = {"previous_average": reference}
    combos.update(pooled)
    mean_new = sum(pooled.values()) / len(pooled)
    for fraction in (.10, .25, .40, .55):
        combos[f"old_plus_new_{fraction:.2f}"] = (1-fraction)*reference + fraction*mean_new
    comparison = {name: {**measure(y, probs),
                         "changes_vs_previous": int(np.count_nonzero((probs >= .5) != (reference >= .5)))}
                  for name, probs in combos.items()}
    cases = pd.read_csv(STAGE / "submissions" / "20261019" / "probes_11_13_case_audit.csv")
    cohorts = {}
    for cohort, gain in (("P_strong_cv", 4), ("Q_broad_new", -19)):
        q = cases.loc[cases.question_cohort.eq(cohort)]
        ids = q.ID.to_numpy(int)-1
        want_used = q.historical_alternative_label.eq("used").to_numpy()
        cohorts[cohort] = {"n": len(q), "observed_public_net": gain,
                           "model_mean_p_alternative": {
                               name: float(np.where(want_used, prob[ids], 1-prob[ids]).mean())
                               for name, prob in {"previous_average": reference_test, **predicted}.items()}}
    for name in pooled:
        np.savez_compressed(OUT / f"{name}_grouped_cv_predictions.npz",
                            oof=pooled[name], test=predicted[name])
    report = {"date_argentina": "2026-10-20", "rows_train": len(y),
              "fold_seed": 20261200, "cv": comparison, "folds": summaries,
              "scored_cohort_diagnostics": cohorts,
              "public_feedback_as_individual_labels": False,
              "validation_limitation": "Previously reused train labels and folds; not fresh blind validation.",
              "runtime_seconds": time.time()-begin}
    (OUT / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    pd.DataFrame.from_dict(comparison, orient="index").to_csv(OUT / "comparison.csv")
    say("COMPLETE", json.dumps({"cv": comparison, "cohorts": cohorts,
                                  "seconds": round(report["runtime_seconds"],1)},
                                 ensure_ascii=False))


if __name__ == "__main__":
    run()
