"""Bounded CatBoost ablations on the frozen original seed-42 split.

Only parent-prepared features are loaded: no raw data reload, video IDs, image
IDs, remote metadata, or confirmation labels enter the selection experiment.
Each fit runs in a separate process so native training stops at the deadline.
"""
from __future__ import annotations
import argparse
import datetime as dt
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "enriched_d9": dict(depth=9, iterations=1500, learning_rate=.035,
                        l2_leaf_reg=12, random_seed=73),
    "native_text_d7": dict(depth=7, iterations=1400, learning_rate=.04,
                          l2_leaf_reg=8, random_seed=51),
}


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def media_frames(data, args):
    import pandas as pd
    if "media_tr" in data and "media_va" in data:
        return data["media_tr"].reset_index(drop=True), data["media_va"].reset_index(drop=True)
    if not args.media_features:
        raise ValueError("Media metadata requires media_tr/media_va or --media-features.")
    features = pd.read_csv(args.media_features)
    if "row_index" in features:
        features = features.set_index("row_index", verify_integrity=True)
        return (features.loc[data["t"]].reset_index(drop=True),
                features.loc[data["v"]].reset_index(drop=True))
    return (features.iloc[data["t"]].reset_index(drop=True),
            features.iloc[data["v"]].reset_index(drop=True))


def fit(args):
    import numpy as np
    import pandas as pd
    from catboost import CatBoostClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
    start = time.monotonic()
    data = pd.read_pickle(args.cache)
    tr, va = data["tr"].copy(), data["va"].copy()
    cats = list(data["cats"])
    y_train, y_eval = np.asarray(data["y_train"]), np.asarray(data["y_eval"])
    if len(tr) != len(y_train) or len(va) != len(y_eval):
        raise ValueError("Feature/label row counts disagree.")
    if len(set(map(int, data["t"])).intersection(map(int, data["v"]))):
        raise ValueError("Train and selection rows overlap.")
    feature_sets = ["original_enriched_numeric_text_svd96"]
    if args.metadata:
        ma, mb = media_frames(data, args)
        columns = [column for column in ma if column.startswith("media_")]
        if not columns:
            raise ValueError("No explicit media_ metadata columns found.")
        if len(ma) != len(tr) or len(mb) != len(va):
            raise ValueError("Media metadata row counts disagree.")
        for column in columns:
            if column in tr:
                raise ValueError(f"Duplicate media feature: {column}")
            tr[column] = pd.to_numeric(ma[column], errors="coerce").to_numpy(dtype="float32")
            va[column] = pd.to_numeric(mb[column], errors="coerce").to_numpy(dtype="float32")
        feature_sets.append("image_video_metadata")
    texts = []
    if args.config.startswith("native"):
        for frame, key in ((tr, "text_tr"), (va, "text_va")):
            frame["full_text"] = pd.Series(data[key]).fillna("").astype(str).replace("", "empty").to_numpy()
        texts = ["full_text"]
        feature_sets.append("catboost_native_full_text")
    for column in cats:
        tr[column] = tr[column].astype("string").fillna("__MISSING__").astype(str)
        va[column] = va[column].astype("string").fillna("__MISSING__").astype(str)
    numeric = tr.columns.difference(cats + texts)
    tr[numeric] = tr[numeric].replace([np.inf, -np.inf], np.nan)
    va[numeric] = va[numeric].replace([np.inf, -np.inf], np.nan)
    del data["tr"], data["va"]
    gc.collect()
    name = "r4_cat_" + args.config + ("_media" if args.metadata else "_original")
    params = dict(CONFIGS[args.config])
    model = CatBoostClassifier(**params, loss_function="Logloss", eval_metric="Accuracy",
                               thread_count=2, allow_writing_files=False,
                               one_hot_max_size=10, random_strength=.5)
    model.fit(tr, y_train, cat_features=cats, text_features=texts,
              eval_set=(va, y_eval), early_stopping_rounds=130, verbose=100)
    probability = model.predict_proba(va)[:, 1]
    if not np.isfinite(probability).all():
        raise ValueError("Nonfinite prediction.")
    frozen_params = dict(params, iterations=int(model.tree_count_))
    prediction = probability >= .5
    out = Path(args.output)
    np.save(out / (name + ".npy"), probability)
    pd.DataFrame({"row_index": data["v"], "y": y_eval, "p": probability}).to_csv(
        out / (name + "_predictions.csv"), index=False)
    model.save_model(str(out / (name + ".cbm")))
    result = dict(name=name, family="cat", model_type="cat", protocol="selection_seed42_holdout3",
                  accuracy=float(accuracy_score(y_eval, prediction)),
                  balanced_accuracy=float(balanced_accuracy_score(y_eval, prediction)),
                  macro_f1=float(f1_score(y_eval, prediction, average="macro")),
                  logloss=float(log_loss(y_eval, probability)), threshold=.5,
                  n_train=len(y_train), n_eval=len(y_eval), n_features=len(tr.columns),
                  params=frozen_params, requested_params=params, feature_sets=feature_sets,
                  config=args.config, media_metadata=bool(args.metadata),
                  cat_features=cats, text_features=texts, seconds=time.monotonic()-start,
                  completed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                  confirmation_used=False, image_pixels_used=False, video_frames_used=False,
                  fingerprint=data.get("fingerprint"),
                  cache=str(Path(args.cache).resolve()))
    atomic_json(out / (name + ".json"), result)
    print("CAT_ROUND4_RESULT", json.dumps(result), flush=True)


def refresh_metrics(out):
    import pandas as pd
    rows = [json.loads(path.read_text()) for path in sorted(out.glob("r4_cat_*.json"))]
    if rows:
        pd.DataFrame([{key: value for key, value in row.items()
                       if key not in {"params", "requested_params", "cat_features", "feature_sets", "text_features"}}
                      for row in rows]).to_csv(out / "cat_round4_metrics.csv", index=False)


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stages = []
    for metadata in (False, True):
        for config in CONFIGS:
            name = "r4_cat_" + config + ("_media" if metadata else "_original")
            if (out / (name + ".json")).exists():
                stages.append(dict(name=name, status="cached"))
                continue
            remaining = args.deadline - time.time() - 5
            if remaining < 30:
                stages.append(dict(name=name, status="not_started_deadline"))
                break
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--cache", args.cache,
                       "--output", str(out), "--deadline", str(args.deadline),
                       "--worker", "--config", config]
            if metadata:
                command.append("--metadata")
            if args.media_features:
                command.extend(["--media-features", args.media_features])
            with (out / (name + "_run.log")).open("w") as logfile:
                process = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    code = process.wait(timeout=remaining)
                    status = "completed" if code == 0 else "failed"
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    status, code = "terminated_deadline", None
            stages.append(dict(name=name, status=status, returncode=code))
            refresh_metrics(out)
            atomic_json(out / "cat_round4_execution.json", dict(stages=stages, deadline=args.deadline))
            print(name, status, flush=True)
    refresh_metrics(out)
    atomic_json(out / "cat_round4_execution.json", dict(stages=stages, deadline=args.deadline))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", default=str(ROOT / "results/round_v4"))
    parser.add_argument("--deadline", required=True, type=float)
    parser.add_argument("--media-features")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--config", choices=list(CONFIGS))
    parser.add_argument("--metadata", action="store_true")
    args = parser.parse_args()
    fit(args) if args.worker else run(args)


if __name__ == "__main__":
    main()
