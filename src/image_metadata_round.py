"""Ablate structured image metadata against the frozen grouped text/tabular holdout.

The original image payloads are unavailable in the competition snapshot. This
experiment uses only dimensions, aspect ratios, resolution spread, and image
quality metadata supplied in each listing's picture records.
"""
from pathlib import Path
import argparse, json, re

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import accuracy_score


def size(value):
    if not isinstance(value, str):
        return np.nan, np.nan
    match = re.search(r"(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)", value)
    return (float(match.group(1)), float(match.group(2))) if match else (np.nan, np.nan)


def summarize_pictures(pictures):
    if not isinstance(pictures, list):
        pictures = []
    current, maximum, ratios, ids = [], [], [], []
    secure = 0
    for item in pictures:
        if not isinstance(item, dict):
            continue
        w, h = size(item.get("size"))
        mw, mh = size(item.get("max_size"))
        if np.isfinite(w) and np.isfinite(h) and w > 0 and h > 0:
            current.append((w, h))
            ratios.append(w / h)
        if np.isfinite(mw) and np.isfinite(mh) and mw > 0 and mh > 0:
            maximum.append((mw, mh))
        if item.get("id"):
            ids.append(str(item["id"]))
        secure += int(bool(item.get("secure_url")))

    areas = [w * h / 1e6 for w, h in current]
    max_areas = [w * h / 1e6 for w, h in maximum]
    ratios = np.asarray(ratios, dtype=float)
    n = len(pictures)
    return {
        "img_unique_id_n": len(set(ids)),
        "img_duplicate_id_n": len(ids) - len(set(ids)),
        "img_secure_fraction": secure / n if n else np.nan,
        "img_mp_mean": float(np.mean(areas)) if areas else np.nan,
        "img_mp_min": float(np.min(areas)) if areas else np.nan,
        "img_mp_std": float(np.std(areas)) if len(areas) > 1 else 0.0 if areas else np.nan,
        "img_max_mp_mean": float(np.mean(max_areas)) if max_areas else np.nan,
        "img_aspect_std": float(np.std(ratios)) if len(ratios) > 1 else 0.0 if ratios.size else np.nan,
        "img_portrait_fraction": float(np.mean(ratios < .9)) if ratios.size else np.nan,
        "img_square_fraction": float(np.mean((ratios >= .9) & (ratios <= 1.1))) if ratios.size else np.nan,
        "img_landscape_fraction": float(np.mean(ratios > 1.1)) if ratios.size else np.nan,
        "img_max_to_current_mp": (float(np.mean(max_areas) / np.mean(areas))
                                  if areas and max_areas and np.mean(areas) > 0 else np.nan),
    }


def run(args):
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cache = pd.read_pickle(args.cache)
    train_idx, valid_idx = cache["t"], cache["v"]
    y, image_rows = [], []
    with (Path(args.data_dir) / "train_data.jsonlines").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            y.append(int(row["condition"] == "used"))
            image_rows.append(summarize_pictures(row.get("pictures")))
    y = np.asarray(y, dtype=np.int8)
    frozen = pd.read_csv(Path(args.cache).parent / "selection_rows.csv")
    if not np.array_equal(frozen.row_index.to_numpy(), valid_idx) or not np.array_equal(frozen.y_used.to_numpy(), y[valid_idx]):
        raise ValueError("Image metadata is not aligned to the frozen seed42 holdout")
    image = pd.DataFrame(image_rows)
    tr, va = cache["tr"].copy(), cache["va"].copy()
    extra_tr = image.iloc[train_idx].reset_index(drop=True)
    extra_va = image.iloc[valid_idx].reset_index(drop=True)
    cfg = json.loads((root / "results/round_v2/lgb_context_svm_tpe_12.json").read_text())
    params = dict(cfg.get("params", cfg))
    params.update(n_jobs=2, n_estimators=2200, verbosity=-1, random_state=42)

    for name, add_images in (("image_meta_base", False), ("image_meta_plus", True)):
        left, right = tr.copy(), va.copy()
        if add_images:
            left = pd.concat([left, extra_tr], axis=1)
            right = pd.concat([right, extra_va], axis=1)
        model = LGBMClassifier(**params)
        model.fit(left, y[train_idx], eval_set=[(right, y[valid_idx])], eval_metric="binary_logloss",
                  callbacks=[early_stopping(100, verbose=False)])
        p = model.predict_proba(right)[:, 1]
        threshold = float(max(np.arange(.45, .551, .005), key=lambda q: accuracy_score(y[valid_idx], p >= q)))
        result = {
            "name": name, "accuracy": float(accuracy_score(y[valid_idx], p >= threshold)),
            "threshold": threshold, "logloss": float(-np.mean(y[valid_idx] * np.log(np.clip(p, 1e-12, 1)) + (1-y[valid_idx]) * np.log(np.clip(1-p, 1e-12, 1)))),
            "best_iteration": int(model.best_iteration_), "added_image_metadata": add_images,
            "feature_count": int(left.shape[1]), "split_seed": 42,
            "image_payload_used": False,
        }
        np.save(output / (name + ".npy"), p)
        (output / (name + ".json")).write_text(json.dumps(result, indent=2))
        print("RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
