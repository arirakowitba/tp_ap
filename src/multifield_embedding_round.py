"""Fuse a pretrained text embedding of listing text with existing image metadata.

Images in this 2015 snapshot are no longer served by the public CDN. The model
therefore uses picture count/resolution/aspect metadata already available in the
rows, while encoding title, subtitle, warranty, tags, and non-condition attributes.
"""
from pathlib import Path
import argparse, hashlib, json, re, time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import accuracy_score

from embedding_round import MODEL_FILE, MODEL_ID, encode_titles, title_data
from image_metadata_round import summarize_pictures


def document(row):
    parts = []
    for field, label in (("title", "titulo"), ("subtitle", "subtitulo"), ("warranty", "garantia")):
        value = str(row.get(field) or "").strip()
        if value:
            parts.append(label + ": " + value)
    tags = row.get("tags") or []
    if isinstance(tags, list) and tags:
        parts.append("etiquetas: " + " ".join(sorted(map(str, tags))))
    attrs = row.get("attributes") or []
    attr_text = []
    if isinstance(attrs, list):
        for attr in attrs:
            if not isinstance(attr, dict):
                continue
            aid = str(attr.get("id") or "")
            name = str(attr.get("name") or "")
            # The item-condition attribute is the target itself; never encode it.
            if re.search(r"condition|condicion|condición|estado", aid + " " + name, re.I):
                continue
            value = attr.get("value_name") or attr.get("value_id")
            if value and name:
                attr_text.append(name + ": " + str(value))
    if attr_text:
        parts.append("atributos: " + "; ".join(attr_text[:32]))
    return " [SEP] ".join(parts)


def run(args):
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    titles, y, groups, fingerprint = title_data(args.data_dir)
    texts = []
    image_rows = []
    with (Path(args.data_dir) / "train_data.jsonlines").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            texts.append(document(row))
            image_rows.append(summarize_pictures(row.get("pictures")))
    if len(texts) != len(y):
        raise ValueError("Text rows do not align with labels")
    recipe = "title+subtitle+warranty+tags+attributes_without_item_condition+picture_metadata_v1"
    digest = hashlib.sha256((fingerprint + recipe).encode()).hexdigest()
    embedding_path = out / "embedding_multifield.npy"
    metadata_path = out / "embedding_multifield_metadata.json"
    if embedding_path.exists() and metadata_path.exists() and json.loads(metadata_path.read_text()).get("input_sha256") == digest:
        embedding = np.load(embedding_path)
    else:
        tmp = out / "multifield_encoder"
        tmp.mkdir(exist_ok=True)
        embedding = encode_titles(texts, args.model_dir, tmp, args.deadline)
        np.save(embedding_path, embedding)
        metadata_path.write_text(json.dumps({"model_id": MODEL_ID, "weights": MODEL_FILE,
                                             "input_sha256": digest, "recipe": recipe,
                                             "attributes_exclude_item_condition": True,
                                             "image_pixels_accessible": False,
                                             "embedding_dimensions": 384}, indent=2))
    del texts, titles

    cache = pd.read_pickle(args.cache)
    train_idx, valid_idx = cache["t"], cache["v"]
    frozen = pd.read_csv(Path(args.cache).parent / "selection_rows.csv")
    if not np.array_equal(frozen.row_index.to_numpy(), valid_idx) or not np.array_equal(frozen.y_used.to_numpy(), y[valid_idx]):
        raise ValueError("Multi-field text does not align with frozen seed42 rows")
    base = cache["tr"].copy()
    valid = cache["va"].copy()
    img = pd.DataFrame(image_rows)
    img_train = img.iloc[train_idx].reset_index(drop=True)
    img_valid = img.iloc[valid_idx].reset_index(drop=True)
    text_train = pd.DataFrame(embedding[train_idx], columns=[f"multi_text_{i}" for i in range(embedding.shape[1])])
    text_valid = pd.DataFrame(embedding[valid_idx], columns=text_train.columns)

    cfg_path = root / "results/round_v2/lgb_context_svm_tpe_12.json"
    cfg = json.loads(cfg_path.read_text())
    params = dict(cfg.get("params", cfg))
    params.update(n_jobs=2, n_estimators=2200, verbosity=-1, random_state=42)
    results = []
    for name, add_text, add_image in (("multimodal_base", False, False),
                                      ("multimodal_text", True, False),
                                      ("multimodal_text_image", True, True)):
        left, right = base.copy(), valid.copy()
        if add_text:
            left = pd.concat([left, text_train], axis=1)
            right = pd.concat([right, text_valid], axis=1)
        if add_image:
            left = pd.concat([left, img_train], axis=1)
            right = pd.concat([right, img_valid], axis=1)
        model = LGBMClassifier(**params)
        model.fit(left, y[train_idx], eval_set=[(right, y[valid_idx])], eval_metric="binary_logloss",
                  callbacks=[early_stopping(100, verbose=False)])
        p = model.predict_proba(right)[:, 1]
        threshold = float(max(np.arange(.45, .551, .005), key=lambda q: accuracy_score(y[valid_idx], p >= q)))
        result = {"name": name, "accuracy": float(accuracy_score(y[valid_idx], p >= threshold)),
                  "threshold": threshold, "best_iteration": int(model.best_iteration_),
                  "text_representation": recipe if add_text else "base_full_text_tfidf_svd",
                  "picture_metadata_added": add_image, "image_pixels_used": False,
                  "feature_count": int(left.shape[1]), "split_seed": 42}
        np.save(out / (name + ".npy"), p)
        (out / (name + ".json")).write_text(json.dumps(result, indent=2))
        results.append(result)
        print("RESULT", json.dumps(result), flush=True)
    pd.DataFrame(results).to_csv(out / "multimodal_ablation.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--deadline", type=float)
    run(parser.parse_args())
