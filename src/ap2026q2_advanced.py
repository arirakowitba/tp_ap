"""Advanced, leakage-aware feature set for the Mercado Libre condition task."""
from __future__ import annotations

import ast
import numpy as np
import pandas as pd


def _parse(value):
    if isinstance(value, (dict, list, tuple)):
        return value
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError):
            return None
    return None


def _nested(value, *keys):
    item = _parse(value)
    for key in keys:
        if not isinstance(item, dict):
            return np.nan
        item = item.get(key)
    return item if isinstance(item, (str, int, float, bool)) else np.nan


def _attributes(raw, row_indices=None, top_k=75):
    """Select frequent structured attributes on training rows only."""
    selected_rows = set(range(len(raw))) if row_indices is None else set(map(int, row_indices))
    parsed = []
    frequency = {}
    for i, value in enumerate(raw.get("attributes", pd.Series([None] * len(raw)))):
        attrs = _parse(value)
        row = {}
        if isinstance(attrs, list):
            for attr in attrs:
                if not isinstance(attr, dict):
                    continue
                key = str(attr.get("id", "")).upper().strip()
                name = str(attr.get("name", "")).strip().lower()
                # The explicit outcome and its direct synonyms are never features.
                if key in {"ITEM_CONDITION", "CONDITION"} or name in {
                    "condición", "condicion", "condition"
                }:
                    continue
                val = attr.get("value_id") or attr.get("value_name")
                if key and val is not None and str(val).strip():
                    row[key] = str(val).strip().lower()
                    if i in selected_rows:
                        frequency[key] = frequency.get(key, 0) + 1
        parsed.append(row)
    attr_ids = sorted(frequency, key=lambda key: (-frequency[key], key))[:top_k]
    values = {
        f"attr_{key}": [row.get(key, "__MISSING__") for row in parsed]
        for key in attr_ids
    }
    return pd.DataFrame(values, index=raw.index), attr_ids


def build_advanced_features(raw: pd.DataFrame, attribute_ids=None, top_k=75):
    """Build features shared by train/test; returns (frame, chosen_attribute_ids)."""
    x = pd.DataFrame(index=raw.index)
    categorical = [
        "seller_id", "category_id", "listing_type_id", "buying_mode", "currency_id",
        "site_id", "status", "sub_status", "listing_source", "official_store_id",
        "catalog_product_id", "international_delivery_mode",
    ]
    for col in categorical:
        if col in raw:
            x[col] = raw[col].astype("string").fillna("__MISSING__")

    for col in ("price", "base_price", "original_price", "initial_quantity",
                "available_quantity", "sold_quantity"):
        if col in raw:
            value = pd.to_numeric(raw[col], errors="coerce").clip(lower=0)
            x[f"log_{col}"] = np.log1p(value).astype("float32")
    if {"price", "base_price"}.issubset(raw.columns):
        price = pd.to_numeric(raw["price"], errors="coerce")
        base = pd.to_numeric(raw["base_price"], errors="coerce").replace(0, np.nan)
        x["discount_ratio"] = ((base - price) / base).replace([np.inf, -np.inf], np.nan).astype("float32")

    title = raw.get("title", pd.Series("", index=raw.index)).fillna("").astype(str).str.lower()
    x["title_text"] = title
    x["title_chars"] = title.str.len().astype("float32")
    x["title_words"] = title.str.split().str.len().astype("float32")
    x["title_mentions_used"] = title.str.contains(
        r"\busad[oa]s?\b|segunda mano|reacondicionad[oa]", regex=True
    ).astype("int8")
    x["title_mentions_new"] = title.str.contains(
        r"\bnuev[oa]s?\b|sin uso|a estrenar|sellad[oa]", regex=True
    ).astype("int8")

    for col in ("warranty", "parent_item_id", "video_id"):
        if col in raw:
            x[f"{col}_missing"] = raw[col].isna().astype("int8")
    nested_counts = (
        ("attributes", "n_attributes"), ("variations", "n_variations"),
        ("pictures", "n_pictures"), ("tags", "n_tags"),
        ("descriptions", "n_descriptions"), ("deal_ids", "n_deal_ids"),
        ("coverage_areas", "n_coverage_areas"),
        ("non_mercado_pago_payment_methods", "n_other_payment_methods"),
    )
    for col, output in nested_counts:
        if col in raw:
            x[output] = raw[col].map(
                lambda value: len(_parse(value)) if isinstance(_parse(value), (list, tuple, dict)) else 0
            ).astype("int16")
    if "seller_address" in raw:
        for name, path in (("seller_state", ("state", "id")), ("seller_city", ("city", "id"))):
            x[name] = raw["seller_address"].map(lambda value, path=path: _nested(value, *path)).astype("string").fillna("__MISSING__")
    if "shipping" in raw:
        for name in ("mode", "free_shipping", "local_pick_up"):
            x[f"shipping_{name}"] = raw["shipping"].map(lambda value, name=name: _nested(value, name)).astype("string").fillna("__MISSING__")

    for col in ("date_created", "last_updated", "start_time", "stop_time"):
        if col in raw:
            dt = pd.to_datetime(raw[col], errors="coerce", utc=True)
            for part, values in (("month", dt.dt.month), ("hour", dt.dt.hour), ("weekday", dt.dt.dayofweek)):
                x[f"{col}_{part}"] = values.astype("Int8").astype("string").fillna("__MISSING__")

    # Human-readable attribute text, with the target-like field explicitly filtered.
    attr_text = []
    for value in raw.get("attributes", pd.Series([None] * len(raw), index=raw.index)):
        attrs = _parse(value)
        tokens = []
        if isinstance(attrs, list):
            for attr in attrs:
                if not isinstance(attr, dict):
                    continue
                key = str(attr.get("id", "")).upper().strip()
                name = str(attr.get("name", "")).strip().lower()
                if key in {"ITEM_CONDITION", "CONDITION"} or name in {
                    "condición", "condicion", "condition"
                }:
                    continue
                value_name = str(attr.get("value_name", "")).strip().lower()
                if name and value_name and value_name != "nan":
                    tokens.append(name.replace(" ", "_") + "=" + value_name.replace(" ", "_"))
        attr_text.append(" ".join(tokens))
    x["attributes_text"] = attr_text

    if attribute_ids is None:
        attr_frame, attribute_ids = _attributes(raw, top_k=top_k)
    else:
        parsed_rows = []
        for value in raw.get("attributes", pd.Series([None] * len(raw), index=raw.index)):
            attrs = _parse(value)
            row = {}
            if isinstance(attrs, list):
                for attr in attrs:
                    if not isinstance(attr, dict):
                        continue
                    key = str(attr.get("id", "")).upper().strip()
                    name = str(attr.get("name", "")).strip().lower()
                    if key in {"ITEM_CONDITION", "CONDITION"} or name in {"condición", "condicion", "condition"}:
                        continue
                    val = attr.get("value_id") or attr.get("value_name")
                    if key in attribute_ids and val is not None and str(val).strip():
                        row[f"attr_{key}"] = str(val).strip().lower()
            parsed_rows.append(row)
        attr_frame = pd.DataFrame(
            {f"attr_{key}": [row.get(f"attr_{key}", "__MISSING__") for row in parsed_rows]
             for key in attribute_ids}, index=raw.index
        )
    x = pd.concat([x, attr_frame], axis=1)
    if "seller_id" in x and "category_id" in x:
        x["seller_category"] = x["seller_id"].astype(str) + "__" + x["category_id"].astype(str)
    if "seller_id" in x and "listing_type_id" in x:
        x["seller_listing"] = x["seller_id"].astype(str) + "__" + x["listing_type_id"].astype(str)
    return x, list(attribute_ids)


def add_text_svd(train_features: pd.DataFrame, eval_features: pd.DataFrame,
                 n_components=72, random_state=42):
    """Fit TF-IDF and SVD on train only, then transform the evaluation rows."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    train_text = (train_features["title_text"].fillna("") + " " + train_features["attributes_text"].fillna("")).astype(str)
    eval_text = (eval_features["title_text"].fillna("") + " " + eval_features["attributes_text"].fillna("")).astype(str)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=55_000,
        strip_accents="unicode", sublinear_tf=True, dtype=np.float32,
    )
    sparse_train = vectorizer.fit_transform(train_text)
    sparse_eval = vectorizer.transform(eval_text)
    svd = TruncatedSVD(n_components=n_components, n_iter=5, random_state=random_state)
    reduced_train = svd.fit_transform(sparse_train).astype("float32")
    reduced_eval = svd.transform(sparse_eval).astype("float32")
    train_dense = train_features.drop(columns=["title_text", "attributes_text"]).copy()
    eval_dense = eval_features.drop(columns=["title_text", "attributes_text"]).copy()
    for j in range(n_components):
        name = f"text_svd_{j:03d}"
        train_dense[name] = reduced_train[:, j]
        eval_dense[name] = reduced_eval[:, j]
    return train_dense, eval_dense, vectorizer, svd
