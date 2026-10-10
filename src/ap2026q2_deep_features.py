"""Deep, leakage-aware feature engineering for AP_2026Q2 MercadoLibre JSONL."""
from __future__ import annotations
import ast, re
from typing import Any
import numpy as np
import pandas as pd

def parse_nested(value: Any):
    if isinstance(value, (dict, list, tuple)):
        return value
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError):
            return None
    return None

def _num(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _size(value):
    if not isinstance(value, str):
        return (np.nan, np.nan)
    m = re.search(r"(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)", value)
    return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)

def _shipping(row):
    z = parse_nested(row)
    if not isinstance(z, dict):
        return {}
    out = {
        "shipping_dimensions_missing": int(not z.get("dimensions")),
        "shipping_methods_n": len(z.get("methods") or []),
        "shipping_free_methods_n": len(z.get("free_methods") or []),
        "shipping_tags_n": len(z.get("tags") or []),
    }
    dims = z.get("dimensions")
    if isinstance(dims, str):
        parts = dims.split(",")
        a,b = _size(parts[0] if parts else None)
        out.update({"shipping_dim_a": a, "shipping_dim_b": b,
                    "shipping_weight_g": _num(parts[1]) if len(parts) > 1 else np.nan})
    else:
        out.update({"shipping_dim_a": np.nan, "shipping_dim_b": np.nan,
                    "shipping_weight_g": np.nan})
    out["shipping_methods_type"] = "|".join(sorted(
        str(m.get("type","")) for m in (z.get("methods") or []) if isinstance(m, dict)
    ))
    return out

def _variations(row, include_post_listing=False):
    z = parse_nested(row)
    if not isinstance(z, list):
        return {}
    prices, stocks, sold, attrs, pics, combos = [], [], [], [], [], []
    for v in z:
        if not isinstance(v, dict):
            continue
        p = _num(v.get("price")); q = _num(v.get("available_quantity"))
        if np.isfinite(p): prices.append(p)
        if np.isfinite(q): stocks.append(q)
        if include_post_listing:
            s = _num(v.get("sold_quantity"))
            if np.isfinite(s): sold.append(s)
        ac = v.get("attribute_combinations") or []
        attrs.extend(a.get("id") for a in ac if isinstance(a, dict) and a.get("id") is not None)
        combos.append("||".join(
            f"{a.get('id')}={a.get('value_id') or a.get('value_name')}"
            for a in ac if isinstance(a, dict)
        ))
        pics.extend(v.get("picture_ids") or [])
    out = {"variation_n": len(z), "variation_picture_n": len(pics),
           "variation_attribute_n": len(set(attrs)), "variation_combo_n": len(set(combos))}
    for name, vals in (("price", prices), ("stock", stocks), ("sold", sold)):
        out[f"variation_{name}_min"] = np.min(vals) if vals else np.nan
        out[f"variation_{name}_max"] = np.max(vals) if vals else np.nan
        out[f"variation_{name}_mean"] = np.mean(vals) if vals else np.nan
        out[f"variation_{name}_std"] = np.std(vals) if len(vals) > 1 else 0.0 if vals else np.nan
    out["variation_post_listing_used"] = int(include_post_listing)
    return out

def _pictures(row):
    z = parse_nested(row)
    if not isinstance(z, list):
        return {}
    widths, heights, max_widths, max_heights = [], [], [], []
    secure = 0
    for p in z:
        if not isinstance(p, dict):
            continue
        w,h = _size(p.get("size"))
        mw,mh = _size(p.get("max_size"))
        if np.isfinite(w): widths.append(w)
        if np.isfinite(h): heights.append(h)
        if np.isfinite(mw): max_widths.append(mw)
        if np.isfinite(mh): max_heights.append(mh)
        secure += int(bool(p.get("secure_url")))
    out = {"picture_n": len(z), "picture_secure_n": secure}
    for name, vals in (("width", widths), ("height", heights),
                       ("max_width", max_widths), ("max_height", max_heights)):
        out[f"picture_{name}_mean"] = np.mean(vals) if vals else np.nan
        out[f"picture_{name}_max"] = np.max(vals) if vals else np.nan
    out["picture_large_fraction"] = (
        np.mean([(w >= 500 and h >= 500) for w,h in zip(widths, heights)])
        if widths and heights and len(widths) == len(heights) else np.nan
    )
    return out

def _payment(row):
    z = parse_nested(row)
    if not isinstance(z, list):
        return {}
    ids, types, desc = [], [], []
    for p in z:
        if isinstance(p, dict):
            ids.append(str(p.get("id","")))
            types.append(str(p.get("type","")))
            desc.append(str(p.get("description","")).lower())
    return {
        "payment_n": len(z),
        "payment_id_n": len(set(ids)),
        "payment_type_n": len(set(types)),
        "payment_description_chars": sum(len(x) for x in desc),
        "payment_types_text": "|".join(sorted(set(types))),
    }

def _attributes_summary(row):
    z = parse_nested(row)
    if not isinstance(z, list):
        return {}
    ids, groups, values, names = [], [], [], []
    for a in z:
        if not isinstance(a, dict): continue
        ids.append(str(a.get("id","")))
        groups.append(str(a.get("attribute_group_id","")))
        values.append(str(a.get("value_id") or a.get("value_name") or ""))
        names.append(str(a.get("name","")))
    return {
        "attribute_id_n": len(set(ids)),
        "attribute_group_n": len(set(groups)),
        "attribute_value_n": len(set(values)),
        "attribute_name_chars": sum(len(x) for x in names),
        "attribute_missing_value_n": sum(v in {"","None","nan"} for v in values),
    }

def build_deep_features(raw: pd.DataFrame, base_features: pd.DataFrame | None = None,
                        include_post_listing=False) -> pd.DataFrame:
    """Add nested, ratio, temporal, text-shape and missingness features."""
    x = base_features.copy() if base_features is not None else pd.DataFrame(index=raw.index)
    for col, fn in (("shipping", _shipping), ("variations", lambda v: _variations(v, include_post_listing)),
                    ("pictures", _pictures), ("non_mercado_pago_payment_methods", _payment),
                    ("attributes", _attributes_summary)):
        rows = raw.get(col, pd.Series([None] * len(raw), index=raw.index)).map(fn)
        frame = pd.DataFrame(rows.tolist(), index=raw.index).add_prefix("deep_")
        x = pd.concat([x, frame], axis=1)
    # Seller country is available but was previously omitted.
    if "seller_address" in raw:
        x["seller_country"] = raw["seller_address"].map(
            lambda v: (parse_nested(v) or {}).get("country", {}).get("id")
            if isinstance(parse_nested(v), dict) else np.nan
        ).astype("string").fillna("__MISSING__")
    # Stable ratios; no target information is used.
    def num(c):
        return pd.to_numeric(raw.get(c, pd.Series(np.nan, index=raw.index)), errors="coerce")
    price, base, original = num("price"), num("base_price"), num("original_price")
    initial, available, sold = num("initial_quantity"), num("available_quantity"), num("sold_quantity")
    x["ratio_price_base"] = price / base.replace(0, np.nan)
    x["ratio_price_original"] = price / original.replace(0, np.nan)
    x["ratio_available_initial"] = available / initial.replace(0, np.nan)
    x["ratio_sold_initial"] = sold / initial.replace(0, np.nan)
    x["ratio_sold_plus_available_initial"] = (sold + available) / initial.replace(0, np.nan)
    x["stock_exhausted_flag"] = ((available <= 0) & initial.notna()).astype("int8")
    # Timestamp intervals and temporal completeness.
    dates = {c: pd.to_datetime(raw.get(c), errors="coerce", utc=True)
             for c in ("date_created","start_time","last_updated","stop_time")}
    x["hours_created_to_update"] = (dates["last_updated"] - dates["date_created"]).dt.total_seconds() / 3600
    x["hours_start_to_stop"] = (dates["stop_time"] - dates["start_time"]).dt.total_seconds() / 3600
    x["created_year"] = dates["date_created"].dt.year.astype("Int16").astype("string").fillna("__MISSING__")
    x["created_day"] = dates["date_created"].dt.day.astype("Int16").astype("string").fillna("__MISSING__")
    x["created_is_weekend"] = dates["date_created"].dt.dayofweek.isin([5,6]).astype("int8")
    # Title/subtitle shape. Subtitle is absent in this snapshot but kept for reuse.
    title = raw.get("title", pd.Series("", index=raw.index)).fillna("").astype(str)
    subtitle = raw.get("subtitle", pd.Series("", index=raw.index)).fillna("").astype(str)
    for name, text in (("title", title), ("subtitle", subtitle)):
        x[f"{name}_digits"] = text.str.count(r"\d").astype("float32")
        x[f"{name}_uppercase"] = text.str.count(r"[A-ZA�IOU�]").astype("float32")
        x[f"{name}_punctuation"] = text.str.count(r"[^\w\s]").astype("float32")
        x[f"{name}_code_like"] = text.str.contains(r"\b[A-Z]{1,5}[- ]?\d{2,}\b", regex=True).astype("int8")
    # Missingness profile: preserve NA semantics instead of converting to zero.
    missing_frame = raw.drop(columns=["condition"], errors="ignore")
    x["raw_missing_n"] = missing_frame.isna().sum(axis=1).astype("int16")
    x["raw_missing_fraction"] = missing_frame.isna().mean(axis=1).astype("float32")
    return x

def add_outlier_flags(train_x: pd.DataFrame, eval_x: pd.DataFrame, columns=None):
    """Fit IQR fences on train only and append outlier flags/capped values."""
    cols = columns or train_x.select_dtypes(include=np.number).columns
    tr, ev = train_x.copy(), eval_x.copy()
    for c in cols:
        q1, q3 = tr[c].quantile([.25,.75])
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0: continue
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        tr[c + "_iqr_outlier"] = ((tr[c] < lo) | (tr[c] > hi)).astype("int8")
        ev[c + "_iqr_outlier"] = ((ev[c] < lo) | (ev[c] > hi)).astype("int8")
        tr[c + "_winsor"] = tr[c].clip(lo, hi)
        ev[c + "_winsor"] = ev[c].clip(lo, hi)
    return tr, ev


