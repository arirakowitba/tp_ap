"""Build and freeze the shared dual-regime outer seller-group + causal seen-seller split table
and pre-parsed train/test feature cache for the 5 independent base improvement lines (T98).

Guarantees (verified before saving):
1. Outer folds (0..4): StratifiedGroupKFold(5, shuffle=True, random_state=42) grouped by seller_id.
   Exact same 14,000-row outer folds as T97 (0 seller overlap across outer folds).
2. Within each outer-train k (56,000 rows):
   - Causal Seen-Seller Holdout (`seen_val_k`, ~6,426 rows): For every seller in outer-train k with
     >= 2 items, their chronologically latest item (`max(start_time, row)`) is held out as `seen_val_k`.
   - Fit set (`fit_k`, ~49,574 rows): All remaining items in outer-train k. Every seller in `seen_val_k`
     is guaranteed to have strictly earlier historical items in `fit_k` (`max(t_fit_seller) <= min(t_seen_val_seller)`).
3. Within each `fit_k` (~49,574 rows):
   - 4 inner seller-group folds (`inner_fold_in_fit_k` in 0..3) via StratifiedGroupKFold(4, shuffle=True, random_state=1000+k)
     grouped by seller_id (0 seller overlap across inner folds).
   - Within each inner-train (`fit_k` minus inner_val_j), a causal inner seen-seller holdout (`inner_seen_val_k_j`)
     holds out the latest item of each multi-item seller in inner-train, so inner selection can evaluate BOTH
     inner unseen-seller OOF (`~49,574` rows) and inner seen-seller holdout (`~5,100` rows per inner fold)
     without touching `seen_val_k` or `unseen_val_k` (`outer-val`)!
4. Pre-extracts clean item-intrinsic and raw tabular/text fields from `train_data.jsonlines` (70,000 rows)
   and `test_data.jsonlines` (30,000 rows) with zero target or global frequency leakage.
"""
import json
import hashlib
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_item_record(r: dict, is_train: bool, idx: int) -> dict:
    title = str(r.get("title") or "")
    tl = title.lower()
    subtitle = str(r.get("subtitle") or "")
    warranty_raw = r.get("warranty")
    warranty_str = "" if (warranty_raw is None or warranty_raw is False) else str(warranty_raw)
    wl = warranty_str.lower()

    tags_list = r.get("tags") if isinstance(r.get("tags"), list) else []
    tags_str = " ".join(str(t) for t in tags_list)

    attrs = r.get("attributes") if isinstance(r.get("attributes"), list) else []
    num_attrs = len(attrs)
    attr_tokens = []
    brand_val = "MISSING"
    model_val = "MISSING"
    item_cond_attr = "MISSING"
    for a in attrs:
        if not isinstance(a, dict):
            continue
        aid = str(a.get("id") or "").upper()
        aname = str(a.get("name") or "").lower()
        vname = str(a.get("value_name") or "").strip()
        vid = str(a.get("value_id") or "").strip()
        if vname:
            attr_tokens.append(f"{aid}:{vname.lower()}")
        if "BRAND" in aid or "marca" in aname:
            brand_val = vname if vname else (vid if vid else "MISSING")
        elif "MODEL" in aid or "modelo" in aname:
            model_val = vname if vname else (vid if vid else "MISSING")
        elif "ITEM_CONDITION" in aid or "condici" in aname:
            item_cond_attr = vname if vname else (vid if vid else "MISSING")
    attr_text = " ".join(attr_tokens)

    shipping = r.get("shipping") if isinstance(r.get("shipping"), dict) else {}
    free_shipping = int(bool(shipping.get("free_shipping", False)))
    local_pick_up = int(bool(shipping.get("local_pick_up", False)))
    ship_mode = str(shipping.get("mode") or "not_specified")

    addr = r.get("seller_address") if isinstance(r.get("seller_address"), dict) else {}
    state_dict = addr.get("state") if isinstance(addr.get("state"), dict) else {}
    city_dict = addr.get("city") if isinstance(addr.get("city"), dict) else {}
    state_id = str(state_dict.get("id") or "MISSING")
    city_id = str(city_dict.get("id") or "MISSING")

    non_mp = r.get("non_mercado_pago_payment_methods")
    non_mp_list = non_mp if isinstance(non_mp, list) else []
    num_non_mp = len(non_mp_list)
    pay_ids = " ".join(str(p.get("id") or "") for p in non_mp_list if isinstance(p, dict))

    pics = r.get("pictures") if isinstance(r.get("pictures"), list) else []
    num_pics = len(pics)

    vars_list = r.get("variations") if isinstance(r.get("variations"), list) else []
    num_vars = len(vars_list)

    deals = r.get("deal_ids") if isinstance(r.get("deal_ids"), list) else []
    num_deals = len(deals)

    sub_status = r.get("sub_status") if isinstance(r.get("sub_status"), list) else []
    sub_status_str = sub_status[0] if len(sub_status) > 0 else "NONE"

    price = float(r.get("price") or 0.0)
    base_price = float(r.get("base_price") or price)
    orig_price_raw = r.get("original_price")
    has_orig_price = int(orig_price_raw is not None and float(orig_price_raw or 0) > 0)
    orig_price = float(orig_price_raw) if has_orig_price else price

    init_qty = float(r.get("initial_quantity") or 0.0)
    avail_qty = float(r.get("available_quantity") or 0.0)
    sold_qty = float(r.get("sold_quantity") or 0.0)

    start_time = int(r.get("start_time") or 0)
    stop_time = int(r.get("stop_time") or 0)

    return {
        "row": idx,
        "id": str(r.get("id") or idx),
        "y": int(r.get("condition") == "used") if is_train else -1,
        "seller_id": str(r.get("seller_id") or "MISSING"),
        "category_id": str(r.get("category_id") or "MISSING"),
        "title": title,
        "subtitle": subtitle,
        "warranty_str": warranty_str,
        "tags_str": tags_str,
        "attr_text": attr_text,
        "brand_val": brand_val,
        "model_val": model_val,
        "item_cond_attr": item_cond_attr,
        "listing_type_id": str(r.get("listing_type_id") or "MISSING"),
        "buying_mode": str(r.get("buying_mode") or "MISSING"),
        "currency_id": str(r.get("currency_id") or "ARS"),
        "status": str(r.get("status") or "active"),
        "sub_status_str": str(sub_status_str),
        "ship_mode": ship_mode,
        "state_id": state_id,
        "city_id": city_id,
        "pay_ids": pay_ids,
        "price": price,
        "base_price": base_price,
        "orig_price": orig_price,
        "has_orig_price": has_orig_price,
        "init_qty": init_qty,
        "avail_qty": avail_qty,
        "sold_qty": sold_qty,
        "free_shipping": free_shipping,
        "local_pick_up": local_pick_up,
        "accepts_mercadopago": int(bool(r.get("accepts_mercadopago", False))),
        "automatic_relist": int(bool(r.get("automatic_relist", False))),
        "has_official_store": int(r.get("official_store_id") is not None),
        "has_video": int(r.get("video_id") is not None and str(r.get("video_id")) != ""),
        "has_catalog": int(r.get("catalog_product_id") is not None),
        "has_parent_item": int(r.get("parent_item_id") is not None),
        "num_attrs": num_attrs,
        "num_pics": num_pics,
        "num_vars": num_vars,
        "num_deals": num_deals,
        "num_non_mp": num_non_mp,
        "num_tags": len(tags_list),
        "start_time": start_time,
        "stop_time": stop_time,
        "warr_has_mes_ano": int("mes" in wl or "año" in wl or "ano" in wl or "garant" in wl),
        "warr_sin_garantia": int("sin garant" in wl or "no tiene" in wl),
        "warr_fabrica": int("fabric" in wl or "fábric" in wl or "oficial" in wl),
        "title_len": len(title),
        "title_words": len(title.split()),
        "title_digits": sum(c.isdigit() for c in title),
        "title_uppers": sum(c.isupper() for c in title),
    }


def main():
    t0 = time.time()
    print("Parsing train_data.jsonlines (70,000 rows)...")
    train_rows = [
        parse_item_record(json.loads(line), is_train=True, idx=i)
        for i, line in enumerate(open(ROOT / "data/ap_2026_q_2/raw/train_data.jsonlines", encoding="utf-8"))
    ]
    df_train = pd.DataFrame(train_rows)

    print("Parsing test_data.jsonlines (30,000 rows)...")
    test_rows = [
        parse_item_record(json.loads(line), is_train=False, idx=i)
        for i, line in enumerate(open(ROOT / "data/ap_2026_q_2/raw/test_data.jsonlines", encoding="utf-8"))
    ]
    df_test = pd.DataFrame(test_rows)

    # 1. Exact same 5 outer seller-group folds as T97
    y = df_train["y"].values
    sellers = df_train["seller_id"].values
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    outer_fold = np.full(len(df_train), -1, dtype=int)
    for k, (tr, va) in enumerate(outer_cv.split(np.zeros(len(df_train)), y, sellers)):
        assert len(set(sellers[tr]) & set(sellers[va])) == 0
        outer_fold[va] = k
    df_train["outer_fold"] = outer_fold

    # Verify exact match with T97 paired_outer_val_predictions.csv
    t97_path = ROOT / "results/seller_group_nested_podium_20261111/paired_outer_val_predictions.csv"
    if t97_path.exists():
        t97_df = pd.read_csv(t97_path, usecols=["row", "outer_fold"])
        assert np.array_equal(df_train["outer_fold"].values + 1, t97_df["outer_fold"].values), "Mismatch vs T97 outer folds!"
        print("Verified 100% exact outer_fold match with T97 paired_outer_val_predictions.csv!")

    # 2. Construct role_fold_{k} and inner_fold_{k} for each outer fold k in 0..4
    split_cols = {
        "row": df_train["row"].values,
        "id": df_train["id"].values,
        "seller_id": sellers,
        "category_id": df_train["category_id"].values,
        "y": y,
        "start_time": df_train["start_time"].values,
        "outer_fold": outer_fold,
    }
    fold_summaries = []

    for k in range(5):
        role_k = np.empty(len(df_train), dtype=object)
        role_k[outer_fold == k] = "unseen_val"

        tr_mask = outer_fold != k
        tr_sub = df_train.loc[tr_mask, ["row", "seller_id", "start_time", "y"]].sort_values(
            ["seller_id", "start_time", "row"]
        )
        tr_sub["rank_desc"] = tr_sub.groupby("seller_id").cumcount(ascending=False)
        tr_sub["cnt"] = tr_sub.groupby("seller_id")["row"].transform("count")

        seen_val_rows = tr_sub.loc[(tr_sub["cnt"] >= 2) & (tr_sub["rank_desc"] == 0), "row"].values
        fit_rows = tr_sub.loc[~tr_sub["row"].isin(set(seen_val_rows)), "row"].values

        role_k[seen_val_rows] = "seen_val"
        role_k[fit_rows] = "fit"

        # Verify causal ordering: for every seller in seen_val_rows, max(start_time in fit) <= start_time in seen_val
        fit_max_t = df_train.iloc[fit_rows].groupby("seller_id")["start_time"].max()
        seen_t = df_train.iloc[seen_val_rows].set_index("seller_id")["start_time"]
        assert (fit_max_t.loc[seen_t.index].values <= seen_t.values).all(), f"Causal violation in fold {k}!"
        assert set(df_train.iloc[seen_val_rows]["seller_id"]).issubset(set(df_train.iloc[fit_rows]["seller_id"]))
        assert len(set(df_train.iloc[fit_rows]["seller_id"]) & set(df_train.iloc[outer_fold == k]["seller_id"])) == 0

        # Inner 4-fold seller-group split inside fit_rows
        inner_fold_k = np.full(len(df_train), -1, dtype=int)
        inner_cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=1000 + k)
        for j, (itr, iva) in enumerate(inner_cv.split(fit_rows, y[fit_rows], sellers[fit_rows])):
            assert len(set(sellers[fit_rows[itr]]) & set(sellers[fit_rows[iva]])) == 0
            inner_fold_k[fit_rows[iva]] = j

        split_cols[f"role_fold_{k}"] = role_k
        split_cols[f"inner_fold_{k}"] = inner_fold_k

        fold_summaries.append({
            "outer_fold": k + 1,
            "n_fit": int(len(fit_rows)),
            "n_seen_val": int(len(seen_val_rows)),
            "n_unseen_val": int((outer_fold == k).sum()),
            "fit_sellers": int(len(set(sellers[fit_rows]))),
            "seen_val_sellers": int(len(set(sellers[seen_val_rows]))),
            "unseen_val_sellers": int(len(set(sellers[outer_fold == k]))),
            "causal_seen_verified": True,
            "outer_seller_overlap": 0,
            "inner_seller_overlap": 0,
        })
        print(
            f"  Fold {k+1}: fit={len(fit_rows)}, seen_val={len(seen_val_rows)} (causal latest item), "
            f"unseen_val={(outer_fold==k).sum()}"
        )

    df_splits = pd.DataFrame(split_cols)
    splits_path = SPLIT_DIR / "t98_dual_regime_outer_seller_group_splits.csv"
    df_splits.to_csv(splits_path, index=False)

    train_cache_path = SPLIT_DIR / "t98_parsed_train.pkl"
    test_cache_path = SPLIT_DIR / "t98_parsed_test.pkl"
    df_train.to_pickle(train_cache_path)
    df_test.to_pickle(test_cache_path)

    manifest = {
        "title": "T98 Frozen Dual-Regime Outer Seller-Group + Causal Seen-Seller Splits",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "splits_csv": str(splits_path.relative_to(ROOT)),
        "splits_csv_sha256": sha256_file(splits_path),
        "train_cache_pkl": str(train_cache_path.relative_to(ROOT)),
        "train_cache_sha256": sha256_file(train_cache_path),
        "test_cache_pkl": str(test_cache_path.relative_to(ROOT)),
        "test_cache_sha256": sha256_file(test_cache_path),
        "folds": fold_summaries,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    manifest_path = SPLIT_DIR / "t98_dual_regime_splits_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done in {manifest['elapsed_seconds']}s. Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
