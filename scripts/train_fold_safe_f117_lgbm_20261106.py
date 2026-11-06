"""Fold-safe F117 / LightGBM Benchmark under Strict Seller-Group Partition.

Complies strictly with CX-0111, CX-0112, CX-0115, and CX-0116:
- Zero pseudo-labels, zero Kaggle quota, zero transductive leakage.
- Exact shared seller-group split (seed=20261104, pure=False):
    fit: 44,834 | tune: 12,542 | confirm: 12,624
- All 117 features reconstructed fold-safe:
    - Global leaky columns (freq_, seller_, sell_mean_, rel_price_cat) recomputed strictly within outer fit.
    - Text model (stateless hashing + NB/LR) and Target Encoding generated fold-safe via nested GroupKFold on outer fit.
    - Tune threshold frozen on tune only; single evaluation on confirm.
    - Paired evaluation against baseline: accuracy, logloss, fixes, breaks, seen/new, 3000-rep seller bootstrap 95% CI.
"""

import argparse
import gc
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold


def digest(v):
    return hashlib.sha256("\n".join(map(str, v)).encode()).hexdigest()


def split(s, seed=20261104, pure=False):
    rng = np.random.default_rng(seed)
    u = np.unique(s)
    rng.shuffle(u)
    if pure:
        k = int(0.15 * len(u))
        tu = np.flatnonzero(np.isin(s, u[:k]))
        co = np.flatnonzero(np.isin(s, u[k : 2 * k]))
        fi = np.flatnonzero(np.isin(s, u[2 * k :]))
        return fi, tu, co
    k = int(0.05 * len(u))
    tu_new = set(u[:k])
    co_new = set(u[k : 2 * k])
    groups = {}
    for i, x in enumerate(s):
        groups.setdefault(x, []).append(i)
    eligible = []
    for x in u[2 * k :]:
        ix = np.asarray(groups[x])
        rng.shuffle(ix)
        eligible.extend(ix[1:])
    eligible = np.asarray(eligible, int)
    rng.shuffle(eligible)
    n = min(9000, len(eligible) // 2)
    tu = np.r_[np.flatnonzero(np.isin(s, list(tu_new))), eligible[:n]]
    co = np.r_[np.flatnonzero(np.isin(s, list(co_new))), eligible[n : 2 * n]]
    fi = np.setdiff1d(np.arange(len(s)), np.r_[tu, co])
    assert not (set(fi) & set(tu) or set(fi) & set(co) or set(tu) & set(co))
    return fi, tu, co


def lst(x):
    return x if isinstance(x, list) else []


def dims(p):
    try:
        w, h = p.get("max_size", p.get("size", "0x0")).split("x")
        return int(w), int(h)
    except Exception:
        return 0, 0


def picnum(p):
    m = re.search(r"/(\d+)-MLA\d+_(\d+)-", p.get("url", "") or "")
    return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)


def extract_intrinsic_features(rows):
    """Extract intrinsic item features without any cross-row or label leakage."""
    t0 = time.time()
    feats = {}
    N = len(rows)

    price = np.array([float(r.get("price") or 0.0) for r in rows], dtype=np.float32)
    base_price = np.array([float(r.get("base_price") or 0.0) for r in rows], dtype=np.float32)
    init_q = np.array([float(r.get("initial_quantity") or 0.0) for r in rows], dtype=np.float32)
    avail_q = np.array([float(r.get("available_quantity") or 0.0) for r in rows], dtype=np.float32)
    sold_q = np.array([float(r.get("sold_quantity") or 0.0) for r in rows], dtype=np.float32)

    feats["price"] = price
    feats["base_price"] = base_price
    feats["initial_quantity"] = init_q
    feats["available_quantity"] = avail_q
    feats["sold_quantity"] = sold_q
    feats["log_price"] = np.log1p(np.maximum(0, price))
    feats["price_eq_base"] = (price == base_price).astype(np.float32)
    feats["sold_ratio"] = sold_q / (init_q + 1.0)
    feats["avail_ratio"] = avail_q / (init_q + 1.0)
    feats["init_minus_avail"] = init_q - avail_q
    feats["price_cents"] = ((price * 100) % 100).round()
    feats["price_round100"] = ((price % 100) == 0).astype(np.float32)
    feats["price_round10"] = ((price % 10) == 0).astype(np.float32)
    feats["price_end9"] = ((np.round(price) % 10) == 9).astype(np.float32)

    feats["has_original_price"] = np.array([1.0 if r.get("original_price") is not None else 0.0 for r in rows], dtype=np.float32)
    feats["official_store"] = np.array([1.0 if r.get("official_store_id") is not None else 0.0 for r in rows], dtype=np.float32)
    feats["has_video"] = np.array([1.0 if r.get("video_id") is not None else 0.0 for r in rows], dtype=np.float32)
    feats["has_catalog"] = np.array([1.0 if r.get("catalog_product_id") is not None else 0.0 for r in rows], dtype=np.float32)
    feats["has_parent"] = np.array([1.0 if r.get("parent_item_id") is not None else 0.0 for r in rows], dtype=np.float32)
    feats["automatic_relist"] = np.array([1.0 if r.get("automatic_relist") else 0.0 for r in rows], dtype=np.float32)
    feats["accepts_mp"] = np.array([1.0 if r.get("accepts_mercadopago") else 0.0 for r in rows], dtype=np.float32)
    feats["n_deals"] = np.array([float(len(lst(r.get("deal_ids")))) for r in rows], dtype=np.float32)
    feats["currency_usd"] = np.array([1.0 if r.get("currency_id") == "USD" else 0.0 for r in rows], dtype=np.float32)

    # Tags
    for t_tag in ["dragged_bids_and_visits", "good_quality_thumbnail", "dragged_visits", "free_relist", "poor_quality_thumbnail"]:
        feats[f"tag_{t_tag}"] = np.array([1.0 if t_tag in lst(r.get("tags")) else 0.0 for r in rows], dtype=np.float32)
    feats["n_tags"] = np.array([float(len(lst(r.get("tags")))) for r in rows], dtype=np.float32)

    # Payment methods
    for code in ["MLATB", "MLAWC", "MLAMO", "MLAOT", "MLACD", "MLABC", "MLAVE", "MLAMC", "MLAAM", "MLADC", "MLAVS", "MLAMS", "MLAWT", "MLAMP", "MLAAB"]:
        feats[f"pm_{code}"] = np.array([1.0 if code in [m.get("id") for m in lst(r.get("non_mercado_pago_payment_methods"))] else 0.0 for r in rows], dtype=np.float32)
    feats["n_pm"] = np.array([float(len(lst(r.get("non_mercado_pago_payment_methods")))) for r in rows], dtype=np.float32)

    # Shipping
    shipping = [r.get("shipping") or {} for r in rows]
    feats["ship_local_pick_up"] = np.array([1.0 if sh.get("local_pick_up") else 0.0 for sh in shipping], dtype=np.float32)
    feats["ship_free"] = np.array([1.0 if sh.get("free_shipping") else 0.0 for sh in shipping], dtype=np.float32)
    feats["ship_dims"] = np.array([1.0 if sh.get("dimensions") is not None else 0.0 for sh in shipping], dtype=np.float32)
    feats["ship_n_methods"] = np.array([float(len(lst(sh.get("methods")))) for sh in shipping], dtype=np.float32)
    feats["ship_n_tags"] = np.array([float(len(lst(sh.get("tags")))) for sh in shipping], dtype=np.float32)

    # Pictures
    n_pics, pic0_w, pic0_h, pic_max_area, pic_mean_area, pic_square = [], [], [], [], [], []
    pic_prefix, pic_code, pic_code_min, pic_code_max, thumb_code = [], [], [], [], []
    for r in rows:
        pics = lst(r.get("pictures"))
        n_pics.append(float(len(pics)))
        if pics:
            p0 = dims(pics[0])
            pic0_w.append(float(p0[0]))
            pic0_h.append(float(p0[1]))
            pic_square.append(1.0 if p0[0] == p0[1] and p0[0] > 0 else 0.0)
            areas = [dims(p)[0] * dims(p)[1] for p in pics]
            pic_max_area.append(float(max(areas)))
            pic_mean_area.append(float(np.mean(areas)))
            pnum0 = picnum(pics[0])
            pic_prefix.append(float(pnum0[0]))
            pic_code.append(float(pnum0[1]))
            all_codes = [picnum(p)[1] for p in pics if picnum(p)[1] >= 0]
            pic_code_min.append(float(min(all_codes)) if all_codes else -1.0)
            pic_code_max.append(float(max(all_codes)) if all_codes else -1.0)
        else:
            pic0_w.append(-1.0)
            pic0_h.append(-1.0)
            pic_square.append(0.0)
            pic_max_area.append(-1.0)
            pic_mean_area.append(-1.0)
            pic_prefix.append(-1.0)
            pic_code.append(-1.0)
            pic_code_min.append(-1.0)
            pic_code_max.append(-1.0)
        u = r.get("thumbnail") or ""
        m = re.search(r"_(\d+)-", u)
        thumb_code.append(float(m.group(1)) if m else -1.0)

    feats["n_pics"] = np.array(n_pics, dtype=np.float32)
    feats["pic0_w"] = np.array(pic0_w, dtype=np.float32)
    feats["pic0_h"] = np.array(pic0_h, dtype=np.float32)
    feats["pic_max_area"] = np.array(pic_max_area, dtype=np.float32)
    feats["pic_mean_area"] = np.array(pic_mean_area, dtype=np.float32)
    feats["pic_square"] = np.array(pic_square, dtype=np.float32)
    feats["pic_prefix"] = np.array(pic_prefix, dtype=np.float32)
    feats["pic_code"] = np.array(pic_code, dtype=np.float32)
    feats["pic_code_min"] = np.array(pic_code_min, dtype=np.float32)
    feats["pic_code_max"] = np.array(pic_code_max, dtype=np.float32)
    feats["thumb_code"] = np.array(thumb_code, dtype=np.float32)

    # Variations & Attributes
    feats["n_variations"] = np.array([float(len(lst(r.get("variations")))) for r in rows], dtype=np.float32)
    feats["n_attributes"] = np.array([float(len(lst(r.get("attributes")))) for r in rows], dtype=np.float32)
    feats["var_sold"] = np.array([float(sum(v.get("sold_quantity", 0) or 0 for v in lst(r.get("variations")))) for r in rows], dtype=np.float32)

    # Dates and Durations
    ref = pd.Timestamp("2013-01-01", tz="UTC")
    dates_created = pd.to_datetime([r.get("date_created") for r in rows], utc=True, errors="coerce")
    dates_updated = pd.to_datetime([r.get("last_updated") for r in rows], utc=True, errors="coerce")
    st = pd.to_datetime([r.get("start_time") for r in rows], unit="ms", utc=True, errors="coerce")
    sp = pd.to_datetime([r.get("stop_time") for r in rows], unit="ms", utc=True, errors="coerce")

    feats["created_days"] = np.nan_to_num((dates_created - ref).total_seconds() / 86400.0, nan=0.0).astype(np.float32)
    feats["updated_days"] = np.nan_to_num((dates_updated - ref).total_seconds() / 86400.0, nan=0.0).astype(np.float32)
    feats["start_days"] = np.nan_to_num((st - ref).total_seconds() / 86400.0, nan=0.0).astype(np.float32)
    feats["stop_days"] = np.nan_to_num((sp - ref).total_seconds() / 86400.0, nan=0.0).astype(np.float32)
    feats["dur_days"] = np.nan_to_num((sp - st).total_seconds() / 86400.0, nan=0.0).astype(np.float32)
    feats["upd_minus_created"] = np.nan_to_num((dates_updated - dates_created).total_seconds() / 86400.0, nan=0.0).astype(np.float32)
    feats["start_minus_created"] = np.nan_to_num((st - dates_created).total_seconds(), nan=0.0).astype(np.float32)
    feats["created_hour"] = np.nan_to_num(dates_created.hour, nan=0.0).astype(np.float32)
    feats["created_dow"] = np.nan_to_num(dates_created.dayofweek, nan=0.0).astype(np.float32)
    feats["created_sec"] = np.nan_to_num(dates_created.second, nan=0.0).astype(np.float32)

    # Warranty cues
    warranties = [str(r.get("warranty") or "") for r in rows]
    feats["warranty_null"] = np.array([1.0 if not r.get("warranty") else 0.0 for r in rows], dtype=np.float32)
    feats["warranty_len"] = np.array([float(len(w)) for w in warranties], dtype=np.float32)
    for kw, pat in {
        "w_sin": r"\bsin\b", "w_no": r"\bno\b", "w_meses": r"mes", "w_anio": r"a[ñn]o",
        "w_fabrica": r"f[aá]brica", "w_oficial": r"oficial", "w_usado": r"usad", "w_nuevo": r"nuev",
        "w_dias": r"d[ií]as", "w_garantia": r"garant", "w_reputacion": r"reputaci|calificaci",
        "w_prueba": r"prueba|probad|funciona"
    }.items():
        feats[kw] = np.array([1.0 if bool(re.search(pat, w.lower())) else 0.0 for w in warranties], dtype=np.float32)

    # Title cues
    titles = [str(r.get("title") or "") for r in rows]
    feats["title_len"] = np.array([float(len(t)) for t in titles], dtype=np.float32)
    feats["title_words"] = np.array([float(len(t.split())) for t in titles], dtype=np.float32)
    feats["title_upper_frac"] = np.array([sum(ch.isupper() for ch in t) / max(len(t), 1) for t in titles], dtype=np.float32)
    feats["title_digits"] = np.array([float(sum(ch.isdigit() for ch in t)) for t in titles], dtype=np.float32)
    for kw, pat in {
        "t_usado": r"usad", "t_nuevo": r"nuev", "t_original": r"original",
        "t_antiguo": r"antig|vintage|retro|colecci", "t_libro": r"libro", "t_lote": r"lote",
        "t_impecable": r"impecable|excelente estado|muy buen estado"
    }.items():
        feats[kw] = np.array([1.0 if bool(re.search(pat, t.lower())) else 0.0 for t in titles], dtype=np.float32)

    # Raw categorical identifiers (to be treated fold-safe)
    cat_df = pd.DataFrame({
        "category_id": [str(r.get("category_id") or "MISSING") for r in rows],
        "listing_type_id": [str(r.get("listing_type_id") or "MISSING") for r in rows],
        "buying_mode": [str(r.get("buying_mode") or "MISSING") for r in rows],
        "status": [str(r.get("status") or "MISSING") for r in rows],
        "seller_id": [str(r.get("seller_id") or "MISSING") for r in rows],
        "sub_status": ["|".join(lst(r.get("sub_status"))) for r in rows],
        "ship_mode": [str(sh.get("mode") or "MISSING") for sh in shipping],
        "ship_tags": ["|".join(sorted(map(str, lst(sh.get("tags"))))) for sh in shipping],
        "state": [str(((r.get("seller_address") or {}).get("state") or {}).get("id") or "MISSING") for r in rows],
        "city": [str(((r.get("seller_address") or {}).get("city") or {}).get("name") or "MISSING").lower().strip() for r in rows],
    })

    # Full text representations
    text_list = []
    for r in rows:
        t = str(r.get("title") or "")
        w = str(r.get("warranty") or "")
        attr_vals = " ".join(str(a.get("value_name") or a.get("name") or "") for a in lst(r.get("attributes")) if isinstance(a, dict))
        text_list.append(f"{t} WARRANTY {w} ATTR {attr_vals}")

    df_num = pd.DataFrame(feats)
    print(f"Extracted intrinsic features: {df_num.shape} in {time.time()-t0:.2f}s", flush=True)
    return df_num, cat_df, np.asarray(text_list, dtype=object)


def text_fit_nb(X, y, tr, va):
    """Supervised NB log-count ratio strictly within tr, predicting on va."""
    pos = np.asarray(X[tr][y[tr] == 1].sum(axis=0)).ravel() + 1.0
    neg = np.asarray(X[tr][y[tr] == 0].sum(axis=0)).ravel() + 1.0
    ratio = np.log((pos / pos.sum()) / (neg / neg.sum()))
    m = LogisticRegression(C=2.0, solver="liblinear", max_iter=300, random_state=104)
    m.fit(X[tr].multiply(ratio), y[tr])
    return m.predict_proba(X[va].multiply(ratio))[:, 1]


def compute_metrics(y, p, base, seller, seen):
    pred = p >= 0.5
    delta = (pred == y).astype(int) - (base == y).astype(int)
    by = pd.Series(delta).groupby(seller).sum().to_numpy()
    rng = np.random.default_rng(104)
    boot = np.asarray([by[rng.integers(0, len(by), len(by))].sum() for _ in range(3000)])
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "logloss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
        "hits": int((pred == y).sum()),
        "net": int(delta.sum()),
        "fixes": int((delta == 1).sum()),
        "breaks": int((delta == -1).sum()),
        "seller_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "seen_new": {
            name: {
                "n": int(mask.sum()),
                "accuracy": float(accuracy_score(y[mask], pred[mask])) if mask.any() else None,
                "net": int(delta[mask].sum()) if mask.any() else 0,
            }
            for name, mask in [("seen", seen), ("new", ~seen)]
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Train Fold-Safe F117 / LightGBM Benchmark")
    parser.add_argument("--data", default="data/ap_2026_q_2/raw/train_data.jsonlines", help="Path to train_data.jsonlines")
    parser.add_argument("--out", default="results/fold_safe_f117_lgbm_20261106", help="Output directory")
    parser.add_argument("--seed", type=int, default=20261104, help="Split seed")
    parser.add_argument("--pure", action="store_true", help="Pure seller group split")
    parser.add_argument("--baseline_pairs", default="results/test_depth5/pairs.csv", help="Baseline pairs.csv for direct comparison")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("=" * 70, flush=True)
    print("FOLD-SAFE F117 / LIGHTGBM BENCHMARK ON STRICT SELLER-GROUP SPLIT", flush=True)
    print("=" * 70, flush=True)

    # 1. Load data
    print(f"Loading {args.data}...", flush=True)
    rows = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    N = len(rows)
    assert N == 70000, f"Expected 70,000 rows, got {N}"

    y = np.asarray([int(r["condition"] == "used") for r in rows], dtype=int)
    sellers = np.asarray([str(r["seller_id"]) for r in rows])
    ids = np.asarray([str(r["id"]) for r in rows])
    assert len(set(ids)) == 70000

    # 2. Split seller-group
    fi, tu, co = split(sellers, args.seed, args.pure)
    print(f"SPLITS: fit={len(fi)}, tune={len(tu)}, confirm={len(co)}", flush=True)
    split_hashes = {n: digest(ids[ix]) for n, ix in [("fit", fi), ("tune", tu), ("confirm", co)]}
    print("Split hashes:", json.dumps(split_hashes, indent=2), flush=True)

    # Verify split hashes against CX-0115 / CX-0116
    expected_fit = "0b15e2c6f97fe810fbdd4616274f813b19986c015a49dc38ba2b253467b1025d"
    expected_tune = "d0bcf08902d3517f5f7d6c0543f7bc03fb12204faf6dd644734a542fbfcc4a2c"
    expected_confirm = "ced0e88f1d8c26cc21e66f948c5c1a5a45cd0a47d37e9f508fb7ea3cc8288cd2"
    assert split_hashes["fit"] == expected_fit, "Fit hash mismatch!"
    assert split_hashes["tune"] == expected_tune, "Tune hash mismatch!"
    assert split_hashes["confirm"] == expected_confirm, "Confirm hash mismatch!"
    print(">>> Split verification PASSED bit-for-bit.", flush=True)

    # 3. Extract intrinsic features
    df_num, cat_df, text_arr = extract_intrinsic_features(rows)
    del rows
    gc.collect()

    # 4. Fold-Safe Aggregations (outer-fit ONLY)
    print("\nComputing outer-fit fold-safe aggregations...", flush=True)
    df_feat = df_num.copy()

    # Fit-only frequencies
    for col in ["seller_id", "category_id", "city"]:
        counts = cat_df.iloc[fi][col].value_counts()
        df_feat["fit_freq_" + col] = cat_df[col].map(counts).fillna(0.0).astype(np.float32)

    # Fit-only category median log-price & relative price
    cat_med = df_num.iloc[fi].groupby(cat_df.iloc[fi]["category_id"])["log_price"].median()
    global_med = float(df_num.iloc[fi]["log_price"].median())
    df_feat["fit_cat_med_lp"] = cat_df["category_id"].map(cat_med).fillna(global_med).astype(np.float32)
    df_feat["fit_rel_price_cat"] = (df_feat["log_price"] - df_feat["fit_cat_med_lp"]).astype(np.float32)

    # Fit-only seller stats
    g_seller = df_num.iloc[fi].groupby(cat_df.iloc[fi]["seller_id"])
    seller_mean_lp = g_seller["log_price"].mean()
    seller_mean_iq = g_seller["sold_ratio"].mean()
    seller_ncats = cat_df.iloc[fi].groupby(cat_df.iloc[fi]["seller_id"])["category_id"].nunique()

    df_feat["fit_seller_mean_logprice"] = cat_df["seller_id"].map(seller_mean_lp).fillna(global_med).astype(np.float32)
    df_feat["fit_seller_mean_sold_ratio"] = cat_df["seller_id"].map(seller_mean_iq).fillna(float(df_num.iloc[fi]["sold_ratio"].mean())).astype(np.float32)
    df_feat["fit_seller_n_cats"] = cat_df["seller_id"].map(seller_ncats).fillna(1.0).astype(np.float32)

    # Add categorical features
    cat_cols = ["category_id", "listing_type_id", "buying_mode", "status", "ship_mode", "state", "city", "sub_status", "ship_tags"]
    for c in cat_cols:
        df_feat[c] = cat_df[c].astype("category")

    # 5. Nested Seller-Group Text Feature Generation (p_text)
    print("\nComputing nested seller-group text features (GroupKFold=5)...", flush=True)
    word = HashingVectorizer(n_features=2**16, alternate_sign=False, ngram_range=(1, 2), norm="l2", dtype=np.float32)
    char = HashingVectorizer(n_features=2**16, alternate_sign=False, analyzer="char_wb", ngram_range=(3, 5), norm="l2", dtype=np.float32)
    X_text = sparse.hstack([word.transform(text_arr), char.transform(text_arr)], format="csr")

    p_nested_text = np.full(N, np.nan, dtype=np.float32)
    p_nested_te = np.full(N, np.nan, dtype=np.float32)

    gkf = GroupKFold(n_splits=5)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(fi, y[fi], sellers[fi])):
        # Audit no seller overlap in nested fold
        assert not set(sellers[fi[tr_idx]]) & set(sellers[fi[va_idx]]), f"Seller leakage in nested fold {fold}!"
        # Fit text model on tr_idx, predict on va_idx
        p_nested_text[fi[va_idx]] = text_fit_nb(X_text, y, fi[tr_idx], fi[va_idx])

        # Nested smoothed target encoding for category_id
        cat_tr = cat_df.iloc[fi[tr_idx]]["category_id"]
        y_tr = y[fi[tr_idx]]
        global_p = float(y_tr.mean())
        m_smooth = 10.0
        counts = cat_tr.value_counts()
        sums = pd.Series(y_tr).groupby(cat_tr.values).sum()
        te_map = (sums + m_smooth * global_p) / (counts + m_smooth)
        p_nested_te[fi[va_idx]] = cat_df.iloc[fi[va_idx]]["category_id"].map(te_map).fillna(global_p).values
        print(f"  Nested fold {fold+1}/5 complete ({time.time()-t_start:.1f}s)", flush=True)

    # Outer fit: train text model and target encoding on all fi, predict on tu and co
    print("Projecting text model and TE onto tune and confirm from all outer-fit...", flush=True)
    p_nested_text[np.r_[tu, co]] = text_fit_nb(X_text, y, fi, np.r_[tu, co])
    cat_fi = cat_df.iloc[fi]["category_id"]
    y_fi = y[fi]
    global_p_fi = float(y_fi.mean())
    counts_fi = cat_fi.value_counts()
    sums_fi = pd.Series(y_fi).groupby(cat_fi.values).sum()
    te_map_fi = (sums_fi + 10.0 * global_p_fi) / (counts_fi + 10.0)
    p_nested_te[np.r_[tu, co]] = cat_df.iloc[np.r_[tu, co]]["category_id"].map(te_map_fi).fillna(global_p_fi).values

    assert not np.isnan(p_nested_text).any(), "NaN in p_nested_text!"
    assert not np.isnan(p_nested_te).any(), "NaN in p_nested_te!"
    del X_text
    gc.collect()

    # 6. Build model datasets
    # Dataset A: purely tabular fold-safe features
    df_tab = df_feat.copy()
    # Dataset B: tabular + nested text meta-probability + nested category TE
    df_text = df_feat.assign(nested_text_p=p_nested_text, nested_cat_te=p_nested_te)

    probs = {}
    tup = {}
    meta = {}

    models_config = [
        ("f117_lgb_tabular", df_tab, {
            "n_estimators": 900, "learning_rate": 0.04, "num_leaves": 45, "max_depth": 7,
            "min_child_samples": 30, "subsample": 0.85, "subsample_freq": 1, "colsample_bytree": 0.85,
            "reg_lambda": 5.0, "random_state": args.seed, "n_jobs": 6, "verbose": -1
        }),
        ("f117_lgb_nested_text", df_text, {
            "n_estimators": 900, "learning_rate": 0.04, "num_leaves": 45, "max_depth": 7,
            "min_child_samples": 30, "subsample": 0.85, "subsample_freq": 1, "colsample_bytree": 0.85,
            "reg_lambda": 5.0, "random_state": args.seed, "n_jobs": 6, "verbose": -1
        })
    ]

    for name, df_model, params in models_config:
        print(f"\nTraining {name} (features={df_model.shape[1]})...", flush=True)
        t0 = time.time()
        clf = lgb.LGBMClassifier(**params)
        clf.fit(
            df_model.iloc[fi], y[fi],
            eval_set=[(df_model.iloc[tu], y[tu])],
            callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=False)]
        )
        tup[name] = clf.predict_proba(df_model.iloc[tu])[:, 1]
        probs[name] = clf.predict_proba(df_model.iloc[co])[:, 1]
        best_it = int(clf.best_iteration_) if hasattr(clf, "best_iteration_") else params["n_estimators"]
        dur = time.time() - t0
        meta[name] = {"best_iteration": best_it, "duration_seconds": dur, "n_features": df_model.shape[1]}
        print(f"  {name} trained: best_it={best_it}, duration={dur:.1f}s", flush=True)

    # 7. Check if baseline pairs exist for direct paired evaluation
    baseline_pairs_path = Path(args.baseline_pairs)
    cb_nested_nb_co = None
    cb_nested_nb_tu = None
    if baseline_pairs_path.exists():
        print(f"\nLoading baseline pairs from {baseline_pairs_path}...", flush=True)
        df_base_co = pd.read_csv(baseline_pairs_path)
        if "p_cb_nested_nb" in df_base_co.columns:
            probs["cb_nested_nb"] = df_base_co["p_cb_nested_nb"].values
            cb_nested_nb_co = probs["cb_nested_nb"]
        if "p_cb_tabular" in df_base_co.columns:
            probs["cb_tabular"] = df_base_co["p_cb_tabular"].values

        tune_pairs_path = baseline_pairs_path.parent / "tune_pairs.csv"
        if tune_pairs_path.exists():
            df_base_tu = pd.read_csv(tune_pairs_path)
            if "p_cb_nested_nb" in df_base_tu.columns:
                tup["cb_nested_nb"] = df_base_tu["p_cb_nested_nb"].values
                cb_nested_nb_tu = tup["cb_nested_nb"]
            if "p_cb_tabular" in df_base_tu.columns:
                tup["cb_tabular"] = df_base_tu["p_cb_tabular"].values

    # 8. Blend Exploration (strictly tuned on tune ONLY)
    if cb_nested_nb_tu is not None:
        print("\nEvaluating potential blend with cb_nested_nb on tune...", flush=True)
        # Search best weight w in [0.0, 1.0] solely on tune
        best_w = 0.0
        best_acc_tu = 0.0
        for w in np.arange(0.0, 1.01, 0.05):
            p_blend_tu = w * tup["f117_lgb_nested_text"] + (1.0 - w) * cb_nested_nb_tu
            acc = accuracy_score(y[tu], p_blend_tu >= 0.5)
            if acc > best_acc_tu:
                best_acc_tu = acc
                best_w = w
        print(f"Tune-locked optimal weight for f117_lgb_nested_text: w={best_w:.2f} (tune acc: {best_acc_tu:.5f})", flush=True)

        # Freeze blend recipe
        tup["blend_cb_f117_tune_locked"] = best_w * tup["f117_lgb_nested_text"] + (1.0 - best_w) * cb_nested_nb_tu
        probs["blend_cb_f117_tune_locked"] = best_w * probs["f117_lgb_nested_text"] + (1.0 - best_w) * cb_nested_nb_co
        # Also predeclare 50/50 blend
        tup["blend_cb_f117_50_50"] = 0.5 * tup["f117_lgb_nested_text"] + 0.5 * cb_nested_nb_tu
        probs["blend_cb_f117_50_50"] = 0.5 * probs["f117_lgb_nested_text"] + 0.5 * cb_nested_nb_co

    # 9. Freeze thresholds on tune ONLY
    print("\nFreezing decision thresholds on tune...", flush=True)
    thresholds = {}
    for name, p_tu in tup.items():
        best_th = float(max(np.arange(0.40, 0.601, 0.01), key=lambda th: accuracy_score(y[tu], p_tu >= th)))
        thresholds[name] = best_th

    # 10. Evaluate on confirm
    seen = np.isin(sellers[co], sellers[fi])
    base_pred = (probs.get("cb_tabular", probs["f117_lgb_tabular"])) >= 0.5
    results_050 = {name: compute_metrics(y[co], p, base_pred, sellers[co], seen) for name, p in probs.items()}

    tuned_results = {}
    base_tuned = (probs.get("cb_tabular", probs["f117_lgb_tabular"])) >= thresholds.get("cb_tabular", thresholds["f117_lgb_tabular"])
    for name, p in probs.items():
        th = thresholds[name]
        pred_th = p >= th
        delta = (pred_th == y[co]).astype(int) - (base_tuned == y[co]).astype(int)
        tuned_results[name] = {
            "threshold": th,
            "tune_accuracy": float(accuracy_score(y[tu], tup[name] >= th)),
            "hits": int((pred_th == y[co]).sum()),
            "net_vs_tuned_base": int(delta.sum()),
            "confirm_accuracy": float(accuracy_score(y[co], pred_th))
        }

    # 11. Direct Paired Comparison vs cb_nested_nb (baseline fuerte)
    paired_vs_baseline = {}
    if cb_nested_nb_co is not None:
        base_nb = cb_nested_nb_co >= 0.5
        for name in ["f117_lgb_tabular", "f117_lgb_nested_text", "blend_cb_f117_tune_locked", "blend_cb_f117_50_50"]:
            if name in probs:
                paired_vs_baseline[name] = compute_metrics(y[co], probs[name], base_nb, sellers[co], seen)

    # 12. Save pairs
    pairs_df = pd.DataFrame({
        "item_id": ids[co],
        "seller_id": sellers[co],
        "is_seen": seen.astype(int),
        "y_true": y[co],
        **{"p_" + n: p for n, p in probs.items()}
    })
    pairs_path = out_dir / "pairs.csv"
    pairs_df.to_csv(pairs_path, index=False)
    pairs_sha256 = hashlib.sha256(pairs_path.read_bytes()).hexdigest()

    tune_pairs_df = pd.DataFrame({
        "item_id": ids[tu],
        "seller_id": sellers[tu],
        "y_true": y[tu],
        **{"p_" + n: p for n, p in tup.items()}
    })
    tune_pairs_path = out_dir / "tune_pairs.csv"
    tune_pairs_df.to_csv(tune_pairs_path, index=False)
    tune_pairs_sha256 = hashlib.sha256(tune_pairs_path.read_bytes()).hexdigest()

    # Feature manifest
    manifest = {
        "n_intrinsic_features": df_num.shape[1],
        "n_fold_safe_features": df_feat.shape[1],
        "dropped_leaky_features": ["freq_seller_id (global)", "freq_category_id (global)", "freq_city (global)", "rel_price_cat (global)", "seller_stats (global)"],
        "added_fold_safe_features": ["fit_freq_seller_id", "fit_freq_category_id", "fit_freq_city", "fit_cat_med_lp", "fit_rel_price_cat", "fit_seller_mean_logprice", "fit_seller_mean_sold_ratio", "fit_seller_n_cats"],
        "nested_supervised_features": ["nested_text_p (GroupKFold=5 on outer-fit)", "nested_cat_te (GroupKFold=5 on outer-fit)"]
    }

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "script": "scripts/train_fold_safe_f117_lgbm_20261106.py",
        "seed": args.seed,
        "pure": args.pure,
        "split_sizes": {"fit": len(fi), "tune": len(tu), "confirm": len(co)},
        "split_hashes": split_hashes,
        "manifest": manifest,
        "model_metadata": meta,
        "results_at_050": results_050,
        "tune_selected": tuned_results,
        "paired_vs_cb_nested_nb": paired_vs_baseline,
        "pairs_sha256": pairs_sha256,
        "tune_pairs_sha256": tune_pairs_sha256,
        "duration_seconds": time.time() - t_start,
        "compliance": "Strictly inductive train-only. No pseudo-labels, no test leakage, no random CV. Thresholds tuned only on tune."
    }

    report_path = out_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {report_path}", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
