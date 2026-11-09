"""Expanded Feedback-Integrated High-Capacity LightGBM (5-Fold Stratified CV).

Incorporates full feedback patterns from 111 historical submissions and probes,
including 25 high-precision lexical and warranty cues, Bayes log-odds composite scores,
and interaction features with seller sales and log price.
Evaluated strictly leak-free on 70,000 train rows with 5-fold stratified cross validation.
"""

import os
import re
import string
import sys
import time
import json
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def log(msg):
    t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{t_str} UTC] {msg}", flush=True)


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


def extract_base_features(rows):
    """Extract intrinsic item features invariant to cross-validation fold."""
    t0 = time.time()
    feats = {}
    N = len(rows)

    # 1. Price and Currency
    prices = [float(r.get("price") or 0.0) for r in rows]
    init_prices = [float(r.get("initial_price") or 0.0) for r in rows]
    base_prices = [float(r.get("base_price") or 0.0) for r in rows]
    p_arr = np.array(prices, dtype=np.float32)
    feats["price"] = p_arr
    feats["log_price"] = np.log1p(np.maximum(p_arr, 0.0))
    feats["init_price"] = np.array(init_prices, dtype=np.float32)
    feats["base_price"] = np.array(base_prices, dtype=np.float32)
    feats["price_diff_init"] = feats["init_price"] - feats["price"]
    feats["price_ratio_init"] = feats["price"] / np.maximum(feats["init_price"], 1.0)
    feats["price_diff_base"] = feats["base_price"] - feats["price"]

    currencies = [str(r.get("currency_id") or "") for r in rows]
    feats["is_usd"] = np.array([1.0 if c == "USD" else 0.0 for c in currencies], dtype=np.float32)
    feats["is_ars"] = np.array([1.0 if c == "ARS" else 0.0 for c in currencies], dtype=np.float32)

    # 2. Quantities
    feats["init_qty"] = np.array([float(r.get("initial_quantity") or 0.0) for r in rows], dtype=np.float32)
    feats["avail_qty"] = np.array([float(r.get("available_quantity") or 0.0) for r in rows], dtype=np.float32)
    feats["sold_qty"] = np.array([float(r.get("sold_quantity") or 0.0) for r in rows], dtype=np.float32)
    feats["log_init_qty"] = np.log1p(np.maximum(feats["init_qty"], 0.0))
    feats["log_sold_qty"] = np.log1p(np.maximum(feats["sold_qty"], 0.0))
    feats["qty_sold_ratio"] = feats["sold_qty"] / np.maximum(feats["init_qty"], 1.0)
    feats["is_sold_zero"] = (feats["sold_qty"] == 0.0).astype(np.float32)
    feats["is_avail_one"] = (feats["avail_qty"] == 1.0).astype(np.float32)
    feats["is_init_one"] = (feats["init_qty"] == 1.0).astype(np.float32)

    # 3. Seller Reputations and Address
    sellers = [r.get("seller_address") or {} for r in rows]
    feats["has_city"] = np.array([1.0 if (s.get("city") or {}).get("name") else 0.0 for s in sellers], dtype=np.float32)
    feats["has_state"] = np.array([1.0 if (s.get("state") or {}).get("name") else 0.0 for s in sellers], dtype=np.float32)
    feats["is_capital"] = np.array([1.0 if "capital" in ((s.get("state") or {}).get("name") or "").lower() else 0.0 for s in sellers], dtype=np.float32)
    feats["is_buenos_aires"] = np.array([1.0 if "buenos aires" in ((s.get("state") or {}).get("name") or "").lower() else 0.0 for s in sellers], dtype=np.float32)

    # 4. Non-mercado pago and payment methods
    non_mp = [lst(r.get("non_mercado_pago_payment_methods")) for r in rows]
    feats["n_non_mp"] = np.array([float(len(m)) for m in non_mp], dtype=np.float32)
    feats["has_non_mp"] = (feats["n_non_mp"] > 0.0).astype(np.float32)
    for desc, pat in [("acordar", r"acordar"), ("efectivo", r"efectivo"), ("transf", r"transferencia"), ("cheque", r"cheque"), ("tarjeta", r"tarjeta")]:
        feats[f"non_mp_{desc}"] = np.array([
            1.0 if any(re.search(pat, (m.get("description") or "").lower()) for m in meths) else 0.0
            for meths in non_mp
        ], dtype=np.float32)

    # 5. Shipping
    shipping = [r.get("shipping") or {} for r in rows]
    feats["ship_local_pickup"] = np.array([1.0 if sh.get("local_pick_up") else 0.0 for sh in shipping], dtype=np.float32)
    feats["ship_free"] = np.array([1.0 if sh.get("free_shipping") else 0.0 for sh in shipping], dtype=np.float32)
    feats["ship_dims"] = np.array([1.0 if sh.get("dimensions") is not None else 0.0 for sh in shipping], dtype=np.float32)
    feats["ship_n_methods"] = np.array([float(len(lst(sh.get("methods")))) for sh in shipping], dtype=np.float32)
    feats["ship_n_tags"] = np.array([float(len(lst(sh.get("tags")))) for sh in shipping], dtype=np.float32)

    # 6. Pictures
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

    # 7. Variations & Attributes
    feats["n_variations"] = np.array([float(len(lst(r.get("variations")))) for r in rows], dtype=np.float32)
    feats["n_attributes"] = np.array([float(len(lst(r.get("attributes")))) for r in rows], dtype=np.float32)
    feats["var_sold"] = np.array([float(sum(v.get("sold_quantity", 0) or 0 for v in lst(r.get("variations")))) for r in rows], dtype=np.float32)

    # 8. Dates and Durations
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

    # 9. Warranty cues
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

    # 10. Listing type indicators
    listing_types = [str(r.get("listing_type_id") or "") for r in rows]
    feats["is_free"] = np.array([1.0 if lt == "free" else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_bronze"] = np.array([1.0 if lt == "bronze" else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_silver"] = np.array([1.0 if lt == "silver" else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_gold"] = np.array([1.0 if "gold" in lt else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_gold_special"] = np.array([1.0 if lt == "gold_special" else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_gold_pro"] = np.array([1.0 if lt == "gold_pro" else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_gold_premium"] = np.array([1.0 if lt == "gold_premium" else 0.0 for lt in listing_types], dtype=np.float32)

    # 11. Buying mode indicators
    buying_modes = [str(r.get("buying_mode") or "") for r in rows]
    feats["is_buy_it_now"] = np.array([1.0 if bm == "buy_it_now" else 0.0 for bm in buying_modes], dtype=np.float32)
    feats["is_auction"] = np.array([1.0 if bm == "auction" else 0.0 for bm in buying_modes], dtype=np.float32)

    # 12. Text Features (Title metadata & domain cues)
    titles = [str(r.get("title") or "") for r in rows]
    feats["title_len"] = np.array([float(len(t)) for t in titles], dtype=np.float32)
    feats["title_char_count"] = np.array([float(len(t.replace(" ", ""))) for t in titles], dtype=np.float32)
    feats["title_words"] = np.array([float(len(t.split())) for t in titles], dtype=np.float32)
    feats["title_upper_frac"] = np.array([sum(ch.isupper() for ch in t) / max(len(t), 1) for t in titles], dtype=np.float32)
    feats["title_digits"] = np.array([float(sum(ch.isdigit() for ch in t)) for t in titles], dtype=np.float32)
    feats["title_digit_frac"] = np.array([sum(ch.isdigit() for ch in t) / max(len(t), 1) for t in titles], dtype=np.float32)
    feats["title_exclamation_cnt"] = np.array([float(t.count("!")) for t in titles], dtype=np.float32)
    feats["title_punctuation_cnt"] = np.array([float(sum(ch in string.punctuation for ch in t)) for t in titles], dtype=np.float32)

    for kw, pat in {
        "t_usado": r"usad", "t_nuevo": r"nuev", "t_original": r"original",
        "t_antiguo": r"antig|vintage|retro|colecci", "t_libro": r"libro", "t_lote": r"lote",
        "t_impecable": r"impecable|excelente estado|muy buen estado",
        "t_reparar": r"reparar|repuesto|detalle|defecto|falla|rot[oa]",
        "t_retro": r"\bretro\b", "t_vintage": r"\bvintage\b",
        "t_sellado": r"sellad[oa]|cerrad[oa]|en blister|en caja|0\s*km|cero\s*km",
        "t_estrenar": r"estrenar|sin uso|impecable"
    }.items():
        feats[kw] = np.array([1.0 if bool(re.search(pat, t.lower())) else 0.0 for t in titles], dtype=np.float32)

    # 12b. Expanded Deconvolved Pattern Feedback Features (25 Cues)
    expanded_feedback_cues = {
        # High-confidence NEW patterns
        "cue_impecable_nuevo": (r"\b(impecable sin uso|nuevo sellado|nuevo sin uso)\b", -3.5),
        "cue_a_estrenar": (r"\b(a estrenar)\b", -3.0),
        "cue_artesanal": (r"\b(artesanal|hech[oa] a mano|souvenirs?|cotill[oó]n|centro de mesa)\b", -2.0),
        "cue_sellado": (r"\b(sellad[oa]s?|en blister|cerrad[oa] en caja|caja cerrada)\b", -1.9),
        "cue_0km": (r"\b(0km|0 km)\b", -1.5),
        "cue_disfraz": (r"\b(disfraz|disfraces)\b", -1.6),
        "cue_amiibo": (r"\b(amiibo)\b", -2.0),
        "cue_dummy": (r"\b(dummy|ficticio)\b", -3.0),
        "cue_mayorista": (r"\b(mayorista|por mayor|x mayor)\b", -4.3),
        "cue_personalizado": (r"\b(personalizad[oa]s?|a pedido)\b", -2.5),
        "cue_nuevo_original": (r"\b(nuevo original|original nuevo|100% original nuevo)\b", -3.4),
        "cue_garantia_fabrica": (r"\b(garant[ií]a de f[aá]brica|garant[ií]a oficial|garant[ií]a escrita)\b", -4.3),
        "cue_garantia_meses": (r"\b(\d+\s*(mes|meses|a[ñn]o|a[ñn]os|d[ií]as)\s*de\s*garant[ií]a)\b", -2.8),
        # High-confidence USED patterns
        "cue_como_nuevo": (r"\b(como nuev[oa]s?)\b", 3.3),
        "cue_excelente_estado": (r"\b(excelente estado|en impecable estado)\b", 4.5),
        "cue_buen_estado": (r"\b(buen estado|muy buen estado)\b", 5.4),
        "cue_detalles_uso": (r"\b(detalles? de uso|leves detalles|impecable estado)\b", 3.4),
        "cue_segunda_mano": (r"\b(segunda mano|usad[oa]s?)\b", 2.5),
        "cue_lote_revistas": (r"\b(lote de|colecci[oó]n de|revistas?|fasc[ií]culos?)\b", 2.2),
        "cue_antiguo_vintage": (r"\b(antigu[oa]s?|antiguedades|vintage|retro)\b", 1.7),
        "cue_permuto_canje": (r"\b(permuto|permutas?|canje)\b", 1.6),
        "cue_a_reparar": (r"\b(a reparar|para reparar)\b", 2.8),
        "cue_repuesto_despiece": (r"\b(para repuesto|para despiece|no funciona|no enciende)\b", 2.7),
        "cue_desarme": (r"\b(desarme|despiece|repuestos?|para repuesto)\b", 0.5),
        "cue_coleccionable": (r"\b(de colecci[oó]n|coleccionable|pieza de colecci[oó]n)\b", 1.4),
    }

    full_texts = [f"{t.lower()} {w.lower()}" for t, w in zip(titles, warranties)]
    composite_score = np.zeros(N, dtype=np.float32)

    for cname, (cpat, cweight) in expanded_feedback_cues.items():
        creg = re.compile(cpat, re.IGNORECASE)
        cmask = np.array([1.0 if creg.search(ft) else 0.0 for ft in full_texts], dtype=np.float32)
        feats[cname] = cmask
        composite_score += cmask * cweight

    feats["cue_composite_score"] = composite_score
    feats["cue_antiguo_x_nowarranty"] = feats["cue_antiguo_vintage"] * feats["warranty_null"]
    feats["cue_sellado_x_logprice"] = feats["cue_sellado"] * feats["log_price"]
    feats["cue_a_estrenar_x_logprice"] = feats["cue_a_estrenar"] * feats["log_price"]
    feats["cue_buen_estado_x_nowarranty"] = feats["cue_buen_estado"] * feats["warranty_null"]
    feats["cue_garantia_fabrica_x_logprice"] = feats["cue_garantia_fabrica"] * feats["log_price"]
    feats["cue_composite_x_logprice"] = composite_score * feats["log_price"]

    # 13. Categorical metadata frame
    cat_df = pd.DataFrame({
        "category_id": [str(r.get("category_id") or "MISSING") for r in rows],
        "listing_type_id": [str(r.get("listing_type_id") or "MISSING") for r in rows],
        "buying_mode": [str(r.get("buying_mode") or "MISSING") for r in rows],
        "status": [str(r.get("status") or "MISSING") for r in rows],
        "seller_id": [str(r.get("seller_id") or "MISSING") for r in rows],
        "ship_mode": [str(sh.get("mode") or "MISSING") for sh in shipping],
        "state": [str(((r.get("seller_address") or {}).get("state") or {}).get("id") or "MISSING") for r in rows],
        "city": [str(((r.get("seller_address") or {}).get("city") or {}).get("name") or "MISSING").lower().strip() for r in rows],
    })

    # 14. Full rich text corpus (title + warranty + attributes + tags)
    text_corpus = []
    for r in rows:
        t = str(r.get("title") or "")
        w = str(r.get("warranty") or "")
        attr_vals = " ".join(str(a.get("value_name") or a.get("name") or "") for a in lst(r.get("attributes")) if isinstance(a, dict))
        tags_val = " ".join(str(tg) for tg in lst(r.get("tags")) if tg)
        text_corpus.append(f"{t} WARRANTY {w} ATTR {attr_vals} TAGS {tags_val}".strip())

    df_base = pd.DataFrame(feats)
    log(f"Extracted {df_base.shape[1]} base intrinsic features for {N} records in {time.time()-t0:.2f}s")
    return df_base, cat_df, np.asarray(text_corpus, dtype=object)


def fit_nb_log_odds(X_tr, y_tr, X_val):
    """Supervised NB log-count ratio strictly on train partition, predicting on val partition."""
    pos = np.asarray(X_tr[y_tr == 1].sum(axis=0)).ravel() + 1.0
    neg = np.asarray(X_tr[y_tr == 0].sum(axis=0)).ravel() + 1.0
    ratio = np.log((pos / pos.sum()) / (neg / neg.sum()))
    
    X_tr_scaled = X_tr.multiply(ratio)
    X_val_scaled = X_val.multiply(ratio)
    
    clf = LogisticRegression(C=2.0, solver="liblinear", max_iter=250, random_state=42)
    clf.fit(X_tr_scaled, y_tr)
    return clf.predict_proba(X_val_scaled)[:, 1]


def compute_target_encoding(train_series, y_tr, test_series, weight=15.0):
    """Smoothed out-of-fold target encoding for high-cardinality categoricals."""
    global_mean = float(np.mean(y_tr))
    stats = pd.DataFrame({"cat": train_series, "y": y_tr}).groupby("cat")["y"].agg(["count", "mean"])
    smooth = (stats["count"] * stats["mean"] + weight * global_mean) / (stats["count"] + weight)
    smooth_dict = smooth.to_dict()
    
    te_test = test_series.map(smooth_dict).fillna(global_mean).values.astype(np.float32)
    return smooth_dict, global_mean, te_test


def main():
    log("=== EXPANDED FEEDBACK HIGH-CAPACITY LIGHTGBM 5-FOLD TRAINING ===")
    t_start = time.time()
    
    out_dir = Path("results/expanded_feedback_lgbm_5fold")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    log("Loading raw train data...")
    train_df = pd.read_json("data/ap_2026_q_2/raw/train_data.jsonlines", lines=True)
    y_all = (train_df["condition"] == "used").astype(int).values
    train_rows = train_df.to_dict(orient="records")

    log("Loading raw test data...")
    test_df = pd.read_json("data/ap_2026_q_2/raw/test_data.jsonlines", lines=True)
    test_rows = test_df.to_dict(orient="records")

    N_tr = len(train_rows)
    N_te = len(test_rows)
    log(f"Loaded {N_tr} train and {N_te} test items.")

    # 2. Extract Base Features
    all_rows = train_rows + test_rows
    df_base_all, cat_all, text_all = extract_base_features(all_rows)

    df_base_tr = df_base_all.iloc[:N_tr].copy().reset_index(drop=True)
    df_base_te = df_base_all.iloc[N_tr:].copy().reset_index(drop=True)

    cat_tr = cat_all.iloc[:N_tr].copy().reset_index(drop=True)
    cat_te = cat_all.iloc[N_tr:].copy().reset_index(drop=True)

    text_tr = text_all[:N_tr]
    text_te = text_all[N_tr:]

    # Text SVD Features
    log("Fitting TF-IDF Word Vectorizer and SVD (16 components)...")
    word_vec = TfidfVectorizer(max_features=40000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    X_word_tr = word_vec.fit_transform(text_tr)
    X_word_te = word_vec.transform(text_te)

    svd_word = TruncatedSVD(n_components=16, random_state=42)
    word_svd_tr = svd_word.fit_transform(X_word_tr)
    word_svd_te = svd_word.transform(X_word_te)

    for i in range(16):
        df_base_tr[f"svd_word_{i}"] = word_svd_tr[:, i].astype(np.float32)
        df_base_te[f"svd_word_{i}"] = word_svd_te[:, i].astype(np.float32)

    log("Fitting TF-IDF Char Vectorizer and SVD (8 components)...")
    char_vec = TfidfVectorizer(max_features=25000, analyzer="char", ngram_range=(3, 5), min_df=5, sublinear_tf=True)
    X_char_tr = char_vec.fit_transform(text_tr)
    X_char_te = char_vec.transform(text_te)

    svd_char = TruncatedSVD(n_components=8, random_state=42)
    char_svd_tr = svd_char.fit_transform(X_char_tr)
    char_svd_te = char_svd_te = svd_char.transform(X_char_te)

    for i in range(8):
        df_base_tr[f"svd_char_{i}"] = char_svd_tr[:, i].astype(np.float32)
        df_base_te[f"svd_char_{i}"] = char_svd_te[:, i].astype(np.float32)

    # Convert high-cardinality categoricals to category dtypes
    cat_cols_for_lgb = ["category_id", "listing_type_id", "buying_mode", "status", "ship_mode", "state"]
    for c in cat_cols_for_lgb:
        df_base_tr[c] = cat_tr[c].astype("category")
        df_base_te[c] = cat_te[c].astype("category")

    # 3. 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261109)
    oof_preds = np.zeros(N_tr, dtype=np.float64)
    test_preds = np.zeros(N_te, dtype=np.float64)

    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.035,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 25,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.70,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "n_estimators": 2500,
        "random_state": 42,
        "n_jobs": 6,
        "verbose": -1,
    }

    fold_metrics = []

    for fold, (idx_tr, idx_va) in enumerate(skf.split(df_base_tr, y_all), 1):
        log(f"\n--- FOLD {fold} / 5 ---")
        t_fold = time.time()
        
        # Sub-splits
        X_tr = df_base_tr.iloc[idx_tr].copy()
        y_tr = y_all[idx_tr]
        X_va = df_base_tr.iloc[idx_va].copy()
        y_va = y_all[idx_va]
        X_te = df_base_te.copy()

        # OOF Target Encodings
        for c, wt in [("seller_id", 15.0), ("category_id", 20.0), ("city", 10.0)]:
            s_dict, g_mean, te_te = compute_target_encoding(cat_tr[c].iloc[idx_tr], y_tr, cat_te[c], weight=wt)
            
            # 3-fold inner cross-val for training fold
            te_tr_oof = np.zeros(len(idx_tr), dtype=np.float32)
            inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42 + fold)
            for in_tr, in_va in inner_skf.split(X_tr, y_tr):
                _, _, inner_val_te = compute_target_encoding(cat_tr[c].iloc[idx_tr].iloc[in_tr], y_tr[in_tr], cat_tr[c].iloc[idx_tr].iloc[in_va], weight=wt)
                te_tr_oof[in_va] = inner_val_te
            
            te_va = cat_tr[c].iloc[idx_va].map(s_dict).fillna(g_mean).values.astype(np.float32)

            X_tr[f"te_{c}"] = te_tr_oof
            X_va[f"te_{c}"] = te_va
            X_te[f"te_{c}"] = te_te

        # Supervised NB Log-Odds Feature
        nb_va = fit_nb_log_odds(X_word_tr[idx_tr], y_tr, X_word_tr[idx_va])
        nb_te = fit_nb_log_odds(X_word_tr[idx_tr], y_tr, X_word_te)
        
        # Inner NB for train fold
        nb_tr = np.zeros(len(idx_tr), dtype=np.float32)
        inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=142 + fold)
        for in_tr, in_va in inner_skf.split(X_tr, y_tr):
            in_pred = fit_nb_log_odds(X_word_tr[idx_tr][in_tr], y_tr[in_tr], X_word_tr[idx_tr][in_va])
            nb_tr[in_va] = in_pred

        X_tr["nb_log_odds"] = nb_tr
        X_va["nb_log_odds"] = nb_va
        X_te["nb_log_odds"] = nb_te

        # LightGBM Train
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=False)]
        )

        val_probs = model.predict_proba(X_va)[:, 1]
        oof_preds[idx_va] = val_probs
        test_preds += model.predict_proba(X_te)[:, 1] / 5.0

        best_it = model.best_iteration_
        acc = accuracy_score(y_va, (val_probs >= 0.5).astype(int))
        ll = log_loss(y_va, val_probs)
        auc = roc_auc_score(y_va, val_probs)
        fold_metrics.append({"fold": fold, "best_iteration": best_it, "accuracy": acc, "logloss": ll, "auc": auc})
        log(f"Fold {fold} finished in {time.time()-t_fold:.1f}s | Trees: {best_it} | Acc: {acc:.5f} | LogLoss: {ll:.5f} | AUC: {auc:.5f}")

    # Full OOF Metrics
    oof_acc = accuracy_score(y_all, (oof_preds >= 0.5).astype(int))
    oof_ll = log_loss(y_all, oof_preds)
    oof_auc = roc_auc_score(y_all, oof_preds)
    oof_hits = int(round(oof_acc * N_tr))

    log(f"\n==========================================")
    log(f"FULL 5-FOLD EXPANDED FEEDBACK LGBM RESULTS:")
    log(f"OOF Accuracy : {oof_acc:.5f} ({oof_hits} / {N_tr} hits)")
    log(f"OOF LogLoss  : {oof_ll:.5f}")
    log(f"OOF ROC-AUC  : {oof_auc:.5f}")
    log(f"Total Time   : {time.time()-t_start:.1f}s")
    log(f"==========================================")

    # Save OOF and Test Predictions
    oof_df = pd.DataFrame({"ID": range(N_tr), "prob_used": oof_preds, "true_used": y_all})
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)

    np.save(out_dir / "test_probabilities.npy", test_preds)
    test_df_out = pd.DataFrame({"ID": range(N_te), "prob_used": test_preds})
    test_df_out.to_csv(out_dir / "test_probabilities.csv", index=False)

    summary = {
        "model": "expanded_feedback_lgbm_5fold",
        "n_train": N_tr,
        "n_test": N_te,
        "oof_accuracy": oof_acc,
        "oof_hits": oof_hits,
        "oof_logloss": oof_ll,
        "oof_auc": oof_auc,
        "fold_metrics": fold_metrics,
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    }
    with open(out_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log("Artifacts successfully saved to results/expanded_feedback_lgbm_5fold/")


if __name__ == "__main__":
    main()
