"""T98 Line 4 — Unseen-Seller (New Seller) Intrinsic Specialist without Global Frequencies.

Mandatory Protocol & Constraints (CX-0164 + User Prompt):
1. Uses ONLY the frozen shared split and pre-parsed caches:
   - `data/splits/t98_parsed_train.pkl` (70,000 rows)
   - `data/splits/t98_parsed_test.pkl` (30,000 rows)
   - `data/splits/t98_dual_regime_outer_seller_group_splits.csv` (70,000 rows)
2. 100% Seller-Free & Zero Global Frequencies:
   - Never uses `seller_id` or `seller_te` or seller item counts as model features.
   - `seller_id` is used ONLY to enforce seller-disjoint cross-fitting (`StratifiedGroupKFold`) inside `fit_idx`.
   - Every category prior, price distribution statistic, brand/model/condition/warranty/shipping/payment/tag
     target & support encoding, TF-IDF vectorizer, SVD projection, and OOF text model is computed strictly
     inside the active training fold (`inner_train` or `fit_idx`).
   - All text representations (`p_text_oof`, `p_text_lr_c05_oof`, `z_text_ridge_oof`, `logodds_text_nb_oof`,
     `p_title_lr_oof`, `z_title_ridge_oof`, `p_struct_lr_oof`, `z_struct_ridge_oof`) are 4-fold seller-grouped
     out-of-fold inside the active training fold, eliminating the in-sample `p_text` train/val calibration shift
     diagnosed in CX-0164 Finding 2.
3. Resource & Governance Discipline:
   - `n_jobs=4` / `thread_count=4`, RAM < 3.5 GB.
   - `status = "DEVELOPMENT_ONLY"`, zero submissions or Kaggle calls, zero inferred test labels.
"""
import gc
import hashlib
import json
import os
import platform
import subprocess
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import catboost
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "splits"
OUT_DIR = ROOT / "results" / "t98_line4_new_seller_specialist_20261111"
CACHE_DIR = OUT_DIR / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

N_THREADS = int(os.environ.get("L4_THREADS", 4))
MAX_FOLDS = int(os.environ.get("L4_MAX_FOLDS", 5))

USED_KWS = [
    "antiguo", "antigua", "antiguos", "antiguas", "vintage", "retro",
    "impecable", "impecables", "poco uso", "como nuevo", "como nueva",
    "despiece", "usado", "usada", "usados", "usadas", "segunda mano",
    "excelente estado", "muy buen estado", "buen estado", "funcionando",
    "detalle", "detalles", "reparar", "repuesto", "coleccion", "lote",
]
NEW_KWS = [
    "nuevo", "nueva", "nuevos", "nuevas", "sellado", "sellada",
    "en caja", "0km", "cero km", "garantia oficial", "de fabrica",
    "original", "cerrado", "blister", "sin uso", "a estrenar",
    "factura a", "factura b", "envio gratis", "distribuidor", "oficial",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def logit(p: np.ndarray) -> np.ndarray:
    p_clip = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p_clip / (1.0 - p_clip))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))


def boot_two_way(diff: np.ndarray, sellers: np.ndarray, cats: np.ndarray, n_boot: int = 2000, seed: int = 20261111) -> np.ndarray:
    rng = np.random.default_rng(seed)
    su, si = np.unique(sellers, return_inverse=True)
    cu, ci = np.unique(cats, return_inverse=True)
    out = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        w = rng.poisson(1.0, len(su))[si] * rng.poisson(1.0, len(cu))[ci]
        out[b] = (w * diff).sum() / w.sum()
    return out


def precompute_row_intrinsic_table(df: pd.DataFrame) -> dict:
    """Extract 100% row-local (single-item) intrinsic features and text views.
    Zero cross-row statistics, zero seller identity, zero target usage.
    """
    n = len(df)
    title = df["title"].fillna("").astype(str)
    tl = title.str.lower()
    warr = df["warranty_str"].fillna("").astype(str)
    wl = warr.str.strip().str.lower()
    attr = df["attr_text"].fillna("").astype(str)
    tags = df["tags_str"].fillna("").astype(str)
    pay = df["pay_ids"].fillna("").astype(str)

    price = np.maximum(0.0, df["price"].fillna(0.0).values.astype(np.float64))
    base_price = np.maximum(0.0, df["base_price"].fillna(0.0).values.astype(np.float64))
    orig_price = np.maximum(0.0, df["orig_price"].fillna(0.0).values.astype(np.float64))
    has_orig_price = df["has_orig_price"].fillna(0).values.astype(np.float32)

    log_price = np.log1p(price)
    log_base_price = np.log1p(base_price)
    log_orig_price = np.log1p(orig_price)
    base_to_price = np.clip(base_price / (price + 1e-5), 0.0, 10.0)
    orig_to_price = np.clip(orig_price / (price + 1e-5), 0.0, 10.0)
    price_diff_base = (np.abs(price - base_price) > 1e-4).astype(np.float32)

    p_round = np.round(price)
    price_is_int = (np.abs(price - p_round) < 1e-4).astype(np.float32)
    price_mod5 = ((price_is_int == 1) & (p_round % 5 == 0)).astype(np.float32)
    price_mod10 = ((price_is_int == 1) & (p_round % 10 == 0)).astype(np.float32)
    price_mod50 = ((price_is_int == 1) & (p_round % 50 == 0)).astype(np.float32)
    price_mod100 = ((price_is_int == 1) & (p_round % 100 == 0)).astype(np.float32)
    price_end9 = ((price_is_int == 1) & (p_round % 10 == 9)).astype(np.float32)
    is_usd = (df["currency_id"].values == "USD").astype(np.float32)

    init_qty = np.maximum(0.0, df["init_qty"].fillna(0.0).values.astype(np.float64))
    avail_qty = np.maximum(0.0, df["avail_qty"].fillna(0.0).values.astype(np.float64))
    sold_qty = np.maximum(0.0, df["sold_qty"].fillna(0.0).values.astype(np.float64))

    log_init_qty = np.log1p(init_qty)
    log_avail_qty = np.log1p(avail_qty)
    log_sold_qty = np.log1p(sold_qty)
    sold_ratio = np.clip(sold_qty / (init_qty + 1.0), 0.0, 50.0)
    avail_ratio = np.clip(avail_qty / (init_qty + 1.0), 0.0, 10.0)
    sold_to_avail = np.clip(sold_qty / (avail_qty + 1.0), 0.0, 50.0)
    log_qty_diff = np.log1p(np.maximum(0.0, init_qty - avail_qty))
    is_single_qty = ((init_qty == 1) & (sold_qty == 0)).astype(np.float32)
    is_multi_init = (init_qty > 1).astype(np.float32)
    has_sold = (sold_qty > 0).astype(np.float32)
    is_stock_ge5 = (init_qty >= 5).astype(np.float32)
    is_stock_ge10 = (init_qty >= 10).astype(np.float32)

    lt = df["listing_type_id"].fillna("MISSING").astype(str).values
    is_lt_free = (lt == "free").astype(np.float32)
    is_lt_bronze = (lt == "bronze").astype(np.float32)
    is_lt_silver = (lt == "silver").astype(np.float32)
    is_lt_gold = np.isin(lt, ["gold", "gold_special", "gold_premium", "gold_pro"]).astype(np.float32)
    is_lt_gold_spec = np.isin(lt, ["gold_special", "gold_pro"]).astype(np.float32)
    lt_ord_map = {"free": 0.0, "bronze": 1.0, "silver": 2.0, "gold_premium": 3.0, "gold": 4.0, "gold_special": 5.0, "gold_pro": 6.0}
    lt_ord = pd.Series(lt).map(lt_ord_map).fillna(1.0).values.astype(np.float32)

    bm = df["buying_mode"].fillna("MISSING").astype(str).values
    is_bm_bin = (bm == "buy_it_now").astype(np.float32)
    is_bm_class = (bm == "classified").astype(np.float32)
    is_bm_auc = (bm == "auction").astype(np.float32)

    sm = df["ship_mode"].fillna("not_specified").astype(str).values
    is_sm_me2 = (sm == "me2").astype(np.float32)
    is_sm_not_spec = (sm == "not_specified").astype(np.float32)
    is_sm_custom = (sm == "custom").astype(np.float32)
    is_sm_me1 = (sm == "me1").astype(np.float32)

    st = df["status"].fillna("active").astype(str).values
    is_st_active = (st == "active").astype(np.float32)
    is_st_paused = (st == "paused").astype(np.float32)

    free_shipping = df["free_shipping"].fillna(0).values.astype(np.float32)
    local_pick_up = df["local_pick_up"].fillna(0).values.astype(np.float32)
    accepts_mp = df["accepts_mercadopago"].fillna(0).values.astype(np.float32)
    auto_relist = df["automatic_relist"].fillna(0).values.astype(np.float32)
    has_off_store = df["has_official_store"].fillna(0).values.astype(np.float32)
    has_video = df["has_video"].fillna(0).values.astype(np.float32)
    has_catalog = df["has_catalog"].fillna(0).values.astype(np.float32)
    has_parent = df["has_parent_item"].fillna(0).values.astype(np.float32)

    num_attrs = df["num_attrs"].fillna(0).values.astype(np.float32)
    num_pics = df["num_pics"].fillna(0).values.astype(np.float32)
    num_vars = df["num_vars"].fillna(0).values.astype(np.float32)
    num_deals = df["num_deals"].fillna(0).values.astype(np.float32)
    num_non_mp = df["num_non_mp"].fillna(0).values.astype(np.float32)
    num_tags = df["num_tags"].fillna(0).values.astype(np.float32)

    log_num_pics = np.log1p(num_pics)
    has_zero_pics = (num_pics == 0).astype(np.float32)
    has_one_pic = (num_pics == 1).astype(np.float32)
    has_many_pics = (num_pics >= 4).astype(np.float32)
    has_vars = (num_vars > 0).astype(np.float32)
    has_multi_vars = (num_vars >= 2).astype(np.float32)
    has_deals = (num_deals > 0).astype(np.float32)

    pay_efectivo = pay.str.contains("MLAMO", regex=False).values.astype(np.float32)
    pay_transf = pay.str.contains("MLATB", regex=False).values.astype(np.float32)
    pay_acordar = pay.str.contains("MLAOT", regex=False).values.astype(np.float32)
    pay_mp_card = pay.str.contains("MLAMP", regex=False).values.astype(np.float32)
    pay_wc = pay.str.contains("MLAWC", regex=False).values.astype(np.float32)

    tag_good_thumb = tags.str.contains("good_quality_thumbnail", regex=False).values.astype(np.float32)
    tag_dragged_bids = tags.str.contains("dragged_bids_and_visits", regex=False).values.astype(np.float32)
    tag_free_relist = tags.str.contains("free_relist", regex=False).values.astype(np.float32)
    tag_dragged_vis_only = (
        tags.str.contains("dragged_visits", regex=False) & ~tags.str.contains("dragged_bids", regex=False)
    ).values.astype(np.float32)

    start_time = df["start_time"].fillna(0).values.astype(np.int64)
    stop_time = df["stop_time"].fillna(0).values.astype(np.int64)
    dur_days = (stop_time - start_time) / (86400.0 * 1000.0)
    log_dur_days = np.log1p(np.maximum(0.0, dur_days))
    is_dur_60d = (np.abs(dur_days - 60.0) < 0.5).astype(np.float32)
    is_dur_long = (dur_days > 65.0).astype(np.float32)
    is_dur_short = (dur_days < 55.0).astype(np.float32)
    start_sec = start_time // 1000
    start_hour = ((start_sec // 3600) % 24).astype(np.float32)
    start_dow = ((start_sec // 86400 + 4) % 7).astype(np.float32)

    title_len = df["title_len"].fillna(0).values.astype(np.float32)
    title_words = df["title_words"].fillna(0).values.astype(np.float32)
    title_digits = df["title_digits"].fillna(0).values.astype(np.float32)
    title_uppers = df["title_uppers"].fillna(0).values.astype(np.float32)
    title_upper_ratio = title_uppers / (title_len + 1.0)
    title_digit_ratio = title_digits / (title_len + 1.0)
    title_avg_wlen = title_len / (title_words + 1.0)

    used_kw_cnt = tl.apply(lambda s: sum(kw in s for kw in USED_KWS)).values.astype(np.float32)
    new_kw_cnt = tl.apply(lambda s: sum(kw in s for kw in NEW_KWS)).values.astype(np.float32)
    kw_diff = used_kw_cnt - new_kw_cnt
    title_has_excl = title.str.contains("!", regex=False).values.astype(np.float32)
    title_has_dash = title.str.contains("-", regex=False).values.astype(np.float32)
    title_has_slash = title.str.contains("/", regex=False).values.astype(np.float32)

    warr_len = warr.str.len().values.astype(np.float32)
    has_warr_text = (warr_len > 0).astype(np.float32)
    warr_has_mes_ano = df["warr_has_mes_ano"].fillna(0).values.astype(np.float32)
    warr_sin_garantia = df["warr_sin_garantia"].fillna(0).values.astype(np.float32)
    warr_fabrica = df["warr_fabrica"].fillna(0).values.astype(np.float32)
    warr_just_si = wl.isin(["sí", "si", "si.", "sí."]).values.astype(np.float32)
    warr_used_phrases = wl.apply(
        lambda s: int(
            any(
                k in s
                for k in (
                    "reputac", "calificac", "confianza", "descripci",
                    "tal cual", "revisar", "probar", "conformidad", "mi garantia",
                )
            )
        )
    ).values.astype(np.float32)
    warr_new_phrases = wl.apply(
        lambda s: int(
            any(
                k in s
                for k in (
                    "sellad", "nuevo", "nueva", "original", "falla",
                    "defecto", "cambio", "directa", "12 meses", "6 meses",
                    "3 meses", "1 año", "un año", "2 años", "dos años",
                )
            )
        )
    ).values.astype(np.float32)

    brand_val = df["brand_val"].fillna("MISSING").astype(str).values
    model_val = df["model_val"].fillna("MISSING").astype(str).values
    cond_attr = df["item_cond_attr"].fillna("MISSING").astype(str).values
    has_brand = (brand_val != "MISSING").astype(np.float32)
    has_model = (model_val != "MISSING").astype(np.float32)
    has_cond_attr = (cond_attr != "MISSING").astype(np.float32)
    has_part_num = attr.str.contains("PART_NUMBER", regex=False).values.astype(np.float32)
    attr_len = attr.str.len().values.astype(np.float32)

    # Composite intrinsic commercial vs casual indices (100% row-local)
    commercial_score = (
        is_multi_init
        + has_sold
        + is_lt_gold
        + is_lt_silver * 0.5
        + free_shipping
        + auto_relist
        + has_off_store
        + warr_fabrica
        + warr_new_phrases
        + has_part_num
        + pay_mp_card
        + tag_good_thumb
        + (1.0 - price_is_int)
        - is_lt_free * 1.5
        - is_single_qty * 1.0
        - warr_sin_garantia * 0.5
        - warr_used_phrases * 1.0
        - tag_free_relist * 1.0
    ).astype(np.float32)

    row_matrix = np.column_stack([
        log_price, log_base_price, log_orig_price, base_to_price, orig_to_price,
        has_orig_price, price_diff_base, price_is_int, price_mod5, price_mod10,
        price_mod50, price_mod100, price_end9, is_usd,
        log_init_qty, log_avail_qty, log_sold_qty, sold_ratio, avail_ratio,
        sold_to_avail, log_qty_diff, is_single_qty, is_multi_init, has_sold,
        is_stock_ge5, is_stock_ge10,
        is_lt_free, is_lt_bronze, is_lt_silver, is_lt_gold, is_lt_gold_spec, lt_ord,
        is_bm_bin, is_bm_class, is_bm_auc,
        is_sm_me2, is_sm_not_spec, is_sm_custom, is_sm_me1,
        is_st_active, is_st_paused,
        free_shipping, local_pick_up, accepts_mp, auto_relist,
        has_off_store, has_video, has_catalog, has_parent,
        num_attrs, num_pics, num_vars, num_deals, num_non_mp, num_tags,
        log_num_pics, has_zero_pics, has_one_pic, has_many_pics,
        has_vars, has_multi_vars, has_deals,
        pay_efectivo, pay_transf, pay_acordar, pay_mp_card, pay_wc,
        tag_good_thumb, tag_dragged_bids, tag_free_relist, tag_dragged_vis_only,
        dur_days, log_dur_days, is_dur_60d, is_dur_long, is_dur_short,
        start_hour, start_dow,
        title_len, title_words, title_digits, title_uppers,
        title_upper_ratio, title_digit_ratio, title_avg_wlen,
        used_kw_cnt, new_kw_cnt, kw_diff,
        title_has_excl, title_has_dash, title_has_slash,
        warr_len, has_warr_text, warr_has_mes_ano, warr_sin_garantia,
        warr_fabrica, warr_just_si, warr_used_phrases, warr_new_phrases,
        has_brand, has_model, has_cond_attr, has_part_num, attr_len,
        commercial_score,
    ]).astype(np.float32)

    cat_id = df["category_id"].fillna("MISSING").astype(str).values
    state_id = df["state_id"].fillna("MISSING").astype(str).values
    city_id = df["city_id"].fillna("MISSING").astype(str).values
    sub_status = df["sub_status_str"].fillna("NONE").astype(str).values
    w_norm_arr = wl.values.astype(str)
    pay_arr = pay.values.astype(str)
    tags_arr = tags.values.astype(str)

    # Pre-build interaction key strings (row-local concatenation of intrinsic fields)
    cat_x_lt = np.array([f"{c}|{l}" for c, l in zip(cat_id, lt)], dtype=object)
    cat_x_sq = np.array([f"{c}|{int(sq)}" for c, sq in zip(cat_id, is_single_qty)], dtype=object)
    cat_x_warr = np.array([f"{c}|{int(wm)}|{int(ws)}" for c, wm, ws in zip(cat_id, warr_has_mes_ano, warr_sin_garantia)], dtype=object)
    cat_x_brand = np.array([f"{c}|{b}" for c, b in zip(cat_id, brand_val)], dtype=object)
    cat_x_ship = np.array([f"{c}|{s}|{int(fs)}" for c, s, fs in zip(cat_id, sm, free_shipping)], dtype=object)
    lt_x_sq = np.array([f"{l}|{int(sq)}|{int(hs)}" for l, sq, hs in zip(lt, is_single_qty, has_sold)], dtype=object)

    full_text = (
        title + " " + tags + " " + attr + " " + warr + " " + df["listing_type_id"].fillna("")
    ).tolist()
    char_text = (title + " " + warr).tolist()
    title_only = title.tolist()
    struct_text = (
        warr + " " + attr + " " + tags + " " + pay + " "
        + df["listing_type_id"].fillna("") + " " + df["buying_mode"].fillna("") + " " + df["ship_mode"].fillna("")
    ).tolist()

    return {
        "row_matrix": row_matrix,
        "price": price,
        "log_price": log_price,
        "is_single_qty": is_single_qty,
        "is_lt_free": is_lt_free,
        "commercial_score": commercial_score,
        "cat_id": cat_id,
        "brand_val": brand_val,
        "model_val": model_val,
        "cond_attr": cond_attr,
        "lt": lt,
        "state_id": state_id,
        "city_id": city_id,
        "sm": sm,
        "bm": bm,
        "sub_status": sub_status,
        "w_norm": w_norm_arr,
        "pay_arr": pay_arr,
        "tags_arr": tags_arr,
        "cat_x_lt": cat_x_lt,
        "cat_x_sq": cat_x_sq,
        "cat_x_warr": cat_x_warr,
        "cat_x_brand": cat_x_brand,
        "cat_x_ship": cat_x_ship,
        "lt_x_sq": lt_x_sq,
        "full_text": full_text,
        "char_text": char_text,
        "title_only": title_only,
        "struct_text": struct_text,
    }


def compute_te_and_support_fit(keys_tr: np.ndarray, y_tr: np.ndarray, m: float, prior: float) -> tuple[pd.Series, pd.Series]:
    s = pd.DataFrame({"k": keys_tr, "y": y_tr}).groupby("k")["y"].agg(["count", "sum"])
    te_map = (s["sum"] + m * prior) / (s["count"] + m)
    # Scale support count by 50,000 / len(keys_tr) so support is invariant to fit set size
    scale = 50000.0 / max(1, len(keys_tr))
    sup_map = np.log1p(s["count"] * scale)
    return te_map, sup_map


def compute_hierarchical_te_fit(
    keys_tr: np.ndarray, parent_keys_tr: np.ndarray, y_tr: np.ndarray, m_parent: float, m_child: float, prior: float
) -> pd.Series:
    p_df = pd.DataFrame({"p": parent_keys_tr, "y": y_tr}).groupby("p")["y"].agg(["count", "sum"])
    p_map = (p_df["sum"] + m_parent * prior) / (p_df["count"] + m_parent)
    c_df = pd.DataFrame({"c": keys_tr, "p": parent_keys_tr, "y": y_tr}).groupby("c").agg(
        count=("y", "count"), sum=("y", "sum"), p=("p", "first")
    )
    c_prior = c_df["p"].map(p_map).fillna(prior)
    c_map = (c_df["sum"] + m_child * c_prior) / (c_df["count"] + m_child)
    return c_map


def extract_fold_tab_and_priors(
    tr_tab: dict,
    y_all: np.ndarray,
    T_idx: np.ndarray,
    sub_folds_T: np.ndarray,
    P_blocks: list[tuple[dict, np.ndarray]],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Compute fold-local category price distributions and seller-grouped OOF target/support encodings
    strictly inside T_idx, and map onto each (tab_dict, idx) in P_blocks.
    """
    yT = y_all[T_idx]
    prior = float(yT.mean())
    nT = len(T_idx)
    unique_sub = np.unique(sub_folds_T)

    # 1. Category price statistics fit strictly on T_idx
    catT = tr_tab["cat_id"][T_idx]
    lpT = tr_tab["log_price"][T_idx]
    prT = tr_tab["price"][T_idx]

    df_cat_p = pd.DataFrame({"cat": catT, "lp": lpT, "pr": prT})
    grp_cat = df_cat_p.groupby("cat")
    cat_lp_med = grp_cat["lp"].median()
    cat_lp_mean = grp_cat["lp"].mean()
    cat_lp_std = grp_cat["lp"].std().fillna(0.5)
    cat_lp_q25 = grp_cat["lp"].quantile(0.25)
    cat_lp_q75 = grp_cat["lp"].quantile(0.75)
    cat_pr_med = grp_cat["pr"].median()

    global_lp_med = float(np.median(lpT))
    global_lp_mean = float(np.mean(lpT))
    global_lp_std = float(np.std(lpT))
    global_pr_med = float(np.median(prT))

    # Category x Listing-Type median log_price on T_idx
    cat_lt_T = tr_tab["cat_x_lt"][T_idx]
    cat_lt_lp_med = pd.DataFrame({"cl": cat_lt_T, "lp": lpT}).groupby("cl")["lp"].median()

    def build_price_rel_feats(tab_d: dict, idx: np.ndarray) -> np.ndarray:
        c = pd.Series(tab_d["cat_id"][idx])
        cl = pd.Series(tab_d["cat_x_lt"][idx])
        lp = tab_d["log_price"][idx]
        pr = tab_d["price"][idx]

        med_lp = c.map(cat_lp_med).fillna(global_lp_med).values
        mean_lp = c.map(cat_lp_mean).fillna(global_lp_mean).values
        std_lp = c.map(cat_lp_std).fillna(global_lp_std).values
        q25_raw = c.map(cat_lp_q25).values.astype(np.float64)
        q25_lp = np.where(np.isnan(q25_raw), med_lp - 0.4, q25_raw)
        q75_raw = c.map(cat_lp_q75).values.astype(np.float64)
        q75_lp = np.where(np.isnan(q75_raw), med_lp + 0.4, q75_raw)
        iqr_lp = np.maximum(0.15, q75_lp - q25_lp)
        med_pr = c.map(cat_pr_med).fillna(global_pr_med).values
        cl_raw = cl.map(cat_lt_lp_med).values.astype(np.float64)
        cl_med_lp = np.where(np.isnan(cl_raw), med_lp, cl_raw)

        lp_diff_med = lp - med_lp
        lp_diff_mean = lp - mean_lp
        lp_z = (lp - mean_lp) / (std_lp + 0.25)
        lp_iqr_z = (lp - med_lp) / (iqr_lp + 0.25)
        pr_ratio_med = np.clip(pr / (med_pr + 1.0), 0.0, 50.0)
        lp_diff_cl_med = lp - cl_med_lp
        below_cat_q25 = (lp < q25_lp).astype(np.float32)
        above_cat_q75 = (lp > q75_lp).astype(np.float32)

        return np.column_stack([
            med_lp, std_lp, iqr_lp, lp_diff_med, lp_diff_mean,
            lp_z, lp_iqr_z, pr_ratio_med, lp_diff_cl_med,
            below_cat_q25, above_cat_q75,
        ]).astype(np.float32)

    price_rel_T = build_price_rel_feats(tr_tab, T_idx)
    price_rel_P_list = [build_price_rel_feats(td, ix) for td, ix in P_blocks]

    # 2. Seller-grouped OOF Target & Support Encodings strictly inside T_idx
    te_specs = [
        ("cat_id", 3.0, True),
        ("cat_id", 10.0, False),
        ("cat_id", 30.0, False),
        ("brand_val", 10.0, True),
        ("model_val", 10.0, True),
        ("cond_attr", 10.0, False),
        ("lt", 20.0, False),
        ("state_id", 20.0, True),
        ("city_id", 15.0, True),
        ("sm", 20.0, False),
        ("bm", 20.0, False),
        ("sub_status", 15.0, False),
        ("w_norm", 10.0, True),
        ("pay_arr", 15.0, False),
        ("tags_arr", 15.0, False),
        ("lt_x_sq", 15.0, False),
    ]
    hier_specs = [
        ("cat_x_lt", "cat_id", 10.0, 8.0),
        ("cat_x_sq", "cat_id", 10.0, 8.0),
        ("cat_x_warr", "cat_id", 10.0, 8.0),
        ("cat_x_brand", "cat_id", 10.0, 8.0),
        ("cat_x_ship", "cat_id", 10.0, 8.0),
    ]

    te_cols_T = []
    te_cols_P = [[] for _ in P_blocks]

    for col_name, m_val, keep_sup in te_specs:
        vals_T = tr_tab[col_name][T_idx]
        oof_te = np.full(nT, prior, dtype=np.float32)
        oof_sup = np.zeros(nT, dtype=np.float32)
        for sf in unique_sub:
            va_m = sub_folds_T == sf
            tr_m = ~va_m
            pr_sub = float(yT[tr_m].mean())
            t_map, s_map = compute_te_and_support_fit(vals_T[tr_m], yT[tr_m], m_val, pr_sub)
            s_va = pd.Series(vals_T[va_m])
            oof_te[va_m] = s_va.map(t_map).fillna(pr_sub).values.astype(np.float32)
            if keep_sup:
                oof_sup[va_m] = s_va.map(s_map).fillna(0.0).values.astype(np.float32)
        te_cols_T.append(oof_te)
        if keep_sup:
            te_cols_T.append(oof_sup)

        full_t_map, full_s_map = compute_te_and_support_fit(vals_T, yT, m_val, prior)
        for p_i, (td, ix) in enumerate(P_blocks):
            s_p = pd.Series(td[col_name][ix])
            te_cols_P[p_i].append(s_p.map(full_t_map).fillna(prior).values.astype(np.float32))
            if keep_sup:
                te_cols_P[p_i].append(s_p.map(full_s_map).fillna(0.0).values.astype(np.float32))

    for child_col, parent_col, m_p, m_c in hier_specs:
        c_T = tr_tab[child_col][T_idx]
        p_T = tr_tab[parent_col][T_idx]
        oof_hte = np.full(nT, prior, dtype=np.float32)
        for sf in unique_sub:
            va_m = sub_folds_T == sf
            tr_m = ~va_m
            pr_sub = float(yT[tr_m].mean())
            h_map = compute_hierarchical_te_fit(c_T[tr_m], p_T[tr_m], yT[tr_m], m_p, m_c, pr_sub)
            p_map, _ = compute_te_and_support_fit(p_T[tr_m], yT[tr_m], m_p, pr_sub)
            s_c = pd.Series(c_T[va_m])
            s_p = pd.Series(p_T[va_m])
            fallback = s_p.map(p_map).fillna(pr_sub)
            oof_hte[va_m] = s_c.map(h_map).fillna(fallback).values.astype(np.float32)
        te_cols_T.append(oof_hte)

        full_h_map = compute_hierarchical_te_fit(c_T, p_T, yT, m_p, m_c, prior)
        full_p_map, _ = compute_te_and_support_fit(p_T, yT, m_p, prior)
        for p_i, (td, ix) in enumerate(P_blocks):
            s_c = pd.Series(td[child_col][ix])
            s_p = pd.Series(td[parent_col][ix])
            fallback = s_p.map(full_p_map).fillna(prior)
            te_cols_P[p_i].append(s_c.map(full_h_map).fillna(fallback).values.astype(np.float32))

    te_mat_T = np.column_stack(te_cols_T).astype(np.float32)
    te_mat_P_list = [np.column_stack(cols).astype(np.float32) for cols in te_cols_P]

    return np.hstack([price_rel_T, te_mat_T]), [
        np.hstack([pr_p, te_p]) for pr_p, te_p in zip(price_rel_P_list, te_mat_P_list)
    ]


def extract_fold_text_and_svd(
    tr_tab: dict,
    y_all: np.ndarray,
    T_idx: np.ndarray,
    sub_folds_T: np.ndarray,
    P_blocks: list[tuple[dict, np.ndarray]],
    svd_seed: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Fit multi-view TF-IDF + SVD(64) strictly on T_idx, compute seller-grouped OOF
    linear/NB predictions on T_idx, and transform P_blocks.
    Returns:
      text_oof_T (nT, 9), svd_T (nT, 64), text_P_list, svd_P_list
    """
    yT = y_all[T_idx]
    nT = len(T_idx)
    unique_sub = np.unique(sub_folds_T)

    # View 1: Full item text (Word 40k + Char 60k)
    c_full_T = [tr_tab["full_text"][i] for i in T_idx]
    c_char_T = [tr_tab["char_text"][i] for i in T_idx]
    vw_full = TfidfVectorizer(ngram_range=(1, 2), max_features=40000, min_df=3, sublinear_tf=True, dtype=np.float32)
    vc_full = TfidfVectorizer(
        ngram_range=(3, 5), analyzer="char_wb", max_features=60000, min_df=5, sublinear_tf=True, dtype=np.float32
    )
    X_full_T = hstack([vw_full.fit_transform(c_full_T), vc_full.fit_transform(c_char_T)]).tocsr()

    # SVD(64) fit strictly on X_full_T
    svd = TruncatedSVD(n_components=64, random_state=svd_seed, algorithm="randomized", n_iter=3)
    svd_T = svd.fit_transform(X_full_T).astype(np.float32)

    # View 2: Title-only (Word 25k + Char 35k)
    c_title_T = [tr_tab["title_only"][i] for i in T_idx]
    vw_title = TfidfVectorizer(ngram_range=(1, 2), max_features=25000, min_df=3, sublinear_tf=True, dtype=np.float32)
    vc_title = TfidfVectorizer(
        ngram_range=(3, 5), analyzer="char_wb", max_features=35000, min_df=5, sublinear_tf=True, dtype=np.float32
    )
    X_title_T = hstack([vw_title.fit_transform(c_title_T), vc_title.fit_transform(c_title_T)]).tocsr()

    # View 3: Structured text (warranty + attributes + tags + payment + modes)
    c_struct_T = [tr_tab["struct_text"][i] for i in T_idx]
    vw_struct = TfidfVectorizer(ngram_range=(1, 2), max_features=15000, min_df=2, sublinear_tf=True, dtype=np.float32)
    vc_struct = TfidfVectorizer(
        ngram_range=(3, 5), analyzer="char_wb", max_features=20000, min_df=4, sublinear_tf=True, dtype=np.float32
    )
    X_struct_T = hstack([vw_struct.fit_transform(c_struct_T), vc_struct.fit_transform(c_struct_T)]).tocsr()

    # Seller-grouped OOF text predictions on T_idx
    p_lr15_T = np.zeros(nT, dtype=np.float32)
    p_lr05_T = np.zeros(nT, dtype=np.float32)
    z_ridge_T = np.zeros(nT, dtype=np.float32)
    lo_nb_T = np.zeros(nT, dtype=np.float32)
    p_title_lr_T = np.zeros(nT, dtype=np.float32)
    z_title_rg_T = np.zeros(nT, dtype=np.float32)
    p_struct_lr_T = np.zeros(nT, dtype=np.float32)
    z_struct_rg_T = np.zeros(nT, dtype=np.float32)

    for sf in unique_sub:
        va_m = sub_folds_T == sf
        tr_m = ~va_m
        y_sub = yT[tr_m]

        X_f_tr, X_f_va = X_full_T[tr_m], X_full_T[va_m]
        lr15 = LogisticRegression(C=1.5, solver="lbfgs", max_iter=250)
        lr15.fit(X_f_tr, y_sub)
        p_lr15_T[va_m] = lr15.predict_proba(X_f_va)[:, 1].astype(np.float32)

        lr05 = LogisticRegression(C=0.5, solver="lbfgs", max_iter=250)
        lr05.fit(X_f_tr, y_sub)
        p_lr05_T[va_m] = lr05.predict_proba(X_f_va)[:, 1].astype(np.float32)

        rg_f = RidgeClassifier(alpha=1.5, solver="lsqr")
        rg_f.fit(X_f_tr, y_sub)
        z_ridge_T[va_m] = rg_f.decision_function(X_f_va).astype(np.float32)

        nb_f = MultinomialNB(alpha=0.5)
        nb_f.fit(X_f_tr, y_sub)
        lprob = nb_f.predict_log_proba(X_f_va)
        lo_nb_T[va_m] = np.clip(lprob[:, 1] - lprob[:, 0], -15.0, 15.0).astype(np.float32)

        X_t_tr, X_t_va = X_title_T[tr_m], X_title_T[va_m]
        lr_t = LogisticRegression(C=1.0, solver="lbfgs", max_iter=250)
        lr_t.fit(X_t_tr, y_sub)
        p_title_lr_T[va_m] = lr_t.predict_proba(X_t_va)[:, 1].astype(np.float32)

        rg_t = RidgeClassifier(alpha=1.5, solver="lsqr")
        rg_t.fit(X_t_tr, y_sub)
        z_title_rg_T[va_m] = rg_t.decision_function(X_t_va).astype(np.float32)

        X_s_tr, X_s_va = X_struct_T[tr_m], X_struct_T[va_m]
        lr_s = LogisticRegression(C=1.0, solver="lbfgs", max_iter=250)
        lr_s.fit(X_s_tr, y_sub)
        p_struct_lr_T[va_m] = lr_s.predict_proba(X_s_va)[:, 1].astype(np.float32)

        rg_s = RidgeClassifier(alpha=1.5, solver="lsqr")
        rg_s.fit(X_s_tr, y_sub)
        z_struct_rg_T[va_m] = rg_s.decision_function(X_s_va).astype(np.float32)

    # Fit calibrated linear combiner of the 8 OOF text signals on T_idx (via the same seller-grouped sub-folds)
    raw_txt_T = np.column_stack([
        logit(p_lr15_T), logit(p_lr05_T), z_ridge_T, lo_nb_T * 0.25,
        logit(p_title_lr_T), z_title_rg_T, logit(p_struct_lr_T), z_struct_rg_T,
    ]).astype(np.float32)
    p_txt_meta_T = np.zeros(nT, dtype=np.float32)
    for sf in unique_sub:
        va_m = sub_folds_T == sf
        tr_m = ~va_m
        meta_lr = LogisticRegression(C=0.5, max_iter=200)
        meta_lr.fit(raw_txt_T[tr_m], yT[tr_m])
        p_txt_meta_T[va_m] = meta_lr.predict_proba(raw_txt_T[va_m])[:, 1].astype(np.float32)
    full_meta_lr = LogisticRegression(C=0.5, max_iter=200)
    full_meta_lr.fit(raw_txt_T, yT)

    text_oof_T = np.column_stack([
        p_lr15_T, p_lr05_T, z_ridge_T, lo_nb_T,
        p_title_lr_T, z_title_rg_T, p_struct_lr_T, z_struct_rg_T, p_txt_meta_T,
    ]).astype(np.float32)

    # Fit full-T text models and transform P_blocks
    lr15_full = LogisticRegression(C=1.5, solver="lbfgs", max_iter=250).fit(X_full_T, yT)
    lr05_full = LogisticRegression(C=0.5, solver="lbfgs", max_iter=250).fit(X_full_T, yT)
    rg_f_full = RidgeClassifier(alpha=1.5, solver="lsqr").fit(X_full_T, yT)
    nb_f_full = MultinomialNB(alpha=0.5).fit(X_full_T, yT)

    lr_t_full = LogisticRegression(C=1.0, solver="lbfgs", max_iter=250).fit(X_title_T, yT)
    rg_t_full = RidgeClassifier(alpha=1.5, solver="lsqr").fit(X_title_T, yT)

    lr_s_full = LogisticRegression(C=1.0, solver="lbfgs", max_iter=250).fit(X_struct_T, yT)
    rg_s_full = RidgeClassifier(alpha=1.5, solver="lsqr").fit(X_struct_T, yT)

    del X_full_T, X_title_T, X_struct_T
    gc.collect()

    text_P_list = []
    svd_P_list = []
    for td, ix in P_blocks:
        c_full_P = [td["full_text"][i] for i in ix]
        c_char_P = [td["char_text"][i] for i in ix]
        X_full_P = hstack([vw_full.transform(c_full_P), vc_full.transform(c_char_P)]).tocsr()
        svd_P = svd.transform(X_full_P).astype(np.float32)

        p15_P = lr15_full.predict_proba(X_full_P)[:, 1].astype(np.float32)
        p05_P = lr05_full.predict_proba(X_full_P)[:, 1].astype(np.float32)
        zr_P = rg_f_full.decision_function(X_full_P).astype(np.float32)
        lp_nb = nb_f_full.predict_log_proba(X_full_P)
        lo_P = np.clip(lp_nb[:, 1] - lp_nb[:, 0], -15.0, 15.0).astype(np.float32)
        del X_full_P

        c_title_P = [td["title_only"][i] for i in ix]
        X_title_P = hstack([vw_title.transform(c_title_P), vc_title.transform(c_title_P)]).tocsr()
        pt_P = lr_t_full.predict_proba(X_title_P)[:, 1].astype(np.float32)
        zt_P = rg_t_full.decision_function(X_title_P).astype(np.float32)
        del X_title_P

        c_struct_P = [td["struct_text"][i] for i in ix]
        X_struct_P = hstack([vw_struct.transform(c_struct_P), vc_struct.transform(c_struct_P)]).tocsr()
        ps_P = lr_s_full.predict_proba(X_struct_P)[:, 1].astype(np.float32)
        zs_P = rg_s_full.decision_function(X_struct_P).astype(np.float32)
        del X_struct_P

        raw_txt_P = np.column_stack([
            logit(p15_P), logit(p05_P), zr_P, lo_P * 0.25,
            logit(pt_P), zt_P, logit(ps_P), zs_P,
        ]).astype(np.float32)
        p_meta_P = full_meta_lr.predict_proba(raw_txt_P)[:, 1].astype(np.float32)

        text_P_list.append(
            np.column_stack([p15_P, p05_P, zr_P, lo_P, pt_P, zt_P, ps_P, zs_P, p_meta_P]).astype(np.float32)
        )
        svd_P_list.append(svd_P)

    return text_oof_T, svd_T, text_P_list, svd_P_list


def assemble_full_matrix(row_mat: np.ndarray, prior_mat: np.ndarray, text_mat: np.ndarray, svd_mat: np.ndarray) -> np.ndarray:
    """Combine row-intrinsic features, fold-local priors, OOF multi-view text signals,
    cross-modal disagreement features, and SVD components.
    """
    p_lr15 = text_mat[:, 0]
    p_title = text_mat[:, 4]
    p_struct = text_mat[:, 6]
    p_meta = text_mat[:, 8]
    cat_te_m10 = prior_mat[:, 12]  # cat_id m=10.0 column

    # Cross-modal disagreement / synergy features (100% intrinsic)
    txt_minus_cat = (p_meta - cat_te_m10).astype(np.float32)
    title_minus_struct = (p_title - p_struct).astype(np.float32)
    txt_uncertainty = (p_meta * (1.0 - p_meta)).astype(np.float32)
    logit_meta = logit(p_meta).astype(np.float32)
    logit_lr15 = logit(p_lr15).astype(np.float32)

    cross_feats = np.column_stack([
        txt_minus_cat, title_minus_struct, txt_uncertainty, logit_meta, logit_lr15
    ]).astype(np.float32)

    return np.hstack([row_mat, prior_mat, text_mat, cross_feats, svd_mat]).astype(np.float32)


def fit_predict_trees(
    X_T: np.ndarray,
    y_T: np.ndarray,
    X_P_list: list[np.ndarray],
    seed_base: int,
    is_outer_full: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Train Seller-Free Intrinsic LightGBM and CatBoost specialists on (X_T, y_T)
    and return probability predictions for each matrix in X_P_list.
    """
    lgb_preds = [np.zeros(len(X_P), dtype=np.float64) for X_P in X_P_list]
    lgb_configs = [
        {
            "n_estimators": 1050,
            "learning_rate": 0.025,
            "num_leaves": 63,
            "min_child_samples": 25,
            "subsample": 0.85,
            "subsample_freq": 1,
            "colsample_bytree": 0.72,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
            "random_state": seed_base + 11,
        },
        {
            "n_estimators": 1150,
            "learning_rate": 0.022,
            "num_leaves": 39,
            "min_child_samples": 35,
            "subsample": 0.82,
            "subsample_freq": 1,
            "colsample_bytree": 0.68,
            "reg_alpha": 1.0,
            "reg_lambda": 3.0,
            "random_state": seed_base + 29,
        },
    ]
    if is_outer_full:
        lgb_configs.append({
            "n_estimators": 1000,
            "learning_rate": 0.025,
            "num_leaves": 55,
            "min_child_samples": 20,
            "subsample": 0.85,
            "subsample_freq": 1,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.2,
            "reg_lambda": 1.5,
            "random_state": seed_base + 77,
        })

    for cfg in lgb_configs:
        model = LGBMClassifier(**cfg, n_jobs=N_THREADS, verbose=-1)
        model.fit(X_T, y_T)
        for i, X_P in enumerate(X_P_list):
            lgb_preds[i] += model.predict_proba(X_P)[:, 1] / len(lgb_configs)
        del model

    cb_preds = [np.zeros(len(X_P), dtype=np.float64) for X_P in X_P_list]
    cb_configs = [
        {
            "iterations": 1200,
            "depth": 6,
            "learning_rate": 0.042,
            "l2_leaf_reg": 4.0,
            "border_count": 128,
            "random_seed": seed_base + 101,
        }
    ]
    if is_outer_full:
        cb_configs.append({
            "iterations": 1250,
            "depth": 7,
            "learning_rate": 0.036,
            "l2_leaf_reg": 5.0,
            "border_count": 128,
            "random_seed": seed_base + 202,
        })

    for cfg in cb_configs:
        cb = CatBoostClassifier(
            **cfg,
            loss_function="Logloss",
            thread_count=N_THREADS,
            verbose=0,
        )
        cb.fit(X_T, y_T)
        for i, X_P in enumerate(X_P_list):
            cb_preds[i] += cb.predict_proba(X_P)[:, 1] / len(cb_configs)
        del cb

    gc.collect()
    return lgb_preds, cb_preds


def optimize_inner_blend(
    p_lgb_oof: np.ndarray,
    p_cb_oof: np.ndarray,
    p_txt_oof: np.ndarray,
    y_fit: np.ndarray,
    inner_f: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """Fit blend parameters strictly on fit_idx inner-OOF (4 seller-disjoint folds).
    Uses cross-fitted inner selection across the 4 inner folds so the blend combiner
    itself has zero in-sample optimism on `p_l4_fit_oof`.
    """
    def loss_fn(w):
        w = np.asarray(w, dtype=np.float64)
        w = w / np.sum(w)
        p = np.clip(w[0] * p_lgb_oof + w[1] * p_cb_oof + w[2] * p_txt_oof, 1e-7, 1.0 - 1e-7)
        return log_loss(y_fit, p)

    res = minimize(
        loss_fn,
        x0=[0.55, 0.30, 0.15],
        bounds=[(0.10, 0.85), (0.05, 0.70), (0.0, 0.35)],
    )
    w_conv = (res.x / np.sum(res.x)).tolist()
    p_conv_oof = w_conv[0] * p_lgb_oof + w_conv[1] * p_cb_oof + w_conv[2] * p_txt_oof

    # Also fit logit-space LogisticRegression across the 4 inner folds
    Z_oof = np.column_stack([logit(p_lgb_oof), logit(p_cb_oof), logit(p_txt_oof)])
    p_stack_oof = np.zeros_like(p_lgb_oof)
    for j in range(4):
        va_m = inner_f == j
        tr_m = ~va_m
        stk = LogisticRegression(C=0.5, max_iter=500).fit(Z_oof[tr_m], y_fit[tr_m])
        p_stack_oof[va_m] = stk.predict_proba(Z_oof[va_m])[:, 1]
    stk_full = LogisticRegression(C=0.5, max_iter=500).fit(Z_oof, y_fit)

    # Evaluate hybrid 0.5 * p_conv + 0.5 * p_stack vs pure p_conv vs pure p_stack on inner-OOF
    candidates = {
        "convex_prob": p_conv_oof,
        "logit_stack": p_stack_oof,
        "hybrid_50_50": 0.5 * p_conv_oof + 0.5 * p_stack_oof,
    }
    best_name = max(
        candidates.keys(),
        key=lambda k: (
            accuracy_score(y_fit, candidates[k] >= 0.5),
            -log_loss(y_fit, np.clip(candidates[k], 1e-7, 1.0 - 1e-7)),
        ),
    )
    p_blend_oof = candidates[best_name]

    info = {
        "selected_combiner": best_name,
        "w_convex_lgb_cb_txt": w_conv,
        "logit_stack_coef": stk_full.coef_[0].tolist(),
        "logit_stack_intercept": float(stk_full.intercept_[0]),
        "inner_oof_metrics": {
            name: {
                "hits": int(((arr >= 0.5) == y_fit).sum()),
                "acc": round(float(accuracy_score(y_fit, arr >= 0.5)), 5),
                "logloss": round(float(log_loss(y_fit, np.clip(arr, 1e-7, 1.0 - 1e-7))), 5),
            }
            for name, arr in {
                "L4_lgb_intrinsic": p_lgb_oof,
                "L4_cb_intrinsic": p_cb_oof,
                "L4_text_intrinsic": p_txt_oof,
                **candidates,
            }.items()
        },
    }
    return info, p_blend_oof


def apply_frozen_blend(info: dict, p_lgb: np.ndarray, p_cb: np.ndarray, p_txt: np.ndarray) -> np.ndarray:
    w = info["w_convex_lgb_cb_txt"]
    p_conv = w[0] * p_lgb + w[1] * p_cb + w[2] * p_txt
    coef = np.asarray(info["logit_stack_coef"], dtype=np.float64)
    inter = float(info["logit_stack_intercept"])
    Z = np.column_stack([logit(p_lgb), logit(p_cb), logit(p_txt)])
    p_stack = sigmoid(Z @ coef + inter)
    sel = info["selected_combiner"]
    if sel == "convex_prob":
        return p_conv
    if sel == "logit_stack":
        return p_stack
    return 0.5 * p_conv + 0.5 * p_stack


def run_outer_fold(
    k: int,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_splits: pd.DataFrame,
    tr_tab: dict,
    te_tab: dict,
) -> dict:
    fold_npz = OUT_DIR / f"fold_{k}_predictions.npz"
    fold_json = CACHE_DIR / f"fold_{k}_meta.json"
    if fold_npz.exists() and fold_json.exists():
        log(f"Outer fold {k} loaded from cache {fold_npz.name}")
        return json.loads(fold_json.read_text(encoding="utf-8"))

    t_f0 = time.time()
    role = df_splits[f"role_fold_{k}"].values
    inner_col = df_splits[f"inner_fold_{k}"].values
    y_all = df_train["y"].values
    sellers_all = df_train["seller_id"].values

    fit_idx = np.where(role == "fit")[0]
    seen_val_idx = np.where(role == "seen_val")[0]
    unseen_val_idx = np.where(role == "unseen_val")[0]
    test_idx = np.arange(len(df_test))

    inner_f = inner_col[fit_idx]
    assert np.all(np.isin(inner_f, [0, 1, 2, 3])), f"Invalid inner_fold_{k} on fit_idx"
    assert len(set(sellers_all[fit_idx]) & set(sellers_all[unseen_val_idx])) == 0, "Seller leak fit vs unseen_val!"

    log(
        f"=== Outer Fold {k}/4: fit={len(fit_idx)}, seen_val={len(seen_val_idx)}, "
        f"unseen_val={len(unseen_val_idx)}, test={len(test_idx)} ==="
    )

    n_fit = len(fit_idx)
    p_lgb_fit_oof = np.zeros(n_fit, dtype=np.float64)
    p_cb_fit_oof = np.zeros(n_fit, dtype=np.float64)
    p_txt_fit_oof = np.zeros(n_fit, dtype=np.float64)

    # Store the 4-fold seller-grouped OOF prior & text matrices on fit_idx computed across inner folds j=0..3
    prior_fit_oof = None
    text_fit_oof = None

    for j in range(4):
        t_j0 = time.time()
        va_pos = np.where(inner_f == j)[0]
        tr_pos = np.where(inner_f != j)[0]
        T_j = fit_idx[tr_pos]
        V_j = fit_idx[va_pos]
        assert len(set(sellers_all[T_j]) & set(sellers_all[V_j])) == 0, f"Inner fold {j} seller overlap!"

        # The remaining 3 inner folds partition T_j into 3 seller-disjoint groups
        sub_folds_Tj = inner_f[tr_pos]

        prior_Tj, [prior_Vj] = extract_fold_tab_and_priors(
            tr_tab, y_all, T_j, sub_folds_Tj, [(tr_tab, V_j)]
        )
        text_Tj, svd_Tj, [text_Vj], [svd_Vj] = extract_fold_text_and_svd(
            tr_tab, y_all, T_j, sub_folds_Tj, [(tr_tab, V_j)], svd_seed=42 + k * 10 + j
        )

        if prior_fit_oof is None:
            prior_fit_oof = np.zeros((n_fit, prior_Vj.shape[1]), dtype=np.float32)
            text_fit_oof = np.zeros((n_fit, text_Vj.shape[1]), dtype=np.float32)
        prior_fit_oof[va_pos] = prior_Vj
        text_fit_oof[va_pos] = text_Vj
        p_txt_fit_oof[va_pos] = text_Vj[:, 8]

        X_Tj = assemble_full_matrix(tr_tab["row_matrix"][T_j], prior_Tj, text_Tj, svd_Tj)
        X_Vj = assemble_full_matrix(tr_tab["row_matrix"][V_j], prior_Vj, text_Vj, svd_Vj)

        [lgb_Vj], [cb_Vj] = fit_predict_trees(
            X_Tj, y_all[T_j], [X_Vj], seed_base=1000 + k * 100 + j * 10, is_outer_full=False
        )
        p_lgb_fit_oof[va_pos] = lgb_Vj
        p_cb_fit_oof[va_pos] = cb_Vj

        y_Vj = y_all[V_j]
        log(
            f"  [Fold {k} Inner {j}] ({time.time()-t_j0:.1f}s) "
            f"acc: LGB={accuracy_score(y_Vj, lgb_Vj>=0.5):.4f}, "
            f"CB={accuracy_score(y_Vj, cb_Vj>=0.5):.4f}, "
            f"TXT={accuracy_score(y_Vj, text_Vj[:,8]>=0.5):.4f}"
        )
        del X_Tj, X_Vj, prior_Tj, text_Tj, svd_Tj, prior_Vj, text_Vj, svd_Vj
        gc.collect()

    # Optimize and freeze blend strictly on fit_idx inner-OOF BEFORE touching seen_val or unseen_val
    blend_info, p_l4_fit_oof = optimize_inner_blend(
        p_lgb_fit_oof, p_cb_fit_oof, p_txt_fit_oof, y_all[fit_idx], inner_f
    )
    log(
        f"  [Fold {k} Inner-OOF Frozen] combiner={blend_info['selected_combiner']}, "
        f"w={np.round(blend_info['w_convex_lgb_cb_txt'], 3)}, "
        f"Fit-OOF acc: Blend={blend_info['inner_oof_metrics'][blend_info['selected_combiner']]['acc']:.5f} "
        f"(LGB={blend_info['inner_oof_metrics']['L4_lgb_intrinsic']['acc']:.5f}, "
        f"CB={blend_info['inner_oof_metrics']['L4_cb_intrinsic']['acc']:.5f}, "
        f"TXT={blend_info['inner_oof_metrics']['L4_text_intrinsic']['acc']:.5f})"
    )

    # Outer Fold Fit on full fit_idx -> predict seen_val, unseen_val, test
    t_out0 = time.time()
    P_blocks_outer = [
        (tr_tab, seen_val_idx),
        (tr_tab, unseen_val_idx),
        (te_tab, test_idx),
    ]
    _, prior_P_outer = extract_fold_tab_and_priors(
        tr_tab, y_all, fit_idx, inner_f, P_blocks_outer
    )

    # Fit full-fit_idx TF-IDF + SVD(64) and transform seen_val, unseen_val, test
    # Note: on fit_idx itself, we use the exact 4-fold seller-grouped OOF `prior_fit_oof` and `text_fit_oof`!
    c_full_fit = [tr_tab["full_text"][i] for i in fit_idx]
    c_char_fit = [tr_tab["char_text"][i] for i in fit_idx]
    vw_full = TfidfVectorizer(ngram_range=(1, 2), max_features=40000, min_df=3, sublinear_tf=True, dtype=np.float32)
    vc_full = TfidfVectorizer(
        ngram_range=(3, 5), analyzer="char_wb", max_features=60000, min_df=5, sublinear_tf=True, dtype=np.float32
    )
    X_full_fit = hstack([vw_full.fit_transform(c_full_fit), vc_full.fit_transform(c_char_fit)]).tocsr()
    svd_full = TruncatedSVD(n_components=64, random_state=42 + k, algorithm="randomized", n_iter=3)
    svd_fit = svd_full.fit_transform(X_full_fit).astype(np.float32)

    c_title_fit = [tr_tab["title_only"][i] for i in fit_idx]
    vw_title = TfidfVectorizer(ngram_range=(1, 2), max_features=25000, min_df=3, sublinear_tf=True, dtype=np.float32)
    vc_title = TfidfVectorizer(
        ngram_range=(3, 5), analyzer="char_wb", max_features=35000, min_df=5, sublinear_tf=True, dtype=np.float32
    )
    X_title_fit = hstack([vw_title.fit_transform(c_title_fit), vc_title.fit_transform(c_title_fit)]).tocsr()

    c_struct_fit = [tr_tab["struct_text"][i] for i in fit_idx]
    vw_struct = TfidfVectorizer(ngram_range=(1, 2), max_features=15000, min_df=2, sublinear_tf=True, dtype=np.float32)
    vc_struct = TfidfVectorizer(
        ngram_range=(3, 5), analyzer="char_wb", max_features=20000, min_df=4, sublinear_tf=True, dtype=np.float32
    )
    X_struct_fit = hstack([vw_struct.fit_transform(c_struct_fit), vc_struct.fit_transform(c_struct_fit)]).tocsr()

    y_fit = y_all[fit_idx]
    lr15_full = LogisticRegression(C=1.5, solver="lbfgs", max_iter=250).fit(X_full_fit, y_fit)
    lr05_full = LogisticRegression(C=0.5, solver="lbfgs", max_iter=250).fit(X_full_fit, y_fit)
    rg_f_full = RidgeClassifier(alpha=1.5, solver="lsqr").fit(X_full_fit, y_fit)
    nb_f_full = MultinomialNB(alpha=0.5).fit(X_full_fit, y_fit)
    lr_t_full = LogisticRegression(C=1.0, solver="lbfgs", max_iter=250).fit(X_title_fit, y_fit)
    rg_t_full = RidgeClassifier(alpha=1.5, solver="lsqr").fit(X_title_fit, y_fit)
    lr_s_full = LogisticRegression(C=1.0, solver="lbfgs", max_iter=250).fit(X_struct_fit, y_fit)
    rg_s_full = RidgeClassifier(alpha=1.5, solver="lsqr").fit(X_struct_fit, y_fit)

    raw_txt_fit_oof = np.column_stack([
        logit(text_fit_oof[:, 0]), logit(text_fit_oof[:, 1]), text_fit_oof[:, 2], text_fit_oof[:, 3] * 0.25,
        logit(text_fit_oof[:, 4]), text_fit_oof[:, 5], logit(text_fit_oof[:, 6]), text_fit_oof[:, 7],
    ]).astype(np.float32)
    full_meta_lr = LogisticRegression(C=0.5, max_iter=200).fit(raw_txt_fit_oof, y_fit)

    del X_full_fit, X_title_fit, X_struct_fit
    gc.collect()

    text_P_outer = []
    svd_P_outer = []
    for td, ix in P_blocks_outer:
        c_full_P = [td["full_text"][i] for i in ix]
        c_char_P = [td["char_text"][i] for i in ix]
        X_full_P = hstack([vw_full.transform(c_full_P), vc_full.transform(c_char_P)]).tocsr()
        svd_P = svd_full.transform(X_full_P).astype(np.float32)
        p15_P = lr15_full.predict_proba(X_full_P)[:, 1].astype(np.float32)
        p05_P = lr05_full.predict_proba(X_full_P)[:, 1].astype(np.float32)
        zr_P = rg_f_full.decision_function(X_full_P).astype(np.float32)
        lp_nb = nb_f_full.predict_log_proba(X_full_P)
        lo_P = np.clip(lp_nb[:, 1] - lp_nb[:, 0], -15.0, 15.0).astype(np.float32)
        del X_full_P

        c_title_P = [td["title_only"][i] for i in ix]
        X_title_P = hstack([vw_title.transform(c_title_P), vc_title.transform(c_title_P)]).tocsr()
        pt_P = lr_t_full.predict_proba(X_title_P)[:, 1].astype(np.float32)
        zt_P = rg_t_full.decision_function(X_title_P).astype(np.float32)
        del X_title_P

        c_struct_P = [td["struct_text"][i] for i in ix]
        X_struct_P = hstack([vw_struct.transform(c_struct_P), vc_struct.transform(c_struct_P)]).tocsr()
        ps_P = lr_s_full.predict_proba(X_struct_P)[:, 1].astype(np.float32)
        zs_P = rg_s_full.decision_function(X_struct_P).astype(np.float32)
        del X_struct_P

        raw_txt_P = np.column_stack([
            logit(p15_P), logit(p05_P), zr_P, lo_P * 0.25,
            logit(pt_P), zt_P, logit(ps_P), zs_P,
        ]).astype(np.float32)
        p_meta_P = full_meta_lr.predict_proba(raw_txt_P)[:, 1].astype(np.float32)

        text_P_outer.append(
            np.column_stack([p15_P, p05_P, zr_P, lo_P, pt_P, zt_P, ps_P, zs_P, p_meta_P]).astype(np.float32)
        )
        svd_P_outer.append(svd_P)

    X_fit_full = assemble_full_matrix(tr_tab["row_matrix"][fit_idx], prior_fit_oof, text_fit_oof, svd_fit)
    X_seen_val = assemble_full_matrix(tr_tab["row_matrix"][seen_val_idx], prior_P_outer[0], text_P_outer[0], svd_P_outer[0])
    X_unseen_val = assemble_full_matrix(tr_tab["row_matrix"][unseen_val_idx], prior_P_outer[1], text_P_outer[1], svd_P_outer[1])
    X_test = assemble_full_matrix(te_tab["row_matrix"][test_idx], prior_P_outer[2], text_P_outer[2], svd_P_outer[2])

    n_features = X_fit_full.shape[1]
    [p_lgb_seen, p_lgb_unseen, p_lgb_test], [p_cb_seen, p_cb_unseen, p_cb_test] = fit_predict_trees(
        X_fit_full, y_fit, [X_seen_val, X_unseen_val, X_test], seed_base=5000 + k * 100, is_outer_full=True
    )
    p_txt_seen = text_P_outer[0][:, 8].astype(np.float64)
    p_txt_unseen = text_P_outer[1][:, 8].astype(np.float64)
    p_txt_test = text_P_outer[2][:, 8].astype(np.float64)

    p_l4_seen_val = apply_frozen_blend(blend_info, p_lgb_seen, p_cb_seen, p_txt_seen)
    p_l4_unseen_val = apply_frozen_blend(blend_info, p_lgb_unseen, p_cb_unseen, p_txt_unseen)
    p_l4_test = apply_frozen_blend(blend_info, p_lgb_test, p_cb_test, p_txt_test)

    del X_fit_full, X_seen_val, X_unseen_val, X_test, svd_fit, svd_P_outer
    gc.collect()

    np.savez_compressed(
        fold_npz,
        fit_idx=fit_idx,
        seen_val_idx=seen_val_idx,
        unseen_val_idx=unseen_val_idx,
        p_l4_fit_oof=p_l4_fit_oof,
        p_lgb_fit_oof=p_lgb_fit_oof,
        p_cb_fit_oof=p_cb_fit_oof,
        p_text_fit_oof=p_txt_fit_oof,
        p_l4_seen_val=p_l4_seen_val,
        p_lgb_seen_val=p_lgb_seen,
        p_cb_seen_val=p_cb_seen,
        p_text_seen_val=p_txt_seen,
        p_l4_unseen_val=p_l4_unseen_val,
        p_lgb_unseen_val=p_lgb_unseen,
        p_cb_unseen_val=p_cb_seen if False else p_cb_unseen,
        p_text_unseen_val=p_txt_unseen,
        p_l4_test=p_l4_test,
        p_lgb_test=p_lgb_test,
        p_cb_test=p_cb_test,
        p_text_test=p_txt_test,
    )

    y_unseen = y_all[unseen_val_idx]
    y_seen = y_all[seen_val_idx]
    meta = {
        "outer_fold": k + 1,
        "n_fit": int(len(fit_idx)),
        "n_seen_val": int(len(seen_val_idx)),
        "n_unseen_val": int(len(unseen_val_idx)),
        "n_features": int(n_features),
        "blend_info": blend_info,
        "unseen_val_hits": {
            "L4_new_seller_blend": int(((p_l4_unseen_val >= 0.5) == y_unseen).sum()),
            "L4_lgb_intrinsic": int(((p_lgb_unseen >= 0.5) == y_unseen).sum()),
            "L4_cb_intrinsic": int(((p_cb_unseen >= 0.5) == y_unseen).sum()),
            "L4_text_intrinsic": int(((p_txt_unseen >= 0.5) == y_unseen).sum()),
        },
        "unseen_val_logloss": {
            "L4_new_seller_blend": round(float(log_loss(y_unseen, np.clip(p_l4_unseen_val, 1e-7, 1 - 1e-7))), 5),
            "L4_lgb_intrinsic": round(float(log_loss(y_unseen, np.clip(p_lgb_unseen, 1e-7, 1 - 1e-7))), 5),
            "L4_cb_intrinsic": round(float(log_loss(y_unseen, np.clip(p_cb_unseen, 1e-7, 1 - 1e-7))), 5),
            "L4_text_intrinsic": round(float(log_loss(y_unseen, np.clip(p_txt_unseen, 1e-7, 1 - 1e-7))), 5),
        },
        "seen_val_hits": {
            "L4_new_seller_blend": int(((p_l4_seen_val >= 0.5) == y_seen).sum()),
            "L4_lgb_intrinsic": int(((p_lgb_seen >= 0.5) == y_seen).sum()),
            "L4_cb_intrinsic": int(((p_cb_seen >= 0.5) == y_seen).sum()),
            "L4_text_intrinsic": int(((p_txt_seen >= 0.5) == y_seen).sum()),
        },
        "seen_val_logloss": {
            "L4_new_seller_blend": round(float(log_loss(y_seen, np.clip(p_l4_seen_val, 1e-7, 1 - 1e-7))), 5),
            "L4_lgb_intrinsic": round(float(log_loss(y_seen, np.clip(p_lgb_seen, 1e-7, 1 - 1e-7))), 5),
            "L4_cb_intrinsic": round(float(log_loss(y_seen, np.clip(p_cb_seen, 1e-7, 1 - 1e-7))), 5),
            "L4_text_intrinsic": round(float(log_loss(y_seen, np.clip(p_txt_seen, 1e-7, 1 - 1e-7))), 5),
        },
        "elapsed_seconds": round(time.time() - t_f0, 1),
    }
    fold_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(
        f"=== Outer Fold {k} Done ({meta['elapsed_seconds']}s) | "
        f"Unseen-Val Hits: Blend={meta['unseen_val_hits']['L4_new_seller_blend']}/14000 "
        f"({meta['unseen_val_hits']['L4_new_seller_blend']/14000:.5f}), "
        f"LGB={meta['unseen_val_hits']['L4_lgb_intrinsic']}, "
        f"CB={meta['unseen_val_hits']['L4_cb_intrinsic']}, "
        f"TXT={meta['unseen_val_hits']['L4_text_intrinsic']} | "
        f"Seen-Val Hits: Blend={meta['seen_val_hits']['L4_new_seller_blend']}/{len(seen_val_idx)} "
        f"({meta['seen_val_hits']['L4_new_seller_blend']/len(seen_val_idx):.5f}) ==="
    )
    return meta


def main():
    t_start = time.time()
    log("Loading frozen shared split and parsed train/test caches...")
    df_train = pd.read_pickle(SPLIT_DIR / "t98_parsed_train.pkl")
    df_test = pd.read_pickle(SPLIT_DIR / "t98_parsed_test.pkl")
    df_splits = pd.read_csv(SPLIT_DIR / "t98_dual_regime_outer_seller_group_splits.csv")

    assert len(df_train) == 70000 and len(df_splits) == 70000 and len(df_test) == 30000
    assert np.array_equal(df_train["row"].values, df_splits["row"].values)

    log("Precomputing 100% row-local item-intrinsic tables for train and test...")
    tr_tab = precompute_row_intrinsic_table(df_train)
    te_tab = precompute_row_intrinsic_table(df_test)

    fold_metas = []
    for k in range(MAX_FOLDS):
        fm = run_outer_fold(k, df_train, df_test, df_splits, tr_tab, te_tab)
        fold_metas.append(fm)

    if MAX_FOLDS < 5:
        log(f"Stopped after MAX_FOLDS={MAX_FOLDS} as requested.")
        return

    log("All 5 outer folds complete. Aggregating predictions and computing paired comparisons vs T97...")
    n_train = len(df_train)
    n_test = len(df_test)

    oof_unseen = {
        "L4_new_seller_blend": np.zeros(n_train, dtype=np.float64),
        "L4_lgb_intrinsic": np.zeros(n_train, dtype=np.float64),
        "L4_cb_intrinsic": np.zeros(n_train, dtype=np.float64),
        "L4_text_intrinsic": np.zeros(n_train, dtype=np.float64),
    }
    test_preds = {
        "L4_new_seller_blend": np.zeros(n_test, dtype=np.float64),
        "L4_lgb_intrinsic": np.zeros(n_test, dtype=np.float64),
        "L4_cb_intrinsic": np.zeros(n_test, dtype=np.float64),
        "L4_text_intrinsic": np.zeros(n_test, dtype=np.float64),
    }
    test_fold_cols = {}
    seen_rows_list = []

    for k in range(5):
        z = np.load(OUT_DIR / f"fold_{k}_predictions.npz")
        u_idx = z["unseen_val_idx"]
        s_idx = z["seen_val_idx"]

        oof_unseen["L4_new_seller_blend"][u_idx] = z["p_l4_unseen_val"]
        oof_unseen["L4_lgb_intrinsic"][u_idx] = z["p_lgb_unseen_val"]
        oof_unseen["L4_cb_intrinsic"][u_idx] = z["p_cb_unseen_val"]
        oof_unseen["L4_text_intrinsic"][u_idx] = z["p_text_unseen_val"]

        test_preds["L4_new_seller_blend"] += z["p_l4_test"] / 5.0
        test_preds["L4_lgb_intrinsic"] += z["p_lgb_test"] / 5.0
        test_preds["L4_cb_intrinsic"] += z["p_cb_test"] / 5.0
        test_preds["L4_text_intrinsic"] += z["p_text_test"] / 5.0
        test_fold_cols[f"p_l4_fold_{k}"] = z["p_l4_test"]

        seen_df_k = pd.DataFrame({
            "outer_fold": k + 1,
            "row": s_idx,
            "id": df_train["id"].values[s_idx],
            "seller_id": df_train["seller_id"].values[s_idx],
            "category_id": df_train["category_id"].values[s_idx],
            "y": df_train["y"].values[s_idx],
            "p_lgb_intrinsic": z["p_lgb_seen_val"],
            "p_cb_intrinsic": z["p_cb_seen_val"],
            "p_text_intrinsic": z["p_text_seen_val"],
            "p_l4_new_seller_blend": z["p_l4_seen_val"],
        })
        seen_rows_list.append(seen_df_k)

    # Save oof_unseen_70k.csv
    oof_unseen_df = pd.DataFrame({
        "row": df_train["row"].values,
        "id": df_train["id"].values,
        "seller_id": df_train["seller_id"].values,
        "category_id": df_train["category_id"].values,
        "outer_fold": df_train["outer_fold"].values + 1,
        "y": df_train["y"].values,
        "p_lgb_intrinsic": oof_unseen["L4_lgb_intrinsic"],
        "p_cb_intrinsic": oof_unseen["L4_cb_intrinsic"],
        "p_text_intrinsic": oof_unseen["L4_text_intrinsic"],
        "p_l4_new_seller_blend": oof_unseen["L4_new_seller_blend"],
    })
    oof_unseen_path = OUT_DIR / "oof_unseen_70k.csv"
    oof_unseen_df.to_csv(oof_unseen_path, index=False)

    # Save oof_seen_all_folds.csv
    oof_seen_df = pd.concat(seen_rows_list, ignore_index=True)
    oof_seen_path = OUT_DIR / "oof_seen_all_folds.csv"
    oof_seen_df.to_csv(oof_seen_path, index=False)

    # Save test_predictions_30k.csv (probabilities only, zero inferred labels)
    test_df_out = pd.DataFrame({
        "ID": df_test["row"].values + 1,
        "row": df_test["row"].values,
        "seller_id": df_test["seller_id"].values,
        "category_id": df_test["category_id"].values,
        "p_lgb_intrinsic": test_preds["L4_lgb_intrinsic"],
        "p_cb_intrinsic": test_preds["L4_cb_intrinsic"],
        "p_text_intrinsic": test_preds["L4_text_intrinsic"],
        "p_l4_new_seller_blend": test_preds["L4_new_seller_blend"],
        **test_fold_cols,
    })
    test_preds_path = OUT_DIR / "test_predictions_30k.csv"
    test_df_out.to_csv(test_preds_path, index=False)

    # Save test_disagreements_vs_s4_probs_only.csv (ONLY ID and model probabilities, zero inferred labels!)
    s4_prob_path = ROOT / "results" / "dual_blend_winner_20261111" / "test_probabilities_dual_blend_winner.csv"
    s4_df = pd.read_csv(s4_prob_path, usecols=["ID", "p_lgb", "p_cb", "p_podium_blend", "p_dual_blend_winner"])
    dis_mask = (test_preds["L4_new_seller_blend"] >= 0.5) != (s4_df["p_podium_blend"].values >= 0.5)
    dis_df = pd.DataFrame({
        "ID": s4_df["ID"].values[dis_mask],
        "p_l4_new_seller_blend": test_preds["L4_new_seller_blend"][dis_mask],
        "p_lgb_intrinsic": test_preds["L4_lgb_intrinsic"][dis_mask],
        "p_cb_intrinsic": test_preds["L4_cb_intrinsic"][dis_mask],
        "p_text_intrinsic": test_preds["L4_text_intrinsic"][dis_mask],
        "p_s4_podium_blend": s4_df["p_podium_blend"].values[dis_mask],
        "p_dual_blend_winner": s4_df["p_dual_blend_winner"].values[dis_mask],
        "abs_prob_diff_vs_s4": np.abs(
            test_preds["L4_new_seller_blend"][dis_mask] - s4_df["p_podium_blend"].values[dis_mask]
        ),
    }).sort_values("abs_prob_diff_vs_s4", ascending=False)
    dis_path = OUT_DIR / "test_disagreements_vs_s4_probs_only.csv"
    dis_df.to_csv(dis_path, index=False)

    # Paired Comparison vs T97 Policy A (58,003 hits) and T97 LightGBM (58,333 hits)
    t97_df = pd.read_csv(ROOT / "results" / "seller_group_nested_podium_20261111" / "paired_outer_val_predictions.csv")
    assert np.array_equal(t97_df["row"].values, df_train["row"].values)
    assert np.array_equal(t97_df["outer_fold"].values, df_train["outer_fold"].values + 1)

    y_true = df_train["y"].values
    sellers = df_train["seller_id"].values
    cats = df_train["category_id"].values
    outer_f1 = df_train["outer_fold"].values + 1

    p_t97_A = t97_df["p_A_podium_recipe"].values
    p_t97_lgb = t97_df["p_lgb"].values
    p_t97_cb = t97_df["p_cb"].values
    p_t97_C = t97_df["p_C_logistic_stack"].values

    c_t97_A = (p_t97_A >= 0.5) == y_true
    c_t97_lgb = (p_t97_lgb >= 0.5) == y_true

    per_fold_rows = []
    for k in range(5):
        f_num = k + 1
        m_f = outer_f1 == f_num
        yv = y_true[m_f]
        cA_f = c_t97_A[m_f]
        cL_f = c_t97_lgb[m_f]

        row_dict = {
            "outer_fold": f_num,
            "n_unseen_val": int(m_f.sum()),
            "n_seen_val": fold_metas[k]["n_seen_val"],
            "n_fit": fold_metas[k]["n_fit"],
            "hits_T97_PolicyA": int(cA_f.sum()),
            "ll_T97_PolicyA": round(float(log_loss(yv, np.clip(p_t97_A[m_f], 1e-7, 1 - 1e-7))), 5),
            "hits_T97_LGB": int(cL_f.sum()),
            "ll_T97_LGB": round(float(log_loss(yv, np.clip(p_t97_lgb[m_f], 1e-7, 1 - 1e-7))), 5),
            "hits_T97_CB": int(((p_t97_cb[m_f] >= 0.5) == yv).sum()),
            "ll_T97_CB": round(float(log_loss(yv, np.clip(p_t97_cb[m_f], 1e-7, 1 - 1e-7))), 5),
        }
        for m_name, p_arr in oof_unseen.items():
            pf = np.clip(p_arr[m_f], 1e-7, 1 - 1e-7)
            cf = (pf >= 0.5) == yv
            row_dict[f"hits_{m_name}"] = int(cf.sum())
            row_dict[f"acc_{m_name}"] = round(float(cf.mean()), 5)
            row_dict[f"ll_{m_name}"] = round(float(log_loss(yv, pf)), 5)
            row_dict[f"fixes_{m_name}_vs_T97_A"] = int((cf & ~cA_f).sum())
            row_dict[f"breaks_{m_name}_vs_T97_A"] = int((~cf & cA_f).sum())
            row_dict[f"delta_{m_name}_vs_T97_A"] = int(cf.sum() - cA_f.sum())
            row_dict[f"fixes_{m_name}_vs_T97_LGB"] = int((cf & ~cL_f).sum())
            row_dict[f"breaks_{m_name}_vs_T97_LGB"] = int((~cf & cL_f).sum())
            row_dict[f"delta_{m_name}_vs_T97_LGB"] = int(cf.sum() - cL_f.sum())

        row_dict["seen_val_hits_L4_blend"] = fold_metas[k]["seen_val_hits"]["L4_new_seller_blend"]
        row_dict["seen_val_acc_L4_blend"] = round(
            fold_metas[k]["seen_val_hits"]["L4_new_seller_blend"] / fold_metas[k]["n_seen_val"], 5
        )
        row_dict["seen_val_ll_L4_blend"] = fold_metas[k]["seen_val_logloss"]["L4_new_seller_blend"]
        per_fold_rows.append(row_dict)

    per_fold_df = pd.DataFrame(per_fold_rows)
    per_fold_path = OUT_DIR / "per_fold_metrics.csv"
    per_fold_df.to_csv(per_fold_path, index=False)

    # Compute overall summary and 2,000-replicate seller x category cluster bootstrap
    summary = {
        "T97_PolicyA": {
            "hits": int(c_t97_A.sum()),
            "accuracy": round(float(c_t97_A.mean()), 5),
            "logloss": round(float(log_loss(y_true, np.clip(p_t97_A, 1e-7, 1 - 1e-7))), 5),
        },
        "T97_LGB": {
            "hits": int(c_t97_lgb.sum()),
            "accuracy": round(float(c_t97_lgb.mean()), 5),
            "logloss": round(float(log_loss(y_true, np.clip(p_t97_lgb, 1e-7, 1 - 1e-7))), 5),
        },
        "T97_CB": {
            "hits": int(((p_t97_cb >= 0.5) == y_true).sum()),
            "accuracy": round(float(((p_t97_cb >= 0.5) == y_true).mean()), 5),
            "logloss": round(float(log_loss(y_true, np.clip(p_t97_cb, 1e-7, 1 - 1e-7))), 5),
        },
        "T97_PolicyC": {
            "hits": int(((p_t97_C >= 0.5) == y_true).sum()),
            "accuracy": round(float(((p_t97_C >= 0.5) == y_true).mean()), 5),
            "logloss": round(float(log_loss(y_true, np.clip(p_t97_C, 1e-7, 1 - 1e-7))), 5),
        },
    }
    for m_name, p_arr in oof_unseen.items():
        c_m = (p_arr >= 0.5) == y_true
        summary[m_name] = {
            "hits": int(c_m.sum()),
            "accuracy": round(float(c_m.mean()), 5),
            "logloss": round(float(log_loss(y_true, np.clip(p_arr, 1e-7, 1 - 1e-7))), 5),
        }

    # Seen-val overall summary across all 5 outer folds (~32,130 rows)
    y_seen_all = oof_seen_df["y"].values
    seen_summary = {}
    for col_key, m_name in [
        ("p_l4_new_seller_blend", "L4_new_seller_blend"),
        ("p_lgb_intrinsic", "L4_lgb_intrinsic"),
        ("p_cb_intrinsic", "L4_cb_intrinsic"),
        ("p_text_intrinsic", "L4_text_intrinsic"),
    ]:
        p_s = np.clip(oof_seen_df[col_key].values, 1e-7, 1 - 1e-7)
        c_s = (p_s >= 0.5) == y_seen_all
        seen_summary[m_name] = {
            "n_rows": int(len(y_seen_all)),
            "hits": int(c_s.sum()),
            "accuracy": round(float(c_s.mean()), 5),
            "logloss": round(float(log_loss(y_seen_all, p_s)), 5),
        }

    comparisons = {}
    baselines = {
        "T97_PolicyA": (p_t97_A, c_t97_A, "delta_{m}_vs_T97_A"),
        "T97_LGB": (p_t97_lgb, c_t97_lgb, "delta_{m}_vs_T97_LGB"),
    }
    for m_name in ["L4_new_seller_blend", "L4_lgb_intrinsic", "L4_cb_intrinsic"]:
        p_m = np.clip(oof_unseen[m_name], 1e-7, 1 - 1e-7)
        c_m = (p_m >= 0.5) == y_true
        for b_name, (p_b, c_b, col_pat) in baselines.items():
            p_b_clip = np.clip(p_b, 1e-7, 1 - 1e-7)
            diff_hits = c_m.astype(np.float64) - c_b.astype(np.float64)
            bs_hits = boot_two_way(diff_hits, sellers, cats, n_boot=2000, seed=20261111) * n_train
            lld = (
                -(y_true * np.log(p_m) + (1 - y_true) * np.log(1 - p_m))
                + (y_true * np.log(p_b_clip) + (1 - y_true) * np.log(1 - p_b_clip))
            )
            bs_ll = boot_two_way(lld, sellers, cats, n_boot=2000, seed=20261112)
            fold_deltas = per_fold_df[col_pat.format(m=m_name)].values
            comparisons[f"{m_name}_vs_{b_name}"] = {
                "delta_hits_70k": int(diff_hits.sum()),
                "fixes": int((diff_hits == 1).sum()),
                "breaks": int((diff_hits == -1).sum()),
                "fold_deltas": [int(x) for x in fold_deltas],
                "folds_positive": int((fold_deltas > 0).sum()),
                "folds_zero": int((fold_deltas == 0).sum()),
                "worst_fold_delta": int(fold_deltas.min()),
                "boot_delta_hits_median": round(float(np.median(bs_hits)), 2),
                "boot_delta_hits_q05": round(float(np.quantile(bs_hits, 0.05)), 2),
                "boot_delta_hits_q95": round(float(np.quantile(bs_hits, 0.95)), 2),
                "boot_prob_positive": round(float((bs_hits > 0).mean()), 4),
                "delta_logloss": round(float(lld.mean()), 6),
                "boot_delta_logloss_q05": round(float(np.quantile(bs_ll, 0.05)), 6),
                "boot_delta_logloss_q95": round(float(np.quantile(bs_ll, 0.95)), 6),
            }

    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        git_head = "unknown"

    manifest = {
        "task": "T98_Line4_Unseen_Seller_Intrinsic_Specialist",
        "status": "DEVELOPMENT_ONLY",
        "submission_generated": False,
        "kaggle_api_used": False,
        "seller_identity_used_in_features": False,
        "global_frequencies_used_in_features": False,
        "in_sample_text_oof_leakage": False,
        "git_head_at_run": git_head,
        "script": "scripts/t98_line4_new_seller_specialist.py",
        "script_sha256": sha256_file(Path(__file__)),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "catboost": catboost.__version__,
        },
        "unseen_val_summary_70k": summary,
        "seen_val_summary_all_folds": seen_summary,
        "paired_comparisons_unseen_70k": comparisons,
        "test_disagreements_vs_s4_count": int(len(dis_df)),
        "folds": fold_metas,
        "artifacts": {
            "oof_unseen_70k_csv": oof_unseen_path.name,
            "oof_unseen_70k_sha256": sha256_file(oof_unseen_path),
            "oof_seen_all_folds_csv": oof_seen_path.name,
            "oof_seen_all_folds_sha256": sha256_file(oof_seen_path),
            "test_predictions_30k_csv": test_preds_path.name,
            "test_predictions_30k_sha256": sha256_file(test_preds_path),
            "test_disagreements_vs_s4_probs_only_csv": dis_path.name,
            "test_disagreements_vs_s4_probs_only_sha256": sha256_file(dis_path),
            "per_fold_metrics_csv": per_fold_path.name,
            "per_fold_metrics_sha256": sha256_file(per_fold_path),
            "fold_npz_sha256": {
                f"fold_{k}_predictions.npz": sha256_file(OUT_DIR / f"fold_{k}_predictions.npz")
                for k in range(5)
            },
        },
        "runtime_minutes": round((time.time() - t_start) / 60.0, 2),
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(json.dumps({"unseen_val_summary_70k": summary, "paired_comparisons_unseen_70k": comparisons}, indent=2))
    log(f"DONE in {manifest['runtime_minutes']} min. Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
