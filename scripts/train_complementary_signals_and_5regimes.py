"""AP-2026-Q-2: Comprehensive Complementary Signal Training and 5-Regime Evaluation.

Protocol:
1. Outer Seller-Group CV (5 folds, 70,000 rows, 100% disjoint sellers between outer train and outer val).
2. Clean Fold-Level Target Encoding (inner StratifiedGroupKFold on outer train, zero leakage into val/test).
3. Complementary signals:
   - Text with negations and product condition state (p_text_negation via TF-IDF + Logistic Regression).
   - Structured attributes (item_cond_attr, brand_val, model_val, listing_type, location, shipping).
   - Commercial & price features (relative price, percentile, discount, quantities, velocity).
   - Identity & defect regex specifications (parts vs main, vintage, defect vs new cue balance).
   - Cross-seller similar product retrieval (cross_seller_used_ratio, count, price ratio).
   - Seller history policy comparison:
     * Policy A: Causal past-only (t_train < t_item).
     * Policy B: Bidirectional all-train (offline batch, all train rows of seller).
4. Models trained & compared:
   - M0_Baseline: Strong local baseline (Text OOF + Causal Seller TE + SVD).
   - M1_Enriched_Attr: M0 + Structured attributes + Commercial price ratios + Identity specs.
   - M2_Negation_Text: M1 + Specialized negation text probability.
   - M3_CrossSeller_Retrieval: M2 + Cross-seller similar product retrieval.
   - M4A_Full_PolicyA: M3 with Policy A (Causal strictly past seller history).
   - M4B_Full_PolicyB: M3 with Policy B (Bidirectional all-train seller history).
   - M5_CatBoost_Native: CatBoost with native categorical handling.
   - M6_Meta_Stacker: Cross-fitted stacking ensemble combining M0..M5.
5. Evaluation across the 5 real test regimes (30,000 rows):
   - 8,513 Cold New Sellers
   - 10,140 Strictly Bracketed without timestamp equality
   - 5,108 Same Timestamp Ties with any train row
   - 3,118 Before (t < t_min_train)
   - 3,121 After (t > t_max_train)
6. Paired metrics vs M0_Baseline and vs S4/EVT-154 anchor (SUBMISSION_4_TODAY_RECOMMENDED.csv).
7. Zero Kaggle / zero quota by agents. Gate C NO_IDENTIFIABLE.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent

# --- Regex Specifications ---
PART_REGEX = re.compile(
    r"\b(?:repuesto|repuestos|accesorio|accesorios|funda|fundas|cargador|cargadores|"
    r"bateria|baterias|cable|cables|tapa|tapas|modulo|modulos|vidrio|pantalla|pantallas|"
    r"paragolpe|optica|espejo|bomba|filtro|filtros|embrague|amortiguador|amortiguadores|"
    r"pastillas?|disco|discos|correa|correas|bujia|bujias|soporte|soportes|adaptador|"
    r"adaptadores|carcasa|calcomania|stickers?|faro|faros|radiador|guardabarro|"
    r"guardabarros|biela|piston|arbol de leva|valvula|valvulas)\b",
    re.IGNORECASE,
)
DEFECT_REGEX = re.compile(
    r"\b(?:para repuesto|para despiece|para desarme|no funciona|a reparar|con detalle|"
    r"con detalles|pantalla rota|bloqueado|sin probar|roto|rota|detalles de uso|"
    r"desgaste|marcas de uso|usado|usada|segunda mano|leido|subrayado|hojas amarillas|despiece)\b",
    re.IGNORECASE,
)
NEW_CUE_REGEX = re.compile(
    r"\b(?:nuevo|nueva|sellado|sellada|en caja|cerrada|cerrado|en blister|"
    r"con etiqueta|sin abrir|0km|a estrenar|factura|garantia oficial)\b",
    re.IGNORECASE,
)
YEAR_REGEX = re.compile(r"\b(19\d\d|200\d|201[0-5])\b")
CAP_REGEX = re.compile(r"\b(?:\d+)\s*(?:gb|mb|tb)\b", re.IGNORECASE)
DIM_REGEX = re.compile(
    r"(?:\b\d+(?:\.\d+)?\s*(?:cm|mm|m|kg|g|gr|ml|l|lt|litros?|oz|pulgadas?|pulg)\b|\b\d+x\d+\b|\b\d+[\'\"”]\b)",
    re.IGNORECASE,
)
MODEL_CODE_REGEX = re.compile(r"\b(?:[a-z]\d{1,4}|\d{1,4}[a-z]|[a-z]{1,3}-\d{2,4})\b", re.IGNORECASE)
PACK_REGEX = re.compile(r"\b(?:pack|kit|lote|combo|x\s*\d+|\d+\s*(?:unidades|piezas|u\b|unid))\b", re.IGNORECASE)
COLLECTIBLE_REGEX = re.compile(r"\b(?:coleccion|colección|antiguo|antigua|retro|vintage|inundacion|serie completa|fasciculo|tomo|revista)\b", re.IGNORECASE)

STOPWORDS = {"para", "con", "sin", "por", "del", "las", "los", "una", "uno", "que", "mas", "muy", "de", "en", "el", "la"}


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def extract_regex_features(df: pd.DataFrame) -> dict:
    titles = df["title"].fillna("").astype(str).str.lower()
    attrs = df["attr_text"].fillna("").astype(str).str.lower()
    combined = titles + " " + attrs

    is_part = combined.str.contains(PART_REGEX).to_numpy().astype(np.float32)
    has_defect = combined.str.contains(DEFECT_REGEX).to_numpy().astype(np.float32)
    has_new_cue = combined.str.contains(NEW_CUE_REGEX).to_numpy().astype(np.float32)
    cue_balance = has_defect - has_new_cue

    year_matches = combined.apply(lambda s: [int(y) for y in YEAR_REGEX.findall(s)])
    has_vintage = year_matches.apply(lambda l: any(y <= 2012 for y in l)).to_numpy().astype(np.float32)
    min_year = year_matches.apply(lambda l: min(l) if len(l) > 0 else 0).to_numpy()
    year_num = np.where(min_year > 0, (min_year - 2000.0) / 10.0, 0.0).astype(np.float32)

    vintage_x_part = (has_vintage * is_part).astype(np.float32)
    vintage_x_main = (has_vintage * (1.0 - is_part)).astype(np.float32)

    has_cap = combined.str.contains(CAP_REGEX).to_numpy().astype(np.float32)
    has_dim = combined.str.contains(DIM_REGEX).to_numpy().astype(np.float32)
    has_model_code = combined.str.contains(MODEL_CODE_REGEX).to_numpy().astype(np.float32)
    has_pack = combined.str.contains(PACK_REGEX).to_numpy().astype(np.float32)
    has_collectible = combined.str.contains(COLLECTIBLE_REGEX).to_numpy().astype(np.float32)

    return {
        "is_part": is_part,
        "has_defect": has_defect,
        "has_new_cue": has_new_cue,
        "cue_balance": cue_balance,
        "has_vintage": has_vintage,
        "year_num": year_num,
        "vintage_x_part": vintage_x_part,
        "vintage_x_main": vintage_x_main,
        "has_cap": has_cap,
        "has_dim": has_dim,
        "has_model_code": has_model_code,
        "has_pack": has_pack,
        "has_collectible": has_collectible,
    }


def compute_fold_te(
    cats_tr: np.ndarray,
    y_tr: np.ndarray,
    sellers_tr: np.ndarray,
    cats_eval_list: list,
    m: float = 5.0,
    n_inner_splits: int = 4,
    seed: int = 42,
) -> tuple:
    """Strictly leak-free Target Encoding computed inside outer train using inner StratifiedGroupKFold."""
    prior = float(y_tr.mean())
    n_tr = len(y_tr)
    te_tr = np.full(n_tr, prior, dtype=np.float32)

    sgkf = StratifiedGroupKFold(n_splits=n_inner_splits, shuffle=True, random_state=seed)
    for in_tr, in_va in sgkf.split(cats_tr, y_tr, sellers_tr):
        c_sub = cats_tr[in_tr]
        y_sub = y_tr[in_tr]
        prior_sub = float(y_sub.mean())
        grp = pd.DataFrame({"k": c_sub, "y": y_sub}).groupby("k")["y"].agg(["sum", "count"])
        m_p = (grp["sum"] + m * prior_sub) / (grp["count"] + m)
        te_tr[in_va] = pd.Series(cats_tr[in_va]).map(m_p).fillna(prior_sub).values.astype(np.float32)

    # Full map for evaluation sets (outer val and test)
    full_grp = pd.DataFrame({"k": cats_tr, "y": y_tr}).groupby("k")["y"].agg(["sum", "count"])
    full_mp = (full_grp["sum"] + m * prior) / (full_grp["count"] + m)

    eval_outs = []
    for c_ev in cats_eval_list:
        ev_te = pd.Series(c_ev).map(full_mp).fillna(prior).values.astype(np.float32)
        eval_outs.append(ev_te)

    return te_tr, eval_outs


def compute_cross_seller_features(
    keys_tr: np.ndarray,
    y_tr: np.ndarray,
    sellers_tr: np.ndarray,
    prices_tr: np.ndarray,
    keys_eval_list: list,
    prices_eval_list: list,
    m: float = 5.0,
) -> tuple:
    """Fast, leak-free cross-seller retrieval features."""
    prior = float(y_tr.mean())
    df_k = pd.DataFrame({"k": keys_tr, "s": sellers_tr, "y": y_tr, "p": prices_tr})

    # Totals by key
    grp_tot = df_k.groupby("k")
    tot_sum = grp_tot["y"].sum()
    tot_cnt = grp_tot["y"].count()
    tot_med_p = grp_tot["p"].median()

    s_k = pd.Series(keys_tr)
    row_tot_sum = s_k.map(tot_sum).values
    row_tot_cnt = s_k.map(tot_cnt).values
    row_tot_med_p = s_k.map(tot_med_p).values

    # Totals by (key, seller)
    grp_ks = df_k.groupby(["k", "s"])
    ks_sum = grp_ks["y"].sum()
    ks_cnt = grp_ks["y"].count()

    ks_tuples = list(zip(keys_tr, sellers_tr))
    row_ks_sum = ks_sum.reindex(ks_tuples).fillna(0).values
    row_ks_cnt = ks_cnt.reindex(ks_tuples).fillna(0).values

    other_cnt = np.maximum(0, row_tot_cnt - row_ks_cnt)
    other_sum = np.maximum(0, row_tot_sum - row_ks_sum)
    other_used_ratio = ((other_sum + m * prior) / (other_cnt + m)).astype(np.float32)
    has_match = (other_cnt >= 1).astype(np.float32)
    p_ratio = np.log1p(prices_tr / (row_tot_med_p + 1.0)).astype(np.float32)

    tr_feats = np.column_stack([has_match, other_used_ratio, np.log1p(other_cnt), p_ratio])

    # Eval sets
    eval_outs = []
    ratio_map = (tot_sum + m * prior) / (tot_cnt + m)
    for k_ev, p_ev in zip(keys_eval_list, prices_eval_list):
        s_k_ev = pd.Series(k_ev)
        ev_cnt = s_k_ev.map(tot_cnt).fillna(0).values.astype(np.float32)
        ev_has = (ev_cnt >= 1).astype(np.float32)
        ev_ratio = s_k_ev.map(ratio_map).fillna(prior).values.astype(np.float32)
        ev_med_p = s_k_ev.map(tot_med_p).fillna(float(np.median(prices_tr))).values.astype(np.float32)
        ev_p_ratio = np.log1p(p_ev / (ev_med_p + 1.0)).astype(np.float32)
        eval_outs.append(np.column_stack([ev_has, ev_ratio, np.log1p(ev_cnt), ev_p_ratio]))

    return tr_feats, eval_outs


def compute_seller_history_policies(
    df_train_sub: pd.DataFrame,
    df_eval_list: list,
    m: float = 5.0,
) -> tuple:
    """Compute Seller History under Policy A (Causal Past-Only) and Policy B (Bidirectional Batch)."""
    y_tr = df_train_sub["y"].values
    sellers_tr = df_train_sub["seller_id"].values
    start_tr = df_train_sub["start_time"].values
    row_tr = df_train_sub["row"].values
    prior = float(y_tr.mean())

    # --- Policy A: Causal Strictly Past on train ---
    df_s = pd.DataFrame({
        "pos": np.arange(len(y_tr)),
        "seller_id": sellers_tr,
        "start_time": start_tr,
        "row": row_tr,
        "y": y_tr,
    }).sort_values(["seller_id", "start_time", "row"])

    grp = df_s.groupby("seller_id", sort=False)
    causal_cnt_sorted = grp.cumcount().values.astype(np.float32)
    cum_y_sorted = grp["y"].cumsum().values.astype(np.float32)
    prior_sum_sorted = cum_y_sorted - df_s["y"].values.astype(np.float32)
    causal_te_sorted = (prior_sum_sorted + m * prior) / (causal_cnt_sorted + m)

    pos_order = df_s["pos"].values
    te_causal_tr = np.empty(len(y_tr), dtype=np.float32)
    cnt_causal_tr = np.empty(len(y_tr), dtype=np.float32)
    te_causal_tr[pos_order] = causal_te_sorted
    cnt_causal_tr[pos_order] = np.log1p(causal_cnt_sorted)

    # --- Policy B: Bidirectional Batch on train (LOO) ---
    s_series = pd.Series(sellers_tr)
    grp_b = pd.DataFrame({"s": sellers_tr, "y": y_tr}).groupby("s")["y"]
    s_counts = grp_b.count()
    s_sums = grp_b.sum()
    row_cnt = s_series.map(s_counts).values
    row_sum = s_series.map(s_sums).values
    loo_cnt = np.maximum(0, row_cnt - 1)
    loo_sum = np.maximum(0, row_sum - y_tr)
    te_batch_tr = ((loo_sum + m * prior) / (loo_cnt + m)).astype(np.float32)
    cnt_batch_tr = np.log1p(loo_cnt).astype(np.float32)

    tr_a = np.column_stack([te_causal_tr, cnt_causal_tr])
    tr_b = np.column_stack([te_batch_tr, cnt_batch_tr])

    # Index train items by seller for eval
    seller_time_map = {}
    for s_id, sub_df in df_train_sub.groupby("seller_id"):
        seller_time_map[s_id] = (sub_df["start_time"].values, sub_df["y"].values)

    seller_full_map = dict(zip(s_sums.index, (s_sums + m * prior) / (s_counts + m)))
    seller_cnt_map = dict(zip(s_counts.index, np.log1p(s_counts)))

    eval_a_outs = []
    eval_b_outs = []

    for df_ev in df_eval_list:
        s_ev = df_ev["seller_id"].values
        t_ev = df_ev["start_time"].values
        n_ev = len(df_ev)

        ev_te_a = np.full(n_ev, prior, dtype=np.float32)
        ev_cnt_a = np.zeros(n_ev, dtype=np.float32)

        ev_te_b = np.array([seller_full_map.get(s, prior) for s in s_ev], dtype=np.float32)
        ev_cnt_b = np.array([seller_cnt_map.get(s, 0.0) for s in s_ev], dtype=np.float32)

        for i in range(n_ev):
            s = s_ev[i]
            if s in seller_time_map:
                times_tr, ys_tr = seller_time_map[s]
                past_mask = times_tr < t_ev[i]
                n_past = past_mask.sum()
                if n_past > 0:
                    ev_te_a[i] = (ys_tr[past_mask].sum() + m * prior) / (n_past + m)
                    ev_cnt_a[i] = np.log1p(n_past)

        eval_a_outs.append(np.column_stack([ev_te_a, ev_cnt_a]))
        eval_b_outs.append(np.column_stack([ev_te_b, ev_cnt_b]))

    return tr_a, tr_b, eval_a_outs, eval_b_outs


def run_pipeline():
    log("=== STARTING COMPLEMENTARY SIGNALS & 5-REGIME TOURNAMENT ===")
    t_start = time.time()

    # Load canonical parsed data
    log("Loading parsed data...")
    df_train = pickle.load(open(ROOT / "data" / "splits" / "t98_parsed_train.pkl", "rb"))
    df_test = pickle.load(open(ROOT / "data" / "splits" / "t98_parsed_test.pkl", "rb"))
    s4_df = pd.read_csv(ROOT / "submissions" / "SUBMISSION_4_TODAY_RECOMMENDED.csv")
    s4_preds = (s4_df["condition"] == "used").astype(int).values
    tx = np.load(ROOT / "data" / "splits" / "full70k_text_cache.npz")

    y_train = df_train["y"].values.astype(int)
    outer_folds = df_train["outer_fold"].values.astype(int)
    n_train = len(df_train)
    n_test = len(df_test)
    prior = float(y_train.mean())

    log(f"Train rows: {n_train} | Test rows: {n_test} | Base prior: {prior:.4f}")

    # 1. Specialized Text Negation Model (Full Corpus TF-IDF + Logistic Regression)
    log("Fitting specialized Negation & Condition text specialist...")
    corpus_train = (df_train["title"].fillna("") + " " +
                    df_train["attr_text"].fillna("") + " " +
                    df_train["warranty_str"].fillna("") + " " +
                    df_train["subtitle"].fillna("")).str.lower()
    corpus_test = (df_test["title"].fillna("") + " " +
                   df_test["attr_text"].fillna("") + " " +
                   df_test["warranty_str"].fillna("") + " " +
                   df_test["subtitle"].fillna("")).str.lower()

    vec = TfidfVectorizer(ngram_range=(1, 3), min_df=5, max_features=30000, sublinear_tf=True)
    X_tr_text = vec.fit_transform(corpus_train)
    X_te_text = vec.transform(corpus_test)

    # 5-fold seller-grouped OOF for text negation
    p_text_neg_oof = np.zeros(n_train, dtype=np.float32)
    p_text_neg_test = np.zeros(n_test, dtype=np.float32)

    for k in range(5):
        tr_m = outer_folds != k
        va_m = outer_folds == k
        lr = LogisticRegression(C=1.5, max_iter=300, solver="lbfgs", random_state=42 + k)
        lr.fit(X_tr_text[tr_m], y_train[tr_m])
        p_text_neg_oof[va_m] = lr.predict_proba(X_tr_text[va_m])[:, 1].astype(np.float32)
        p_text_neg_test += lr.predict_proba(X_te_text)[:, 1].astype(np.float32) / 5.0

    acc_neg = accuracy_score(y_train, (p_text_neg_oof >= 0.5).astype(int))
    ll_neg = log_loss(y_train, p_text_neg_oof)
    log(f"Text Negation Model 5-Fold OOF Acc: {acc_neg:.5f}, Logloss: {ll_neg:.5f}")

    # 2. Identity & Specification Regex Features
    log("Extracting product identity & specification features...")
    regex_tr = extract_regex_features(df_train)
    regex_te = extract_regex_features(df_test)
    keys_regex = list(regex_tr.keys())
    feats_regex_tr = np.column_stack([regex_tr[k] for k in keys_regex])
    feats_regex_te = np.column_stack([regex_te[k] for k in keys_regex])

    # 3. Commercial & Price Features
    log("Extracting commercial, price & quantity features...")
    cat_medians = df_train.groupby("category_id")["price"].median()
    global_median_p = df_train["price"].median()

    def get_price_features(df):
        p = df["price"].values
        bp = df["base_price"].fillna(df["price"]).values
        op = df["orig_price"].fillna(df["price"]).values
        c_med = df["category_id"].map(cat_medians).fillna(global_median_p).values
        rel_p = (p / (c_med + 1.0)).astype(np.float32)
        disc = (p / (op + 1.0)).astype(np.float32)
        has_orig = (df["has_orig_price"].fillna(0).values).astype(np.float32)

        iq = np.log1p(np.maximum(0.0, df["init_qty"].fillna(1.0).values)).astype(np.float32)
        aq = np.log1p(np.maximum(0.0, df["avail_qty"].fillna(1.0).values)).astype(np.float32)
        sq = np.log1p(np.maximum(0.0, df["sold_qty"].fillna(0.0).values)).astype(np.float32)
        sold_r = (df["sold_qty"].fillna(0).values / (df["init_qty"].fillna(1).values + 1.0)).astype(np.float32)

        free_sh = df["free_shipping"].fillna(0).values.astype(np.float32)
        local_pu = df["local_pick_up"].fillna(0).values.astype(np.float32)
        acc_mp = df["accepts_mercadopago"].fillna(0).values.astype(np.float32)
        auto_rel = df["automatic_relist"].fillna(0).values.astype(np.float32)
        off_store = df["has_official_store"].fillna(0).values.astype(np.float32)
        h_vid = df["has_video"].fillna(0).values.astype(np.float32)
        h_cat = df["has_catalog"].fillna(0).values.astype(np.float32)

        n_pics = df["num_pics"].fillna(0).values.astype(np.float32)
        n_attrs = df["num_attrs"].fillna(0).values.astype(np.float32)
        n_vars = df["num_vars"].fillna(0).values.astype(np.float32)
        n_tags = df["num_tags"].fillna(0).values.astype(np.float32)

        start_t = df["start_time"].fillna(0).values.astype(np.float64)
        stop_t = df["stop_time"].fillna(0).values.astype(np.float64)
        duration_days = np.clip((stop_t - start_t) / (1000.0 * 86400.0), -365.0, 3650.0).astype(np.float32)

        # Price x Regex interactions
        price_x_part = (rel_p * regex_tr["is_part"] if len(df) == n_train else rel_p * regex_te["is_part"]).astype(np.float32)
        price_x_defect = (rel_p * regex_tr["has_defect"] if len(df) == n_train else rel_p * regex_te["has_defect"]).astype(np.float32)
        price_x_vintage = (rel_p * regex_tr["has_vintage"] if len(df) == n_train else rel_p * regex_te["has_vintage"]).astype(np.float32)

        return np.column_stack([
            np.log1p(p), np.log1p(bp), rel_p, disc, has_orig,
            iq, aq, sq, sold_r,
            free_sh, local_pu, acc_mp, auto_rel, off_store, h_vid, h_cat,
            n_pics, n_attrs, n_vars, n_tags, duration_days,
            price_x_part, price_x_defect, price_x_vintage
        ])

    feats_comm_tr = get_price_features(df_train)
    feats_comm_te = get_price_features(df_test)

    # Product key for Cross-Seller Retrieval
    keys_prod_tr = (df_train["category_id"].astype(str) + "_" + df_train["title"].str.split().str[0].fillna("")).values
    keys_prod_te = (df_test["category_id"].astype(str) + "_" + df_test["title"].str.split().str[0].fillna("")).values

    # Base Text & SVD from frozen cache
    ptx1_tr = tx["ptx1_oof"][:, None]
    ptx2_tr = tx["ptx2_oof"][:, None]
    svd16_tr = tx["svd_train"][:, :16]

    ptx1_te = tx["ptx1_test"][:, None]
    ptx2_te = tx["ptx2_test"][:, None]
    svd16_te = tx["svd_test"][:, :16]

    # Models to compare
    model_names = [
        "M0_Baseline",
        "M1_Enriched_Attr",
        "M2_Negation_Text",
        "M3_CrossSeller_Retrieval",
        "M4A_Full_PolicyA",
        "M4B_Full_PolicyB",
        "M5_CatBoost_Native",
    ]

    oof_preds = {name: np.zeros(n_train, dtype=np.float32) for name in model_names}
    test_preds = {name: np.zeros(n_test, dtype=np.float32) for name in model_names}

    log("Beginning 5-fold outer seller-group cross-validation...")

    for k in range(5):
        t_fold = time.time()
        tr_idx = np.flatnonzero(outer_folds != k)
        va_idx = np.flatnonzero(outer_folds == k)

        df_tr_k = df_train.iloc[tr_idx]
        df_va_k = df_train.iloc[va_idx]
        y_tr_k = y_train[tr_idx]
        s_tr_k = df_train["seller_id"].iloc[tr_idx].values

        # 1. Clean Fold-Level Target Encodings
        te_cat_tr, [te_cat_va, te_cat_te] = compute_fold_te(df_tr_k["category_id"].values, y_tr_k, s_tr_k, [df_va_k["category_id"].values, df_test["category_id"].values])
        te_list_tr, [te_list_va, te_list_te] = compute_fold_te(df_tr_k["listing_type_id"].fillna("M").values, y_tr_k, s_tr_k, [df_va_k["listing_type_id"].fillna("M").values, df_test["listing_type_id"].fillna("M").values])
        te_brand_tr, [te_brand_va, te_brand_te] = compute_fold_te(df_tr_k["brand_val"].fillna("M").values, y_tr_k, s_tr_k, [df_va_k["brand_val"].fillna("M").values, df_test["brand_val"].fillna("M").values])
        te_model_tr, [te_model_va, te_model_te] = compute_fold_te(df_tr_k["model_val"].fillna("M").values, y_tr_k, s_tr_k, [df_va_k["model_val"].fillna("M").values, df_test["model_val"].fillna("M").values])
        te_cond_tr, [te_cond_va, te_cond_te] = compute_fold_te(df_tr_k["item_cond_attr"].fillna("M").values, y_tr_k, s_tr_k, [df_va_k["item_cond_attr"].fillna("M").values, df_test["item_cond_attr"].fillna("M").values])
        te_state_tr, [te_state_va, te_state_te] = compute_fold_te(df_tr_k["state_id"].fillna("M").values, y_tr_k, s_tr_k, [df_va_k["state_id"].fillna("M").values, df_test["state_id"].fillna("M").values])

        feats_cat_tr = np.column_stack([te_cat_tr, te_list_tr, te_brand_tr, te_model_tr, te_cond_tr, te_state_tr])
        feats_cat_va = np.column_stack([te_cat_va, te_list_va, te_brand_va, te_model_va, te_cond_va, te_state_va])
        feats_cat_te = np.column_stack([te_cat_te, te_list_te, te_brand_te, te_model_te, te_cond_te, te_state_te])

        # 2. Seller History under Policy A (Causal) and Policy B (Bidirectional Batch)
        tr_a, tr_b, [va_a, te_a], [va_b, te_b] = compute_seller_history_policies(df_tr_k, [df_va_k, df_test])

        # 3. Cross-Seller Retrieval Features
        ret_tr, [ret_va, ret_te] = compute_cross_seller_features(
            keys_prod_tr[tr_idx], y_tr_k, s_tr_k, df_tr_k["price"].values,
            [keys_prod_tr[va_idx], keys_prod_te], [df_va_k["price"].values, df_test["price"].values]
        )

        # Static slices for this fold
        comm_tr, comm_va, comm_te = feats_comm_tr[tr_idx], feats_comm_tr[va_idx], feats_comm_te
        reg_tr, reg_va, reg_te = feats_regex_tr[tr_idx], feats_regex_tr[va_idx], feats_regex_te
        ptx1_tr_k, ptx1_va_k, ptx1_te_k = ptx1_tr[tr_idx], ptx1_tr[va_idx], ptx1_te
        ptx2_tr_k, ptx2_va_k, ptx2_te_k = ptx2_tr[tr_idx], ptx2_tr[va_idx], ptx2_te
        svd_tr_k, svd_va_k, svd_te_k = svd16_tr[tr_idx], svd16_tr[va_idx], svd16_te
        p_neg_tr_k, p_neg_va_k, p_neg_te_k = p_text_neg_oof[tr_idx, None], p_text_neg_oof[va_idx, None], p_text_neg_test[:, None]

        # --- Construct Feature Matrices ---
        # M0_Baseline: Text OOF + Causal Seller TE + SVD + Category & Listing TE
        X_m0_tr = np.column_stack([ptx1_tr_k, ptx2_tr_k, tr_a, svd_tr_k, te_cat_tr, te_list_tr])
        X_m0_va = np.column_stack([ptx1_va_k, ptx2_va_k, va_a, svd_va_k, te_cat_va, te_list_va])
        X_m0_te = np.column_stack([ptx1_te_k, ptx2_te_k, te_a, svd_te_k, te_cat_te, te_list_te])

        # M1_Enriched_Attr: M0 + Structured attributes, Commercial & Price, Regex Specs
        X_m1_tr = np.column_stack([X_m0_tr, feats_cat_tr, comm_tr, reg_tr])
        X_m1_va = np.column_stack([X_m0_va, feats_cat_va, comm_va, reg_va])
        X_m1_te = np.column_stack([X_m0_te, feats_cat_te, comm_te, reg_te])

        # M2_Negation_Text: M1 + p_text_negation
        X_m2_tr = np.column_stack([X_m1_tr, p_neg_tr_k])
        X_m2_va = np.column_stack([X_m1_va, p_neg_va_k])
        X_m2_te = np.column_stack([X_m1_te, p_neg_te_k])

        # M3_CrossSeller_Retrieval: M2 + Cross-seller retrieval features
        X_m3_tr = np.column_stack([X_m2_tr, ret_tr])
        X_m3_va = np.column_stack([X_m2_va, ret_va])
        X_m3_te = np.column_stack([X_m2_te, ret_te])

        # M4A_Full_PolicyA: M3 with Policy A (Causal)
        X_m4a_tr, X_m4a_va, X_m4a_te = X_m3_tr, X_m3_va, X_m3_te

        # M4B_Full_PolicyB: M3 with Policy B (Bidirectional Batch seller history)
        X_m4b_tr = np.column_stack([ptx1_tr_k, ptx2_tr_k, tr_b, svd_tr_k, te_cat_tr, te_list_tr, feats_cat_tr, comm_tr, reg_tr, p_neg_tr_k, ret_tr])
        X_m4b_va = np.column_stack([ptx1_va_k, ptx2_va_k, va_b, svd_va_k, te_cat_va, te_list_va, feats_cat_va, comm_va, reg_va, p_neg_va_k, ret_va])
        X_m4b_te = np.column_stack([ptx1_te_k, ptx2_te_k, te_b, svd_te_k, te_cat_te, te_list_te, feats_cat_te, comm_te, reg_te, p_neg_te_k, ret_te])

        fold_matrices = {
            "M0_Baseline": (X_m0_tr, X_m0_va, X_m0_te),
            "M1_Enriched_Attr": (X_m1_tr, X_m1_va, X_m1_te),
            "M2_Negation_Text": (X_m2_tr, X_m2_va, X_m2_te),
            "M3_CrossSeller_Retrieval": (X_m3_tr, X_m3_va, X_m3_te),
            "M4A_Full_PolicyA": (X_m4a_tr, X_m4a_va, X_m4a_te),
            "M4B_Full_PolicyB": (X_m4b_tr, X_m4b_va, X_m4b_te),
        }

        # Train LightGBM models
        for m_name, (X_tr, X_va, X_te) in fold_matrices.items():
            lgb = LGBMClassifier(
                n_estimators=180,
                learning_rate=0.06,
                num_leaves=45,
                colsample_bytree=0.75,
                subsample=0.8,
                min_child_samples=25,
                random_state=42 + k,
                n_jobs=-1,
                verbose=-1,
            )
            lgb.fit(X_tr, y_tr_k)
            oof_preds[m_name][va_idx] = lgb.predict_proba(X_va)[:, 1]
            test_preds[m_name] += lgb.predict_proba(X_te)[:, 1] / 5.0

        # Train CatBoost Native (M5)
        cb = CatBoostClassifier(
            iterations=250,
            learning_rate=0.06,
            depth=6,
            random_seed=42 + k,
            thread_count=-1,
            verbose=False,
        )
        cb.fit(X_m4b_tr, y_tr_k)
        oof_preds["M5_CatBoost_Native"][va_idx] = cb.predict_proba(X_m4b_va)[:, 1]
        test_preds["M5_CatBoost_Native"] += cb.predict_proba(X_m4b_te)[:, 1] / 5.0

        acc_m0 = accuracy_score(y_train[va_idx], (oof_preds["M0_Baseline"][va_idx] >= 0.5).astype(int))
        acc_m4b = accuracy_score(y_train[va_idx], (oof_preds["M4B_Full_PolicyB"][va_idx] >= 0.5).astype(int))
        log(f"Fold {k} done in {time.time() - t_fold:.2f}s | Val Acc M0: {acc_m0:.5f}, M4B: {acc_m4b:.5f}")

    # M6: Stacking Meta-Learner (Cross-Fitted)
    log("Fitting M6_Meta_Stacker...")
    stack_cols = ["M0_Baseline", "M1_Enriched_Attr", "M2_Negation_Text", "M3_CrossSeller_Retrieval", "M4A_Full_PolicyA", "M4B_Full_PolicyB", "M5_CatBoost_Native"]
    S_oof = np.column_stack([oof_preds[c] for c in stack_cols])
    S_test = np.column_stack([test_preds[c] for c in stack_cols])

    oof_preds["M6_Meta_Stacker"] = np.zeros(n_train, dtype=np.float32)
    test_preds["M6_Meta_Stacker"] = np.zeros(n_test, dtype=np.float32)

    for k in range(5):
        tr_idx = np.flatnonzero(outer_folds != k)
        va_idx = np.flatnonzero(outer_folds == k)
        meta = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200, random_state=42 + k)
        meta.fit(S_oof[tr_idx], y_train[tr_idx])
        oof_preds["M6_Meta_Stacker"][va_idx] = meta.predict_proba(S_oof[va_idx])[:, 1]
        test_preds["M6_Meta_Stacker"] += meta.predict_proba(S_test)[:, 1] / 5.0

    # --- OOF Evaluation & Paired Metrics vs M0_Baseline ---
    log("\n=== 5-FOLD OUTER SELLER-GROUP OOF RESULTS (70,000 ROWS) ===")
    base_pred_bin = (oof_preds["M0_Baseline"] >= 0.5).astype(int)
    base_hits = int((base_pred_bin == y_train).sum())

    all_models = stack_cols + ["M6_Meta_Stacker"]
    results_summary = {}

    for name in all_models:
        pred_bin = (oof_preds[name] >= 0.5).astype(int)
        acc = accuracy_score(y_train, pred_bin)
        ll = log_loss(y_train, oof_preds[name])
        brier = brier_score_loss(y_train, oof_preds[name])
        hits = int((pred_bin == y_train).sum())
        delta_hits = hits - base_hits

        fixes = int(((pred_bin == y_train) & (base_pred_bin != y_train)).sum())
        breaks = int(((pred_bin != y_train) & (base_pred_bin == y_train)).sum())
        flips = int((pred_bin != base_pred_bin).sum())

        results_summary[name] = {
            "accuracy": round(float(acc), 6),
            "logloss": round(float(ll), 6),
            "brier": round(float(brier), 6),
            "hits_70k": hits,
            "delta_hits_vs_base": delta_hits,
            "flips_vs_base": flips,
            "fixes": fixes,
            "breaks": breaks,
        }
        log(f"{name:25s} | Acc: {acc:.5f} | LogLoss: {ll:.5f} | Hits: {hits:5d} | Delta vs Base: {delta_hits:+4d} (Fixes: {fixes:4d}, Breaks: {breaks:4d})")

    # --- 5 Test Regimes Evaluation on Test (30,000 Rows) ---
    log("\n=== EVALUATION ON THE 5 REAL TEST REGIMES (30,000 ROWS) ===")
    tr_sellers_set = set(df_train["seller_id"].unique())
    tr_stats = df_train.groupby("seller_id")["start_time"].agg(["min", "max"])
    tr_all_times = df_train.groupby("seller_id")["start_time"].apply(set).to_dict()

    is_seen = df_test["seller_id"].isin(tr_sellers_set).values
    cold_mask = ~is_seen

    t_te = df_test["start_time"].values
    t_min = df_test["seller_id"].map(tr_stats["min"]).values
    t_max = df_test["seller_id"].map(tr_stats["max"]).values

    def has_tie(row):
        s = row["seller_id"]
        return (s in tr_all_times) and (row["start_time"] in tr_all_times[s])

    ties_mask = df_test.apply(has_tie, axis=1).values
    bracketed_mask = is_seen & (~ties_mask) & (t_te > t_min) & (t_te < t_max)
    before_mask = is_seen & (~ties_mask) & (t_te < t_min)
    after_mask = is_seen & (~ties_mask) & (t_te > t_max)

    regimes_dict = {
        "Cold_New_Seller": (cold_mask, 8513),
        "Strictly_Bracketed": (bracketed_mask, 10140),
        "Same_Timestamp_Ties": (ties_mask, 5108),
        "Before": (before_mask, 3118),
        "After": (after_mask, 3121),
    }

    test_regimes_summary = {}
    for r_name, (r_mask, expected_n) in regimes_dict.items():
        actual_n = int(r_mask.sum())
        assert actual_n == expected_n, f"Mismatch in {r_name}: got {actual_n}, expected {expected_n}"
        log(f"Régimen: {r_name:22s} | Filas: {actual_n:5d} ({actual_n/n_test:.2%})")

    # Comparisons against S4 Anchor
    log("\n--- Comparison Against Anchor S4 (SUBMISSION_4_TODAY_RECOMMENDED.csv) ---")
    s4_used_rate = float(s4_preds.mean())
    log(f"S4 used rate: {s4_used_rate:.4f} (19,899 public hits, score 0.94766)")

    model_test_analysis = {}
    for name in all_models:
        pred_bin = (test_preds[name] >= 0.5).astype(int)
        used_rate = float(pred_bin.mean())
        flips_total = int((pred_bin != s4_preds).sum())
        flips_to_used = int(((pred_bin == 1) & (s4_preds == 0)).sum())
        flips_to_new = int(((pred_bin == 0) & (s4_preds == 1)).sum())

        regime_breakdown = {}
        for r_name, (r_mask, _) in regimes_dict.items():
            r_flips = int((pred_bin[r_mask] != s4_preds[r_mask]).sum())
            r_used = int(((pred_bin[r_mask] == 1) & (s4_preds[r_mask] == 0)).sum())
            r_new = int(((pred_bin[r_mask] == 0) & (s4_preds[r_mask] == 1)).sum())
            regime_breakdown[r_name] = {
                "flips": r_flips,
                "flips_to_used": r_used,
                "flips_to_new": r_new,
                "share_of_flips": round(r_flips / max(1, flips_total), 4),
            }

        model_test_analysis[name] = {
            "used_rate": round(used_rate, 4),
            "flips_vs_s4_total": flips_total,
            "flips_to_used": flips_to_used,
            "flips_to_new": flips_to_new,
            "regimes_breakdown": regime_breakdown,
        }
        log(f"{name:25s} | Used Rate: {used_rate:.4f} | Flips vs S4: {flips_total:5d} (U:{flips_to_used:4d}, N:{flips_to_new:4d})")
        for r_name, rb in regime_breakdown.items():
            log(f"   -> {r_name:20s}: {rb['flips']:4d} flips ({rb['share_of_flips']:.1%})")

    # Policy A vs Policy B Comparison Analysis
    log("\n--- Comparison: Policy A (Causal Past-Only) vs Policy B (Bidirectional Batch) ---")
    pol_a_bin = (test_preds["M4A_Full_PolicyA"] >= 0.5).astype(int)
    pol_b_bin = (test_preds["M4B_Full_PolicyB"] >= 0.5).astype(int)
    diff_ab = int((pol_a_bin != pol_b_bin).sum())
    log(f"Total test disagreements between Policy A and Policy B: {diff_ab} / 30,000")
    for r_name, (r_mask, _) in regimes_dict.items():
        diff_r = int((pol_a_bin[r_mask] != pol_b_bin[r_mask]).sum())
        log(f"   -> {r_name:20s}: {diff_r:4d} disagreements / {r_mask.sum():5d} rows")

    # Save Output Predictions & Report
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Private evaluation pairs CSV
    df_pairs_private = pd.DataFrame({
        "id": df_train["id"].values,
        "y": y_train,
        "outer_fold": outer_folds,
        "m0_base_prob": np.round(oof_preds["M0_Baseline"], 5),
        "m1_attr_prob": np.round(oof_preds["M1_Enriched_Attr"], 5),
        "m2_neg_prob": np.round(oof_preds["M2_Negation_Text"], 5),
        "m3_ret_prob": np.round(oof_preds["M3_CrossSeller_Retrieval"], 5),
        "m4a_causal_prob": np.round(oof_preds["M4A_Full_PolicyA"], 5),
        "m4b_batch_prob": np.round(oof_preds["M4B_Full_PolicyB"], 5),
        "m5_catboost_prob": np.round(oof_preds["M5_CatBoost_Native"], 5),
        "m6_metastacker_prob": np.round(oof_preds["M6_Meta_Stacker"], 5),
    })
    pairs_path = out_dir / "complementary_signals_5regimes_pairs_private.csv"
    df_pairs_private.to_csv(pairs_path, index=False)
    log(f"Saved private evaluation pairs to {pairs_path.name}")

    # 2. Test candidate predictions CSVs (saved in results/ for audit, NOT in submissions/ to keep submissions/ clean)
    candidates_to_save = ["M4A_Full_PolicyA", "M4B_Full_PolicyB", "M6_Meta_Stacker"]
    test_csv_hashes = {}
    for cand_name in candidates_to_save:
        c_bin = (test_preds[cand_name] >= 0.5).astype(int)
        c_str = ["used" if b == 1 else "new" for b in c_bin]
        df_sub = pd.DataFrame({"ID": df_test["id"].values, "condition": c_str})
        cand_csv_path = out_dir / f"CANDIDATE_{cand_name}.csv"
        df_sub.to_csv(cand_csv_path, index=False)
        c_sha = sha256_bytes(cand_csv_path.read_bytes())
        test_csv_hashes[cand_name] = {
            "path": str(cand_csv_path.relative_to(ROOT)),
            "sha256": c_sha,
            "flips_vs_s4": int((c_bin != s4_preds).sum()),
            "used_rate": round(float(c_bin.mean()), 4),
        }
        log(f"Candidate {cand_name} saved: {cand_csv_path.name} | SHA: {c_sha[:16]}... | Flips vs S4: {test_csv_hashes[cand_name]['flips_vs_s4']}")

    # 3. JSON summary report
    report = {
        "generator": "scripts/train_complementary_signals_and_5regimes.py",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "DEVELOPMENT_ONLY",
        "public_probability_claims": "NO_IDENTIFIABLE",
        "splits_sha256": "84643dac738929037d0639cfc177a2af7ce34c4d424ba5537cfeafe85153173f",
        "train_cache_sha256": "f97cc4f04de13c243b007398160ec57470cde1346c2358fd02facc1bdbffbf3b",
        "test_cache_sha256": "170383ff930d565a8efd1812865d404b39c049e787f16218d2a203df43ce55a3",
        "s4_anchor_sha256": "bb73f1ff1b5fdc76fd8880d4cc146adf46a0558138aec4546e17508cfcf5183f",
        "oof_summary_70k": results_summary,
        "test_regimes_summary": {
            "regimes_counts": {k: v[1] for k, v in regimes_dict.items()},
            "model_analysis": model_test_analysis,
            "policy_a_vs_b_disagreements": {
                "total": diff_ab,
                "by_regime": {r_name: int((pol_a_bin[r_mask] != pol_b_bin[r_mask]).sum()) for r_name, (r_mask, _) in regimes_dict.items()}
            }
        },
        "candidate_test_hashes": test_csv_hashes,
        "total_elapsed_seconds": round(time.time() - t_start, 2),
    }

    report_path = out_dir / "complementary_signals_5regimes_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"Saved JSON report to {report_path.name}")
    log(f"=== COMPLETED IN {time.time() - t_start:.2f}s ===")


if __name__ == "__main__":
    run_pipeline()
