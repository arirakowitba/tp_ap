"""High-Capacity Tree & Tabular Specialist Model (AP-2026-Q-2).

Specialist Module for Dual-Regime Frozen Holdout Optimization.
Strict Zero-Leakage Fold-Local Feature Engineering on 56,000 Fit Pool.

Incorporates:
1. Lexical condition cues in title ('como nuevo', 'impecable', 'excelente estado', 'caja cerrada', 'nuevo sellado').
2. Warranty cues ('de fabrica', 'del fabricante', 'oficial', 'sin garantia').
3. Category relative price features: log(price / (cat_median_price + 1)), sold_quantity / (initial_quantity + 1).
4. Strictly fold-local feature engineering on the 56,000 fit pool using frozen partition indices.
5. Evaluation on the 14,000 frozen dual-regime holdout (10,027 seen / 3,973 unseen).
6. Saving out-of-fold, holdout, and test predictions to results/dual_regime_20261109/subagent_trees_predictions.npz.
"""

import os
import sys
import gc
import json
import time
import hashlib
import unicodedata
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
from catboost import CatBoostClassifier

ROOT = Path(".")
DATA_DIR = ROOT / "data/ap_2026_q_2/raw"
OUT_DIR = ROOT / "results/dual_regime_20261109"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def strip_accents(s):
    if not isinstance(s, str):
        return ""
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def extract_warranty_months(text):
    m = re.search(r"(\d+)\s*(?:mes|meses)", text)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+)\s*(?:ano|anos|año|años)", text)
    if m:
        return float(m.group(1)) * 12.0
    return 0.0


def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class HighCapacityTreeFeatureExtractor:
    """Strictly fold-local high-capacity tabular & text feature extractor."""

    def __init__(self):
        self.cat_map = {}
        self.ship_map = {}
        self.buy_map = {}
        self.list_map = {}
        self.cat_median_price = {}
        self.cat_mean_price = {}
        self.global_median_price = 0.0
        self.global_mean_price = 0.0
        self.cat_te = {}
        self.seller_te = {}
        self.list_te = {}
        self.global_prior = 0.5
        self.vec_word = None
        self.svd_word = None
        self.vec_char = None
        self.svd_char = None

    def fit(self, df_tr, y_tr):
        self.global_prior = float(y_tr.mean())

        # Categorical mappings
        for col, attr in [("category_id", "cat_map"), ("buying_mode", "buy_map"), ("listing_type_id", "list_map")]:
            vals = df_tr[col].dropna().unique()
            setattr(self, attr, {v: i for i, v in enumerate(vals)})

        ship_modes = df_tr["shipping"].apply(lambda s: s.get("mode") if isinstance(s, dict) else "unknown").unique()
        self.ship_map = {v: i for i, v in enumerate(ship_modes)}

        # Category price statistics
        prices = df_tr["price"].fillna(0.0).astype(np.float32)
        self.global_median_price = float(prices.median())
        self.global_mean_price = float(prices.mean())
        self.cat_median_price = df_tr.groupby("category_id")["price"].median().to_dict()
        self.cat_mean_price = df_tr.groupby("category_id")["price"].mean().to_dict()

        # Fold-local target encodings with empirical Bayes m-estimates
        # 1. Category TE (m=15)
        cat_counts = df_tr.groupby("category_id").size()
        cat_means = df_tr.groupby("category_id")["target"].mean()
        self.cat_te = ((cat_counts * cat_means + 15.0 * self.global_prior) / (cat_counts + 15.0)).to_dict()

        # 2. Seller TE (m=25)
        seller_counts = df_tr.groupby("seller_id").size()
        seller_means = df_tr.groupby("seller_id")["target"].mean()
        self.seller_te = ((seller_counts * seller_means + 25.0 * self.global_prior) / (seller_counts + 25.0)).to_dict()

        # 3. Listing Type TE (m=10)
        list_counts = df_tr.groupby("listing_type_id").size()
        list_means = df_tr.groupby("listing_type_id")["target"].mean()
        self.list_te = ((list_counts * list_means + 10.0 * self.global_prior) / (list_counts + 10.0)).to_dict()

        # Fold-local Text TF-IDF + SVD
        titles = df_tr["title"].fillna("").astype(str).str.lower()
        self.vec_word = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True)
        self.svd_word = TruncatedSVD(n_components=48, random_state=42)
        w_mat = self.vec_word.fit_transform(titles)
        self.svd_word.fit(w_mat)

        self.vec_char = TfidfVectorizer(max_features=25000, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.svd_char = TruncatedSVD(n_components=24, random_state=42)
        c_mat = self.vec_char.fit_transform(titles)
        self.svd_char.fit(c_mat)

        return self

    def transform(self, df):
        feats = pd.DataFrame(index=df.index)

        # -------------------------------------------------------------
        # 1. Lexical condition cues in title
        # -------------------------------------------------------------
        t_raw = df["title"].fillna("").astype(str).str.lower()
        t_clean = t_raw.apply(strip_accents)

        # Exact prompt required cues
        # 'como nuevo'
        c_como_nuevo = (
            t_clean.str.contains(r"como nuev[oa]", regex=True)
            | t_raw.str.contains("como nuevo", regex=False)
            | t_raw.str.contains("como nueva", regex=False)
        ).astype(np.int8)

        # 'impecable'
        c_impecable = (
            t_clean.str.contains(r"impecable", regex=True)
            | t_raw.str.contains("impecable", regex=False)
        ).astype(np.int8)

        # 'excelente estado'
        c_excelente_estado = (
            t_clean.str.contains(r"(?:excelente|muy buen|perfecto|optimo) estado", regex=True)
            | t_raw.str.contains("excelente estado", regex=False)
            | t_raw.str.contains("muy buen estado", regex=False)
        ).astype(np.int8)

        # 'caja cerrada'
        c_caja_cerrada = (
            t_clean.str.contains(r"caja cerrada", regex=True)
            | t_raw.str.contains("caja cerrada", regex=False)
        ).astype(np.int8)

        # 'nuevo sellado'
        c_nuevo_sellado = (
            t_clean.str.contains(r"nuev[oa] sellad[oa]|sellad[oa] en caja", regex=True)
            | t_raw.str.contains("nuevo sellado", regex=False)
            | t_raw.str.contains("nueva sellada", regex=False)
        ).astype(np.int8)

        # Extended high-signal lexical cues
        c_usado = (
            t_clean.str.contains(r"\busad[oa]s?\b", regex=True)
            | t_raw.str.contains("usado", regex=False)
        ).astype(np.int8)

        c_sin_uso = t_clean.str.contains(r"sin uso|a estrenar|0km", regex=True).astype(np.int8)
        c_detalle_repuesto = t_clean.str.contains(r"repuesto|no funciona|para reparar|con detalle|detalles", regex=True).astype(np.int8)

        feats["cue_como_nuevo"] = c_como_nuevo
        feats["cue_impecable"] = c_impecable
        feats["cue_excelente_estado"] = c_excelente_estado
        feats["cue_caja_cerrada"] = c_caja_cerrada
        feats["cue_nuevo_sellado"] = c_nuevo_sellado
        feats["cue_usado"] = c_usado
        feats["cue_sin_uso"] = c_sin_uso
        feats["cue_detalle_repuesto"] = c_detalle_repuesto

        # Cue aggregates
        cue_used_sum = c_como_nuevo + c_impecable + c_excelente_estado + c_usado + c_detalle_repuesto
        cue_new_sum = c_caja_cerrada + c_nuevo_sellado + c_sin_uso
        feats["cue_total_used"] = cue_used_sum.astype(np.int8)
        feats["cue_total_new"] = cue_new_sum.astype(np.int8)
        feats["cue_net_used"] = (cue_used_sum - cue_new_sum).astype(np.int8)

        # -------------------------------------------------------------
        # 2. Warranty cues
        # -------------------------------------------------------------
        w_raw = df["warranty"].fillna("").astype(str).str.lower()
        w_clean = w_raw.apply(strip_accents)

        # Exact prompt required warranty cues
        # 'de fabrica'
        w_de_fabrica = (
            w_clean.str.contains(r"de fabrica", regex=True)
            | w_raw.str.contains("de fabrica", regex=False)
            | w_raw.str.contains("de fábrica", regex=False)
        ).astype(np.int8)

        # 'del fabricante'
        w_del_fabricante = (
            w_clean.str.contains(r"del fabricante", regex=True)
            | w_raw.str.contains("del fabricante", regex=False)
        ).astype(np.int8)

        # 'oficial'
        w_oficial = (
            w_clean.str.contains(r"oficial", regex=True)
            | w_raw.str.contains("oficial", regex=False)
        ).astype(np.int8)

        # 'sin garantia'
        w_sin_garantia = (
            w_clean.str.contains(r"sin garantia|no tiene garantia|sin ningun tipo de garantia", regex=True)
            | w_raw.str.contains("sin garantia", regex=False)
            | w_raw.str.contains("sin garantía", regex=False)
        ).astype(np.int8)

        feats["warranty_de_fabrica"] = w_de_fabrica
        feats["warranty_del_fabricante"] = w_del_fabricante
        feats["warranty_oficial"] = w_oficial
        feats["warranty_sin_garantia"] = w_sin_garantia

        # Extended warranty features
        has_warranty = (df["warranty"].notna() & (df["warranty"].astype(str).str.strip() != "")).astype(np.int8)
        feats["has_warranty"] = has_warranty
        feats["warranty_len"] = w_raw.str.len().astype(np.int16)
        feats["warranty_is_seller"] = w_clean.str.contains(r"vendedor|local|comercio", regex=True).astype(np.int8)
        feats["warranty_months"] = w_clean.apply(extract_warranty_months).astype(np.float32)

        # -------------------------------------------------------------
        # 3. Category relative price features & quantities
        # -------------------------------------------------------------
        price = df["price"].fillna(0.0).astype(np.float32)
        feats["price"] = price
        feats["log_price"] = np.log1p(np.maximum(price, 0.0)).astype(np.float32)

        cat_med = df["category_id"].map(self.cat_median_price).fillna(self.global_median_price).astype(np.float32)
        cat_mean = df["category_id"].map(self.cat_mean_price).fillna(self.global_mean_price).astype(np.float32)

        # Exact prompt required: log(price / (cat_median_price + 1))
        feats["log_price_ratio_cat"] = np.log((price + 1.0) / (cat_med + 1.0)).astype(np.float32)
        feats["price_ratio_cat"] = ((price + 1.0) / (cat_med + 1.0)).astype(np.float32)
        feats["price_diff_cat_med"] = (price - cat_med).astype(np.float32)
        feats["price_diff_cat_mean"] = (price - cat_mean).astype(np.float32)

        init_qty = df["initial_quantity"].fillna(1).astype(np.float32)
        sold_qty = df["sold_quantity"].fillna(0).astype(np.float32)
        avail_qty = df["available_quantity"].fillna(0).astype(np.float32)

        feats["initial_quantity"] = init_qty
        feats["sold_quantity"] = sold_qty
        feats["available_quantity"] = avail_qty

        # Exact prompt required: sold_quantity / (initial_quantity + 1)
        feats["sold_ratio_p1"] = (sold_qty / (init_qty + 1.0)).astype(np.float32)
        feats["sold_ratio_eps"] = (sold_qty / (init_qty + 1e-4)).astype(np.float32)
        feats["available_ratio"] = (avail_qty / (init_qty + 1.0)).astype(np.float32)
        feats["has_sold"] = (sold_qty > 0).astype(np.int8)
        feats["is_single_item"] = (init_qty == 1).astype(np.int8)

        # -------------------------------------------------------------
        # 4. Listing metadata & structural features
        # -------------------------------------------------------------
        feats["title_len"] = t_raw.str.len().astype(np.int16)
        feats["title_words"] = t_raw.str.split().apply(len).astype(np.int8)
        feats["num_pictures"] = df["pictures"].apply(lambda p: len(p) if isinstance(p, list) else 0).astype(np.int8)
        feats["accepts_mercadopago"] = df["accepts_mercadopago"].fillna(False).astype(np.int8)
        feats["automatic_relist"] = df["automatic_relist"].fillna(False).astype(np.int8)

        shipping = df["shipping"]
        feats["has_shipping"] = shipping.notna().astype(np.int8)
        feats["free_shipping"] = shipping.apply(lambda s: s.get("free_shipping", False) if isinstance(s, dict) else False).astype(np.int8)
        feats["local_pick_up"] = shipping.apply(lambda s: s.get("local_pick_up", False) if isinstance(s, dict) else False).astype(np.int8)

        # Deterministic categorical mappings (-1 for unseen)
        feats["cat_code"] = df["category_id"].map(self.cat_map).fillna(-1).astype(np.int32)
        feats["buy_code"] = df["buying_mode"].map(self.buy_map).fillna(-1).astype(np.int32)
        feats["list_code"] = df["listing_type_id"].map(self.list_map).fillna(-1).astype(np.int32)

        ship_mode_series = shipping.apply(lambda s: s.get("mode") if isinstance(s, dict) else "unknown")
        feats["ship_code"] = ship_mode_series.map(self.ship_map).fillna(-1).astype(np.int32)

        # Fold-local target encodings
        feats["cat_te"] = df["category_id"].map(self.cat_te).fillna(self.global_prior).astype(np.float32)
        feats["seller_te"] = df["seller_id"].map(self.seller_te).fillna(self.global_prior).astype(np.float32)
        feats["list_te"] = df["listing_type_id"].map(self.list_te).fillna(self.global_prior).astype(np.float32)

        # -------------------------------------------------------------
        # 5. High-resolution SVD text projections
        # -------------------------------------------------------------
        w_mat = self.vec_word.transform(t_raw)
        w_svd = self.svd_word.transform(w_mat).astype(np.float32)
        c_mat = self.vec_char.transform(t_raw)
        c_svd = self.svd_char.transform(c_mat).astype(np.float32)

        dense_matrix = np.hstack([feats.values.astype(np.float32), w_svd, c_svd])
        return dense_matrix


class BaselineFoldFeatureExtractor:
    """Exact replica of the baseline feature extractor for strict pair audit."""

    def __init__(self):
        self.cat_map = {}
        self.ship_map = {}
        self.buy_map = {}
        self.list_map = {}
        self.cat_median_price = {}
        self.global_median_price = 0.0
        self.cat_te = {}
        self.seller_te = {}
        self.global_prior = 0.5
        self.vec_word = None
        self.vec_char = None
        self.svd_word = None
        self.svd_char = None

    def fit(self, df_tr, y_tr):
        self.global_prior = float(y_tr.mean())
        for col, attr in [("category_id", "cat_map"), ("buying_mode", "buy_map"), ("listing_type_id", "list_map")]:
            vals = df_tr[col].dropna().unique()
            setattr(self, attr, {v: i for i, v in enumerate(vals)})

        ship_modes = df_tr["shipping"].apply(lambda s: s.get("mode") if isinstance(s, dict) else "unknown").unique()
        self.ship_map = {v: i for i, v in enumerate(ship_modes)}

        prices = df_tr["price"].fillna(0.0).astype(np.float32)
        self.global_median_price = float(prices.median())
        self.cat_median_price = df_tr.groupby("category_id")["price"].median().to_dict()

        cat_counts = df_tr.groupby("category_id").size()
        cat_means = df_tr.groupby("category_id")["target"].mean()
        self.cat_te = ((cat_counts * cat_means + 15.0 * self.global_prior) / (cat_counts + 15.0)).to_dict()

        seller_counts = df_tr.groupby("seller_id").size()
        seller_means = df_tr.groupby("seller_id")["target"].mean()
        self.seller_te = ((seller_counts * seller_means + 25.0 * self.global_prior) / (seller_counts + 25.0)).to_dict()

        titles = df_tr["title"].fillna("").astype(str).str.lower()
        self.vec_word = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), sublinear_tf=True)
        self.svd_word = TruncatedSVD(n_components=32, random_state=42)
        w_mat = self.vec_word.fit_transform(titles)
        self.svd_word.fit(w_mat)

        self.vec_char = TfidfVectorizer(max_features=20000, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.svd_char = TruncatedSVD(n_components=16, random_state=42)
        c_mat = self.vec_char.fit_transform(titles)
        self.svd_char.fit(c_mat)
        return self

    def transform(self, df):
        feats = pd.DataFrame(index=df.index)
        price = df["price"].fillna(0.0).astype(np.float32)
        feats["price"] = price
        feats["log_price"] = np.log1p(np.maximum(price, 0.0))

        cat_med = df["category_id"].map(self.cat_median_price).fillna(self.global_median_price).astype(np.float32)
        feats["log_price_ratio_cat"] = np.log((price + 1.0) / (cat_med + 1.0))

        feats["initial_quantity"] = df["initial_quantity"].fillna(1).astype(np.float32)
        feats["sold_quantity"] = df["sold_quantity"].fillna(0).astype(np.float32)
        feats["sold_ratio"] = feats["sold_quantity"] / (feats["initial_quantity"] + 1e-4)
        feats["available_quantity"] = df["available_quantity"].fillna(0).astype(np.float32)

        warranty_str = df["warranty"].fillna("").astype(str).str.lower()
        feats["has_warranty"] = df["warranty"].notna().astype(np.int8)
        feats["warranty_len"] = warranty_str.str.len().astype(np.int16)
        feats["warranty_is_factory"] = warranty_str.str.contains(r"fabrica|fabricante", regex=True).astype(np.int8)
        feats["warranty_is_seller"] = warranty_str.str.contains(r"vendedor|local", regex=True).astype(np.int8)
        feats["warranty_is_official"] = warranty_str.str.contains(r"oficial", regex=True).astype(np.int8)
        feats["warranty_is_none"] = warranty_str.str.contains(r"sin garant|no.*garant", regex=True).astype(np.int8)

        titles = df["title"].fillna("").astype(str).str.lower()
        feats["title_len"] = titles.str.len().astype(np.int16)
        feats["title_words"] = titles.str.split().apply(len).astype(np.int8)
        feats["num_pictures"] = df["pictures"].apply(lambda p: len(p) if isinstance(p, list) else 0).astype(np.int8)
        feats["accepts_mercadopago"] = df["accepts_mercadopago"].astype(np.int8)
        feats["automatic_relist"] = df["automatic_relist"].astype(np.int8)

        feats["cat_code"] = df["category_id"].map(self.cat_map).fillna(-1).astype(np.int32)
        feats["buy_code"] = df["buying_mode"].map(self.buy_map).fillna(-1).astype(np.int32)
        feats["list_code"] = df["listing_type_id"].map(self.list_map).fillna(-1).astype(np.int32)

        ship_mode_series = df["shipping"].apply(lambda s: s.get("mode") if isinstance(s, dict) else "unknown")
        feats["ship_code"] = ship_mode_series.map(self.ship_map).fillna(-1).astype(np.int32)

        feats["cat_te"] = df["category_id"].map(self.cat_te).fillna(self.global_prior).astype(np.float32)
        feats["seller_te"] = df["seller_id"].map(self.seller_te).fillna(self.global_prior).astype(np.float32)

        w_mat = self.vec_word.transform(titles)
        w_svd = self.svd_word.transform(w_mat)
        c_mat = self.vec_char.transform(titles)
        c_svd = self.svd_char.transform(c_mat)

        return np.hstack([feats.values, w_svd, c_svd])


def run_pipeline():
    print("=" * 80)
    print("HIGH-CAPACITY TREE & TABULAR SPECIALIST TRAINING PIPELINE")
    print("=" * 80)

    # 1. Load Data
    print("\n[1/5] Loading Data...")
    df_train = pd.read_json(DATA_DIR / "train_data.jsonlines", lines=True)
    df_test = pd.read_json(DATA_DIR / "test_data.jsonlines", lines=True)

    y_all = (df_train["condition"] == "used").astype(np.int8).values
    df_train["target"] = y_all

    with open(OUT_DIR / "frozen_dual_regime_indices.json", "r") as f:
        idx_data = json.load(f)

    fit_idx = np.array(idx_data["fit_idx"])
    seen_holdout_idx = np.array(idx_data["seen_holdout_idx"])
    unseen_holdout_idx = np.array(idx_data["unseen_holdout_idx"])

    df_fit = df_train.iloc[fit_idx].reset_index(drop=True)
    y_fit = df_fit["target"].values
    sellers_fit = df_fit["seller_id"].values

    df_seen_ho = df_train.iloc[seen_holdout_idx].reset_index(drop=True)
    y_seen_ho = df_seen_ho["target"].values

    df_unseen_ho = df_train.iloc[unseen_holdout_idx].reset_index(drop=True)
    y_unseen_ho = df_unseen_ho["target"].values

    print(f"  Fit pool: {len(df_fit)} items | Seen holdout: {len(df_seen_ho)} | Unseen holdout: {len(df_unseen_ho)} | Test: {len(df_test)}")

    # 2. Honest 5-Fold StratifiedGroupKFold on 56,000 Fit Pool
    print("\n[2/5] Honest 5-Fold Cross-Fitting on 56k Fit Pool (Strict Fold-Local Isolation)...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_base_lgb = np.zeros(len(df_fit), dtype=np.float32)
    oof_hc_lgb = np.zeros(len(df_fit), dtype=np.float32)
    oof_hc_cat = np.zeros(len(df_fit), dtype=np.float32)

    t_cv_start = time.time()
    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(df_fit, y_fit, sellers_fit)):
        f_t0 = time.time()
        print(f"\n  --- Fold {fold + 1}/5 (Train: {len(tr_idx)}, Val: {len(va_idx)}) ---")

        df_tr = df_fit.iloc[tr_idx].reset_index(drop=True)
        y_tr = y_fit[tr_idx]
        df_va = df_fit.iloc[va_idx].reset_index(drop=True)
        y_va = y_fit[va_idx]

        # Verify zero seller leakage between train and val
        assert len(set(df_tr["seller_id"]).intersection(set(df_va["seller_id"]))) == 0, f"Leakage detected in fold {fold+1}!"

        # A. Baseline Feature Extractor
        b_fe = BaselineFoldFeatureExtractor()
        b_fe.fit(df_tr, y_tr)
        X_tr_base = b_fe.transform(df_tr)
        X_va_base = b_fe.transform(df_va)

        # Baseline LGBM
        m_base_lgb = lgb.LGBMClassifier(n_estimators=380, learning_rate=0.06, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        m_base_lgb.fit(X_tr_base, y_tr)
        oof_base_lgb[va_idx] = m_base_lgb.predict_proba(X_va_base)[:, 1]

        # B. High-Capacity Feature Extractor
        hc_fe = HighCapacityTreeFeatureExtractor()
        hc_fe.fit(df_tr, y_tr)
        X_tr_hc = hc_fe.transform(df_tr)
        X_va_hc = hc_fe.transform(df_va)

        # High-Capacity LightGBM
        # Higher capacity: num_leaves=63, max_depth=8, 600 trees, regularized
        m_hc_lgb = lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.035,
            num_leaves=63,
            max_depth=8,
            colsample_bytree=0.80,
            subsample=0.85,
            subsample_freq=1,
            reg_alpha=0.5,
            reg_lambda=1.5,
            min_child_samples=25,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=-1,
        )
        m_hc_lgb.fit(X_tr_hc, y_tr)
        oof_hc_lgb[va_idx] = m_hc_lgb.predict_proba(X_va_hc)[:, 1]

        # High-Capacity CatBoost
        # Higher capacity: depth=7, 650 iterations, learning_rate=0.045
        m_hc_cat = CatBoostClassifier(
            iterations=650,
            learning_rate=0.045,
            depth=7,
            l2_leaf_reg=5.0,
            random_seed=42 + fold,
            verbose=0,
            thread_count=-1,
        )
        m_hc_cat.fit(X_tr_hc, y_tr)
        oof_hc_cat[va_idx] = m_hc_cat.predict_proba(X_va_hc)[:, 1]

        f_acc_base = accuracy_score(y_va, (oof_base_lgb[va_idx] >= 0.5))
        f_acc_lgb = accuracy_score(y_va, (oof_hc_lgb[va_idx] >= 0.5))
        f_acc_cat = accuracy_score(y_va, (oof_hc_cat[va_idx] >= 0.5))
        p_blend_val = 0.5 * oof_hc_lgb[va_idx] + 0.5 * oof_hc_cat[va_idx]
        f_acc_blend = accuracy_score(y_va, (p_blend_val >= 0.5))

        print(f"    Elapsed: {time.time() - f_t0:.1f}s | Base LGB: {f_acc_base:.4f} | HC-LGB: {f_acc_lgb:.4f} | HC-Cat: {f_acc_cat:.4f} | HC-Blend: {f_acc_blend:.4f}")

    print(f"\n  5-Fold CV Completed in {time.time() - t_cv_start:.1f}s.")

    # 3. Analyze OOF Performance
    print("\n[3/5] Honest OOF Evaluation on 56k Fit Pool:")
    oof_hc_blend = 0.5 * oof_hc_lgb + 0.5 * oof_hc_cat

    for name, p in [
        ("Baseline LGBM", oof_base_lgb),
        ("High-Capacity LightGBM", oof_hc_lgb),
        ("High-Capacity CatBoost", oof_hc_cat),
        ("High-Capacity Trees Blend (50/50)", oof_hc_blend),
    ]:
        acc = accuracy_score(y_fit, (p >= 0.5))
        loss = log_loss(y_fit, p)
        auc = roc_auc_score(y_fit, p)
        fixes = int(np.sum(((oof_base_lgb >= 0.5) != y_fit) & ((p >= 0.5) == y_fit)))
        breaks = int(np.sum(((oof_base_lgb >= 0.5) == y_fit) & ((p >= 0.5) != y_fit)))
        net = fixes - breaks
        print(f"  {name:35s} | Acc: {acc:.5f} | LogLoss: {loss:.5f} | ROC-AUC: {auc:.5f} | Net vs Base: {net:+d} (Fix: {fixes}, Brk: {breaks})")

    # 4. Train Full Models on 56k Fit Pool & Evaluate on Frozen Dual-Regime Holdout
    print("\n[4/5] Full 56k Model Fitting & Frozen Dual-Regime Holdout Evaluation...")
    t_full_start = time.time()

    # Baseline on full 56k
    print("  Fitting Baseline Feature Extractor and Model on 56k...")
    b_fe_full = BaselineFoldFeatureExtractor()
    b_fe_full.fit(df_fit, y_fit)
    X_fit_base = b_fe_full.transform(df_fit)
    X_seen_base = b_fe_full.transform(df_seen_ho)
    X_unseen_base = b_fe_full.transform(df_unseen_ho)
    X_test_base = b_fe_full.transform(df_test)

    m_base_full_lgb = lgb.LGBMClassifier(n_estimators=420, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
    m_base_full_lgb.fit(X_fit_base, y_fit)

    p_seen_base = m_base_full_lgb.predict_proba(X_seen_base)[:, 1]
    p_unseen_base = m_base_full_lgb.predict_proba(X_unseen_base)[:, 1]
    p_test_base = m_base_full_lgb.predict_proba(X_test_base)[:, 1]

    pred_seen_base = (p_seen_base >= 0.5).astype(np.int8)
    pred_unseen_base = (p_unseen_base >= 0.5).astype(np.int8)

    acc_seen_base = accuracy_score(y_seen_ho, pred_seen_base)
    acc_unseen_base = accuracy_score(y_unseen_ho, pred_unseen_base)
    comp_acc_base = (acc_seen_base * 0.7162) + (acc_unseen_base * 0.2838)

    print(f"  Baseline Confirmed:")
    print(f"    Seen Acc:   {acc_seen_base:.5f} ({np.sum(pred_seen_base == y_seen_ho)} / {len(y_seen_ho)})")
    print(f"    Unseen Acc: {acc_unseen_base:.5f} ({np.sum(pred_unseen_base == y_unseen_ho)} / {len(y_unseen_ho)})")
    print(f"    Comp Acc:   {comp_acc_base:.5f}")

    # High-Capacity Feature Extractor on full 56k
    print("\n  Fitting High-Capacity Feature Extractor on 56k...")
    hc_fe_full = HighCapacityTreeFeatureExtractor()
    hc_fe_full.fit(df_fit, y_fit)
    X_fit_hc = hc_fe_full.transform(df_fit)
    X_seen_hc = hc_fe_full.transform(df_seen_ho)
    X_unseen_hc = hc_fe_full.transform(df_unseen_ho)
    X_test_hc = hc_fe_full.transform(df_test)

    # Train High-Capacity LightGBM on full 56k
    print("  Fitting High-Capacity LightGBM on 56k...")
    m_hc_full_lgb = lgb.LGBMClassifier(
        n_estimators=700,
        learning_rate=0.035,
        num_leaves=63,
        max_depth=8,
        colsample_bytree=0.80,
        subsample=0.85,
        subsample_freq=1,
        reg_alpha=0.5,
        reg_lambda=1.5,
        min_child_samples=25,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    m_hc_full_lgb.fit(X_fit_hc, y_fit)

    # Train High-Capacity CatBoost on full 56k
    print("  Fitting High-Capacity CatBoost on 56k...")
    m_hc_full_cat = CatBoostClassifier(
        iterations=750,
        learning_rate=0.045,
        depth=7,
        l2_leaf_reg=5.0,
        random_seed=42,
        verbose=0,
        thread_count=-1,
    )
    m_hc_full_cat.fit(X_fit_hc, y_fit)

    print(f"  Full model fitting done in {time.time() - t_full_start:.1f}s.")

    # Predict Holdout
    p_seen_hc_lgb = m_hc_full_lgb.predict_proba(X_seen_hc)[:, 1]
    p_seen_hc_cat = m_hc_full_cat.predict_proba(X_seen_hc)[:, 1]
    p_seen_hc_blend = 0.5 * p_seen_hc_lgb + 0.5 * p_seen_hc_cat

    p_unseen_hc_lgb = m_hc_full_lgb.predict_proba(X_unseen_hc)[:, 1]
    p_unseen_hc_cat = m_hc_full_cat.predict_proba(X_unseen_hc)[:, 1]
    p_unseen_hc_blend = 0.5 * p_unseen_hc_lgb + 0.5 * p_unseen_hc_cat

    # Predict Test
    p_test_hc_lgb = m_hc_full_lgb.predict_proba(X_test_hc)[:, 1]
    p_test_hc_cat = m_hc_full_cat.predict_proba(X_test_hc)[:, 1]
    p_test_hc_blend = 0.5 * p_test_hc_lgb + 0.5 * p_test_hc_cat

    # Function to evaluate candidate against baseline
    def eval_candidate(name, p_s, p_u):
        pred_s = (p_s >= 0.5).astype(np.int8)
        pred_u = (p_u >= 0.5).astype(np.int8)

        acc_s = accuracy_score(y_seen_ho, pred_s)
        acc_u = accuracy_score(y_unseen_ho, pred_u)
        acc_comp = (acc_s * 0.7162) + (acc_u * 0.2838)

        # Fixes and breaks on seen
        f_s = int(np.sum((pred_seen_base != y_seen_ho) & (pred_s == y_seen_ho)))
        b_s = int(np.sum((pred_seen_base == y_seen_ho) & (pred_s != y_seen_ho)))
        net_s = f_s - b_s

        # Fixes and breaks on unseen
        f_u = int(np.sum((pred_unseen_base != y_unseen_ho) & (pred_u == y_unseen_ho)))
        b_u = int(np.sum((pred_unseen_base == y_unseen_ho) & (pred_u != y_unseen_ho)))
        net_u = f_u - b_u

        tot_f = f_s + f_u
        tot_b = b_s + b_u
        tot_net = tot_f - tot_b

        print(f"\n  --- MODEL: {name} ---")
        print(f"    Seen Holdout (10,027):   Acc: {acc_s:.5f} ({np.sum(pred_s == y_seen_ho)} / {len(y_seen_ho)}) | Fixes: {f_s:3d}, Breaks: {b_s:3d}, Net: {net_s:+4d}")
        print(f"    Unseen Holdout (3,973):  Acc: {acc_u:.5f} ({np.sum(pred_u == y_unseen_ho)} / {len(y_unseen_ho)}) | Fixes: {f_u:3d}, Breaks: {b_u:3d}, Net: {net_u:+4d}")
        print(f"    Composite Mix (71.62%):  Acc: {acc_comp:.5f} (Base: {comp_acc_base:.5f}) | Total Fixes: {tot_f:3d}, Total Breaks: {tot_b:3d}, Total Net: {tot_net:+4d}")
        return {
            "name": name,
            "acc_seen": acc_s,
            "acc_unseen": acc_u,
            "acc_composite": acc_comp,
            "fixes_seen": f_s,
            "breaks_seen": b_s,
            "net_seen": net_s,
            "fixes_unseen": f_u,
            "breaks_unseen": b_u,
            "net_unseen": net_u,
            "total_fixes": tot_f,
            "total_breaks": tot_b,
            "total_net": tot_net,
        }

    res_lgb = eval_candidate("High-Capacity LightGBM", p_seen_hc_lgb, p_unseen_hc_lgb)
    res_cat = eval_candidate("High-Capacity CatBoost", p_seen_hc_cat, p_unseen_hc_cat)
    res_blend = eval_candidate("High-Capacity Trees Blend (50/50)", p_seen_hc_blend, p_unseen_hc_blend)

    # 5. Save out-of-fold and holdout predictions
    print("\n[5/5] Saving Predictions & Audit Report to Disk...")
    npz_path = OUT_DIR / "subagent_trees_predictions.npz"
    np.savez_compressed(
        npz_path,
        oof_lgb=oof_hc_lgb,
        oof_cat=oof_hc_cat,
        oof_blend=oof_hc_blend,
        oof_base_lgb=oof_base_lgb,
        holdout_seen_lgb=p_seen_hc_lgb,
        holdout_seen_cat=p_seen_hc_cat,
        holdout_seen_blend=p_seen_hc_blend,
        holdout_unseen_lgb=p_unseen_hc_lgb,
        holdout_unseen_cat=p_unseen_hc_cat,
        holdout_unseen_blend=p_unseen_hc_blend,
        test_lgb=p_test_hc_lgb,
        test_cat=p_test_hc_cat,
        test_blend=p_test_hc_blend,
        y_fit=y_fit,
        y_seen=y_seen_ho,
        y_unseen=y_unseen_ho,
        base_pred_seen=pred_seen_base,
        base_pred_unseen=pred_unseen_base,
    )
    npz_hash = sha256_file(npz_path)
    print(f"  Saved npz to: {npz_path} ({npz_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  SHA-256: {npz_hash}")

    # Save detailed JSON summary
    report_data = {
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "description": "High-Capacity Tree & Tabular Specialist (LGBM + CatBoost + Blend) with Lexical and Warranty Cues",
        "npz_file": str(npz_path),
        "npz_sha256": npz_hash,
        "baseline_holdout": {
            "seen_acc": float(acc_seen_base),
            "unseen_acc": float(acc_unseen_base),
            "composite_acc": float(comp_acc_base),
        },
        "models": {
            "high_capacity_lgbm": res_lgb,
            "high_capacity_catboost": res_cat,
            "high_capacity_blend": res_blend,
        },
    }

    report_p = OUT_DIR / "subagent_trees_report.json"
    with open(report_p, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"  Saved JSON report to: {report_p}")

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
