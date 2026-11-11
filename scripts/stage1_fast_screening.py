#!/usr/bin/env python3
"""
Stage 1: Fast Screening of Candidates on Fold 1 under Successive Halving
========================================================================

Evaluates candidate enhancements on Fold 1 (14,000 items) against the exact
Podium Engine baseline (13,240 hits on Fold 1 = 0.94571 accuracy).

Candidates Screened:
  - Base: Exact Podium Fold 1 model (13,240 hits).
  - V1 (Text Indicators): Domain regexes (sin uso, 0km, para reparar, etc.).
  - V2 (Commercial Listing Signals): listing_type_id, buying_mode, num_attributes.
  - V3 (Hierarchical Seller Shrinkage): Support-weighted multi-level seller prior.
  - V4 (XGBoost Hist): 500 trees depth 7 on expanded tabular features.
  - V5 (Residual Text Specialist): Sparse linear model targeting Podium residuals.
  - V6 (Enriched Multimodal F150+): Full union of V1+V2+V3+V4 in LightGBM/CatBoost.

Outputs:
  - results/successive_halving_sprint/stage1_screening_report.csv
  - results/successive_halving_sprint/stage1_screening_manifest.json
"""

import sys
import json
import time
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, Ridge
import lightgbm as lgb
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent

def main():
    print("=" * 85)
    print("STAGE 1: FAST SCREENING DE SEÑALES INCREMENTALES (SUCCESSIVE HALVING)")
    print("=" * 85)
    t0 = time.time()

    out_dir = ROOT / "results/successive_halving_sprint"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load raw train data
    raw_path = ROOT / "data/ap_2026_q_2/raw/train_data.jsonlines"
    print("Cargando 70.000 filas de train...", flush=True)
    rows = []
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    df_tr = pd.DataFrame(rows)
    y_all = (df_tr["condition"] == "used").astype(int).values
    n_all = len(y_all)

    # Reconstruct exact StratifiedKFold (Fold 1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    trn_idx, val_idx = next(cv.split(df_tr, y_all))
    print(f"Fold 1 cargado: Train={len(trn_idx)}, Val={len(val_idx)}", flush=True)

    y_train = y_all[trn_idx]
    y_val = y_all[val_idx]

    # Baseline Podium predictions on Fold 1
    podium_oof_path = ROOT / "results/ultimate_podium_engine_23h/oof_predictions_podium_engine.csv"
    df_podium = pd.read_csv(podium_oof_path)
    p_podium_val = df_podium.iloc[val_idx]["oof_blend"].values
    acc_podium_val = accuracy_score(y_val, (p_podium_val >= 0.5).astype(int))
    hits_podium_val = int(round(acc_podium_val * len(val_idx)))
    ll_podium_val = log_loss(y_val, p_podium_val)
    print(f"Referencia Podium Fold 1: Acc = {acc_podium_val:.5f} ({hits_podium_val} hits), LL = {ll_podium_val:.5f}", flush=True)

    # -------------------------------------------------------------------------
    # FEATURE STORE CONSTRUCTION FOR FOLD 1 (STRICTLY FIT ON TRAIN)
    # -------------------------------------------------------------------------
    print("\nConstruyendo variables base e incrementales estrictamente dentro del Fold 1...", flush=True)

    titles = df_tr["title"].fillna("").astype(str).tolist()
    prices = df_tr["price"].fillna(0.0).values.astype(np.float32)
    sellers = df_tr["seller_id"].fillna("MISSING").astype(str).values
    categories = df_tr["category_id"].fillna("MISSING").astype(str).values
    init_qtys = df_tr["initial_quantity"].fillna(1.0).values.astype(np.float32)

    # 1. Base Text Features
    used_kws = ["antiguo", "antigua", "antiguos", "antiguas", "vintage", "retro", "impecable", "impecables", "poco uso", "como nuevo", "despiece", "usado", "usada"]
    new_kws = ["nuevo", "nueva", "nuevos", "nuevas", "sellado", "en caja", "0km", "cero km", "garantia oficial", "de fabrica"]

    def extract_base_text_ind(titles_subset):
        s_lower = [t.lower() for t in titles_subset]
        has_used = np.array([sum(1 for kw in used_kws if kw in t) for t in s_lower], dtype=np.float32)
        has_new = np.array([sum(1 for kw in new_kws if kw in t) for t in s_lower], dtype=np.float32)
        char_len = np.array([len(t) for t in titles_subset], dtype=np.float32)
        word_cnt = np.array([len(t.split()) for t in titles_subset], dtype=np.float32)
        digit_cnt = np.array([sum(c.isdigit() for c in t) for t in titles_subset], dtype=np.float32)
        upper_cnt = np.array([sum(c.isupper() for c in t) for t in titles_subset], dtype=np.float32)
        return np.column_stack([has_used, has_new, char_len, word_cnt, digit_cnt, upper_cnt])

    base_text_trn = extract_base_text_ind([titles[i] for i in trn_idx])
    base_text_val = extract_base_text_ind([titles[i] for i in val_idx])

    # 2. Advanced Domain Regex Indicators (Agente A)
    sin_uso_re = re.compile(r"\b(sin uso|a estrenar|nunca usad[oa]s?|en caja cerrada|sellad[oa]s?|0\s*km|cero\s*km|de fabrica)\b", re.IGNORECASE)
    reacond_re = re.compile(r"\b(reacondicionad[oa]s?|refurbished|outlet|remanufacturad[oa]s?|recertificad[oa]s?)\b", re.IGNORECASE)
    reparar_re = re.compile(r"\b(para reparar|para repuesto|despiece|no funciona|repuestos?|a revisar|para arreglo|dañad[oa]|detalles?)\b", re.IGNORECASE)
    exhibicion_re = re.compile(r"\b(de exhibicion|en vidriera|en exposicion|sin caja|abierto para revisar)\b", re.IGNORECASE)
    coleccion_re = re.compile(r"\b(antigu[oa]s?|vintage|retro|coleccion|coleccionista|de epoca|numismatica|filatelia)\b", re.IGNORECASE)
    lote_re = re.compile(r"\b(lote de|pack x|combo x|set de|coleccion de)\b", re.IGNORECASE)
    negacion_re = re.compile(r"\b(no |sin |nunca |jam[aá]s )\b", re.IGNORECASE)

    def extract_domain_regexes(titles_subset):
        r_sin_uso = np.array([bool(sin_uso_re.search(t)) for t in titles_subset], dtype=np.float32)
        r_reacond = np.array([bool(reacond_re.search(t)) for t in titles_subset], dtype=np.float32)
        r_reparar = np.array([bool(reparar_re.search(t)) for t in titles_subset], dtype=np.float32)
        r_exhib = np.array([bool(exhibicion_re.search(t)) for t in titles_subset], dtype=np.float32)
        r_colecc = np.array([bool(coleccion_re.search(t)) for t in titles_subset], dtype=np.float32)
        r_lote = np.array([bool(lote_re.search(t)) for t in titles_subset], dtype=np.float32)
        r_neg = np.array([bool(negacion_re.search(t)) for t in titles_subset], dtype=np.float32)
        return np.column_stack([r_sin_uso, r_reacond, r_reparar, r_exhib, r_colecc, r_lote, r_neg])

    dom_regex_trn = extract_domain_regexes([titles[i] for i in trn_idx])
    dom_regex_val = extract_domain_regexes([titles[i] for i in val_idx])

    # 3. Commercial Listing Signals & Attributes (Agente B)
    listing_types = df_tr["listing_type_id"].fillna("").astype(str).tolist()
    buying_modes = df_tr["buying_mode"].fillna("").astype(str).tolist()
    attributes_col = df_tr["attributes"].tolist()
    warranty_col = df_tr["warranty"].fillna("").astype(str).tolist()

    # Commercial listing type indicator (gold_pro / gold_special)
    is_gold_trn = np.array(["gold" in listing_types[i] for i in trn_idx], dtype=np.float32)
    is_gold_val = np.array(["gold" in listing_types[i] for i in val_idx], dtype=np.float32)

    is_buy_now_trn = np.array([buying_modes[i] == "buy_it_now" for i in trn_idx], dtype=np.float32)
    is_buy_now_val = np.array([buying_modes[i] == "buy_it_now" for i in val_idx], dtype=np.float32)

    num_attrs_trn = np.array([len(attributes_col[i]) if isinstance(attributes_col[i], list) else 0 for i in trn_idx], dtype=np.float32)
    num_attrs_val = np.array([len(attributes_col[i]) if isinstance(attributes_col[i], list) else 0 for i in val_idx], dtype=np.float32)

    # Detailed warranty parsing
    def parse_warr_months(w):
        w_l = str(w).lower()
        if "mes" in w_l or "año" in w_l or "garant" in w_l:
            m = re.search(r"(\d+)\s*(mes|año)", w_l)
            if m:
                val = int(m.group(1))
                if "año" in m.group(2):
                    return min(val * 12.0, 60.0)
                return min(float(val), 60.0)
            return 6.0
        return 0.0

    warr_months_trn = np.array([parse_warr_months(warranty_col[i]) for i in trn_idx], dtype=np.float32)
    warr_months_val = np.array([parse_warr_months(warranty_col[i]) for i in val_idx], dtype=np.float32)

    # Price transformations & Category Price Ratio
    log_price_trn = np.log1p(np.maximum(0.0, prices[trn_idx]))
    log_price_val = np.log1p(np.maximum(0.0, prices[val_idx]))

    cat_med_price = pd.Series(prices[trn_idx]).groupby(categories[trn_idx]).median().to_dict()
    global_med_price = float(np.median(prices[trn_idx]))

    price_ratio_trn = np.clip(prices[trn_idx] / np.array([cat_med_price.get(c, global_med_price) + 1.0 for c in categories[trn_idx]], dtype=np.float32), 0.0, 50.0)
    price_ratio_val = np.clip(prices[val_idx] / np.array([cat_med_price.get(c, global_med_price) + 1.0 for c in categories[val_idx]], dtype=np.float32), 0.0, 50.0)

    log_init_qty_trn = np.log1p(init_qtys[trn_idx])
    log_init_qty_val = np.log1p(init_qtys[val_idx])

    is_bulk_trn = (init_qtys[trn_idx] >= 10.0).astype(np.float32)
    is_bulk_val = (init_qtys[val_idx] >= 10.0).astype(np.float32)

    # 4. Target Encoding for Category and Seller (Strictly on Train Fold)
    global_prior = float(y_train.mean())
    df_trn_target = pd.DataFrame({"cat": categories[trn_idx], "seller": sellers[trn_idx], "target": y_train})

    cat_stats = df_trn_target.groupby("cat")["target"].agg(["count", "sum"])
    cat_smooth = ((cat_stats["sum"] + 10.0 * global_prior) / (cat_stats["count"] + 10.0)).to_dict()

    seller_stats = df_trn_target.groupby("seller")["target"].agg(["count", "sum"])
    seller_smooth = ((seller_stats["sum"] + 5.0 * global_prior) / (seller_stats["count"] + 5.0)).to_dict()
    seller_counts = seller_stats["count"].to_dict()

    # Seller LOO for train, smoothed transform for val
    seller_te_trn = np.zeros(len(trn_idx), dtype=np.float32)
    for i in range(len(trn_idx)):
        s = sellers[trn_idx[i]]
        y_i = y_train[i]
        c_i = categories[trn_idx[i]]
        p_c = cat_smooth.get(c_i, global_prior)
        cnt = seller_counts.get(s, 0)
        if cnt > 1:
            seller_te_trn[i] = (seller_stats.loc[s, "sum"] - y_i + 5.0 * p_c) / (cnt - 1 + 5.0)
        else:
            seller_te_trn[i] = p_c

    seller_te_val = np.array([seller_smooth.get(s, cat_smooth.get(c, global_prior)) for s, c in zip(sellers[val_idx], categories[val_idx])], dtype=np.float32)
    cat_te_trn = np.array([cat_smooth.get(c, global_prior) for c in categories[trn_idx]], dtype=np.float32)
    cat_te_val = np.array([cat_smooth.get(c, global_prior) for c in categories[val_idx]], dtype=np.float32)

    # Singleton indicator
    is_singleton_trn = np.array([seller_counts.get(s, 0) <= 1 for s in sellers[trn_idx]], dtype=np.float32)
    is_singleton_val = np.array([seller_counts.get(s, 0) == 0 for s in sellers[val_idx]], dtype=np.float32)

    # 5. Text Specialist Linear Model (TF-IDF fitted strictly on train fold)
    print("Ajustando TF-IDF 100k estrictamente dentro de train...", flush=True)
    tr_corpus = [titles[i] for i in trn_idx]
    val_corpus = [titles[i] for i in val_idx]

    vec_w = TfidfVectorizer(ngram_range=(1, 2), max_features=40000, min_df=3, sublinear_tf=True)
    vec_c = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=60000, min_df=5, sublinear_tf=True)

    X_tr_w = vec_w.fit_transform(tr_corpus)
    X_val_w = vec_w.transform(val_corpus)
    X_tr_c = vec_c.fit_transform(tr_corpus)
    X_val_c = vec_c.transform(val_corpus)

    X_tr_tfidf = sparse.hstack([X_tr_w, X_tr_c], format="csr")
    X_val_tfidf = sparse.hstack([X_val_w, X_val_c], format="csr")

    # Fast Linear Specialist
    lr_text = LogisticRegression(C=1.5, max_iter=300, solver="lbfgs", random_state=42)
    lr_text.fit(X_tr_tfidf, y_train)
    p_text_trn = lr_text.predict_proba(X_tr_tfidf)[:, 1].astype(np.float32)
    p_text_val = lr_text.predict_proba(X_val_tfidf)[:, 1].astype(np.float32)

    # SVD 64 dimensions
    svd = TruncatedSVD(n_components=64, random_state=42)
    X_tr_svd = svd.fit_transform(X_tr_tfidf).astype(np.float32)
    X_val_svd = svd.transform(X_val_tfidf).astype(np.float32)

    # -------------------------------------------------------------------------
    # MATRICES FOR SCREENING
    # -------------------------------------------------------------------------
    # Podium baseline feature set (F130)
    X_base_trn = np.column_stack([
        log_price_trn, price_ratio_trn, num_attrs_trn, (warr_months_trn > 0).astype(np.float32),
        log_init_qty_trn, cat_te_trn, seller_te_trn, base_text_trn, p_text_trn, X_tr_svd
    ])
    X_base_val = np.column_stack([
        log_price_val, price_ratio_val, num_attrs_val, (warr_months_val > 0).astype(np.float32),
        log_init_qty_val, cat_te_val, seller_te_val, base_text_val, p_text_val, X_val_svd
    ])

    # Enriched Feature Set (F150+)
    X_enriched_trn = np.column_stack([
        X_base_trn, dom_regex_trn, is_gold_trn, is_buy_now_trn, warr_months_trn, is_bulk_trn, is_singleton_trn
    ])
    X_enriched_val = np.column_stack([
        X_base_val, dom_regex_val, is_gold_val, is_buy_now_val, warr_months_val, is_bulk_val, is_singleton_val
    ])

    print(f"Dimensiones de Matrices: Base={X_base_trn.shape}, Enriched={X_enriched_trn.shape}")

    # -------------------------------------------------------------------------
    # EVALUATION OF CANDIDATES (STAGE 1)
    # -------------------------------------------------------------------------
    print("\nEntrenando y evaluando candidatos en Fold 1...", flush=True)
    results = []

    # C0: Exact Podium Blend from OOF file
    diff_podium = (p_podium_val >= 0.5).astype(int) != y_val
    results.append({
        "candidate": "C0_Podium_Baseline_Blend",
        "features": "F130_Podium_Original",
        "model": "Optimal_Convex_Blend",
        "accuracy": round(acc_podium_val, 5),
        "hits_fold1": hits_podium_val,
        "delta_hits_vs_podium": 0,
        "fixes_vs_podium": 0,
        "breaks_vs_podium": 0,
        "log_loss": round(ll_podium_val, 5),
        "status": "REFERENCE_BASELINE"
    })

    # C1: LightGBM Base Reproduction
    lgb_base = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.03, num_leaves=55, subsample=0.85, colsample_bytree=0.8, random_state=42, n_jobs=8, verbose=-1)
    lgb_base.fit(X_base_trn, y_train)
    p_c1 = lgb_base.predict_proba(X_base_val)[:, 1]
    acc_c1 = accuracy_score(y_val, (p_c1 >= 0.5).astype(int))
    hits_c1 = int(round(acc_c1 * len(y_val)))
    ll_c1 = log_loss(y_val, p_c1)
    pred_c1 = (p_c1 >= 0.5).astype(int)
    fixes_c1 = int((diff_podium & (pred_c1 == y_val)).sum())
    breaks_c1 = int(((~diff_podium) & (pred_c1 != y_val)).sum())

    results.append({
        "candidate": "C1_LGBM_Base_Reproduction",
        "features": "F130_Base",
        "model": "LightGBM_800",
        "accuracy": round(acc_c1, 5),
        "hits_fold1": hits_c1,
        "delta_hits_vs_podium": hits_c1 - hits_podium_val,
        "fixes_vs_podium": fixes_c1,
        "breaks_vs_podium": breaks_c1,
        "log_loss": round(ll_c1, 5),
        "status": "PASS_STAGE1" if hits_c1 >= hits_podium_val - 20 else "REJECT"
    })

    # C2: LightGBM Enriched F150+ (Agentes A + B + C)
    lgb_enr = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.03, num_leaves=63, subsample=0.85, colsample_bytree=0.8, random_state=42, n_jobs=8, verbose=-1)
    lgb_enr.fit(X_enriched_trn, y_train)
    p_c2 = lgb_enr.predict_proba(X_enriched_val)[:, 1]
    acc_c2 = accuracy_score(y_val, (p_c2 >= 0.5).astype(int))
    hits_c2 = int(round(acc_c2 * len(y_val)))
    ll_c2 = log_loss(y_val, p_c2)
    pred_c2 = (p_c2 >= 0.5).astype(int)
    fixes_c2 = int((diff_podium & (pred_c2 == y_val)).sum())
    breaks_c2 = int(((~diff_podium) & (pred_c2 != y_val)).sum())

    results.append({
        "candidate": "C2_LGBM_Enriched_F150",
        "features": "F150_Domain_Regex_and_Commercial",
        "model": "LightGBM_800_Leaves63",
        "accuracy": round(acc_c2, 5),
        "hits_fold1": hits_c2,
        "delta_hits_vs_podium": hits_c2 - hits_podium_val,
        "fixes_vs_podium": fixes_c2,
        "breaks_vs_podium": breaks_c2,
        "log_loss": round(ll_c2, 5),
        "status": "PASS_STAGE1" if hits_c2 >= hits_podium_val - 15 else "REJECT"
    })

    # C3: XGBoost Hist on Enriched F150+ (Agente D)
    xgb_enr = xgb.XGBClassifier(n_estimators=600, learning_rate=0.035, max_depth=7, subsample=0.85, colsample_bytree=0.8, tree_method="hist", random_state=42, n_jobs=8)
    xgb_enr.fit(X_enriched_trn, y_train)
    p_c3 = xgb_enr.predict_proba(X_enriched_val)[:, 1]
    acc_c3 = accuracy_score(y_val, (p_c3 >= 0.5).astype(int))
    hits_c3 = int(round(acc_c3 * len(y_val)))
    ll_c3 = log_loss(y_val, p_c3)
    pred_c3 = (p_c3 >= 0.5).astype(int)
    fixes_c3 = int((diff_podium & (pred_c3 == y_val)).sum())
    breaks_c3 = int(((~diff_podium) & (pred_c3 != y_val)).sum())

    results.append({
        "candidate": "C3_XGBoost_Enriched_F150",
        "features": "F150_Domain_Regex_and_Commercial",
        "model": "XGBoost_Hist_600",
        "accuracy": round(acc_c3, 5),
        "hits_fold1": hits_c3,
        "delta_hits_vs_podium": hits_c3 - hits_podium_val,
        "fixes_vs_podium": fixes_c3,
        "breaks_vs_podium": breaks_c3,
        "log_loss": round(ll_c3, 5),
        "status": "PASS_STAGE1" if hits_c3 >= hits_podium_val - 15 else "REJECT"
    })

    # C4: Dual Blend (Podium Reference + Enriched LGBM)
    p_c4 = 0.65 * p_podium_val + 0.35 * p_c2
    acc_c4 = accuracy_score(y_val, (p_c4 >= 0.5).astype(int))
    hits_c4 = int(round(acc_c4 * len(y_val)))
    ll_c4 = log_loss(y_val, p_c4)
    pred_c4 = (p_c4 >= 0.5).astype(int)
    fixes_c4 = int((diff_podium & (pred_c4 == y_val)).sum())
    breaks_c4 = int(((~diff_podium) & (pred_c4 != y_val)).sum())

    results.append({
        "candidate": "C4_Podium_plus_Enriched_LGBM",
        "features": "Podium_Blend_plus_F150",
        "model": "Convex_Ensemble_65_35",
        "accuracy": round(acc_c4, 5),
        "hits_fold1": hits_c4,
        "delta_hits_vs_podium": hits_c4 - hits_podium_val,
        "fixes_vs_podium": fixes_c4,
        "breaks_vs_podium": breaks_c4,
        "log_loss": round(ll_c4, 5),
        "status": "PASS_STAGE1" if hits_c4 >= hits_podium_val else "REJECT"
    })

    # C5: Triple Blend (Podium Reference + Enriched LGBM + Enriched XGBoost)
    p_c5 = 0.55 * p_podium_val + 0.25 * p_c2 + 0.20 * p_c3
    acc_c5 = accuracy_score(y_val, (p_c5 >= 0.5).astype(int))
    hits_c5 = int(round(acc_c5 * len(y_val)))
    ll_c5 = log_loss(y_val, p_c5)
    pred_c5 = (p_c5 >= 0.5).astype(int)
    fixes_c5 = int((diff_podium & (pred_c5 == y_val)).sum())
    breaks_c5 = int(((~diff_podium) & (pred_c5 != y_val)).sum())

    results.append({
        "candidate": "C5_Podium_Triad_Ensemble",
        "features": "Podium_plus_LGB_plus_XGB",
        "model": "Convex_Triad_55_25_20",
        "accuracy": round(acc_c5, 5),
        "hits_fold1": hits_c5,
        "delta_hits_vs_podium": hits_c5 - hits_podium_val,
        "fixes_vs_podium": fixes_c5,
        "breaks_vs_podium": breaks_c5,
        "log_loss": round(ll_c5, 5),
        "status": "PASS_STAGE1" if hits_c5 >= hits_podium_val else "REJECT"
    })

    df_res = pd.DataFrame(results)
    df_res.to_csv(out_dir / "stage1_screening_report.csv", index=False)

    print("\n" + "=" * 85)
    print("RESULTADOS DEL SCREENING DE STAGE 1 (FOLD 1, 14.000 FILAS):")
    print("=" * 85)
    print(df_res[["candidate", "accuracy", "hits_fold1", "delta_hits_vs_podium", "fixes_vs_podium", "breaks_vs_podium", "log_loss", "status"]].to_string(index=False))

    manifest = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": "Stage 1 Fast Screening",
        "fold_evaluated": 1,
        "n_val": len(val_idx),
        "reference_baseline_hits": hits_podium_val,
        "candidates": results,
        "passed_to_stage2": [r["candidate"] for r in results if "PASS" in r["status"]]
    }
    with open(out_dir / "stage1_screening_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nStage 1 finalizado en {time.time() - t0:.2f} s.")

if __name__ == "__main__":
    main()
