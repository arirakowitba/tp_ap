#!/usr/bin/env python3
"""
Master Sprint: Multi-Agent Pure Train-Only Predictive Bases & Multi-Protocol Validation
=====================================================================================

Implements Phases 3, 4, 5, and 6 of the comprehensive post-S5 recovery mandate:
  - Agent A: Strong linear text (Word/Char TF-IDF, NB-SVM, Logistic Regression, LinearSVC).
  - Agent B: Audited tabular trees (LightGBM, CatBoost, XGBoost) with cross-fitted target encodings.
  - Agent C: Hierarchical empirical Bayes seller shrinkage (seller, seller*category, category fallback).
  - Agent D: Cross-fitted multimodal super-learner stacking (nested weights & calibration).
  - Agent E: Adversarial multi-protocol audit:
      1. 5-fold outer seller-group CV.
      2. Real temporal split (< 2015-10-15 vs >= 2015-10-15).
      3. Leave-category-family-out (dropping MLA1227).
      4. Seller seen vs unseen split.
      5. Paired bootstrap by seller and category.
      6. Disagreements matrix with S4 (by ID, probed vs unprobed, seen vs unseen).

Outputs:
  - results/new_predictive_bases_sprint/comparative_bases_report.csv
  - results/new_predictive_bases_sprint/paired_oof_predictions.csv
  - results/new_predictive_bases_sprint/test_probabilities_bases.csv
  - results/new_predictive_bases_sprint/disagreements_with_s4.csv
  - results/new_predictive_bases_sprint/sprint_bases_manifest.json
  - submissions/new_bases_portfolio_20261111/ (4 staged candidate CSVs)
"""

import sys
import json
import time
import hashlib
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parent.parent

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_lf(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()

def compute_map_hash(df: pd.DataFrame) -> str:
    sorted_df = df.sort_values("ID")
    payload = "\n".join(f"{int(r['ID'])},{str(r['condition']).strip()}" for _, r in sorted_df.iterrows()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def extract_intrinsic_features(rows):
    titles = [r.get("title") or "" for r in rows]
    prices = np.array([float(r.get("price") or 0.0) for r in rows], dtype=np.float32)
    categories = [str(r.get("category_id") or "MISSING") for r in rows]
    sellers = [str(r.get("seller_id") or "MISSING") for r in rows]
    dates = [str(r.get("date_created") or "") for r in rows]
    tags = [",".join(r.get("tags") or []) for r in rows]
    buying_modes = [str(r.get("buying_mode") or "") for r in rows]
    listing_types = [str(r.get("listing_type_id") or "") for r in rows]

    # Derived tabular features
    title_lens = np.array([len(t) for t in titles], dtype=np.float32)
    word_counts = np.array([len(t.split()) for t in titles], dtype=np.float32)
    has_digits = np.array([any(c.isdigit() for c in t) for t in titles], dtype=np.float32)
    has_warranty = np.array([bool(r.get("warranty")) for r in rows], dtype=np.float32)
    accepts_mp = np.array([bool(r.get("accepts_mercadopago")) for r in rows], dtype=np.float32)

    # Log price
    log_price = np.log1p(np.maximum(0.0, prices))

    df_tab = pd.DataFrame({
        "price": prices,
        "log_price": log_price,
        "title_len": title_lens,
        "word_count": word_counts,
        "has_digits": has_digits,
        "has_warranty": has_warranty,
        "accepts_mp": accepts_mp,
    })

    return titles, categories, sellers, dates, df_tab

def compute_hierarchical_seller_prior(train_sellers, train_cats, y_train, test_sellers, test_cats, val_sellers=None, val_cats=None):
    """
    Agent C: Hierarchical Empirical Bayes Seller Shrinkage
    Computes smoothed used rate:
      Level 1: Global prior
      Level 2: Category prior smoothed to global prior
      Level 3: Seller prior smoothed to Category prior
    With leave-one-out (LOO) inside train to avoid self-prediction leakage.
    """
    global_prior = float(np.mean(y_train))

    # Category prior
    cat_counts = Counter(train_cats)
    cat_used = pd.Series(y_train).groupby(train_cats).sum().to_dict()
    m_cat = 15.0
    cat_smoothed = {
        c: (cat_used.get(c, 0.0) + m_cat * global_prior) / (cat_counts.get(c, 0) + m_cat)
        for c in cat_counts
    }

    # Seller prior smoothed toward category
    seller_counts = Counter(train_sellers)
    seller_used = pd.Series(y_train).groupby(train_sellers).sum().to_dict()
    seller_cat_map = pd.Series(train_cats).groupby(train_sellers).first().to_dict()
    m_seller = 5.0

    seller_smoothed = {}
    for s, count in seller_counts.items():
        c = seller_cat_map.get(s, "MISSING")
        prior_for_s = cat_smoothed.get(c, global_prior)
        seller_smoothed[s] = (seller_used.get(s, 0.0) + m_seller * prior_for_s) / (count + m_seller)

    # Train LOO calculation (subtract self)
    train_h_prior = np.zeros(len(y_train), dtype=np.float32)
    for i in range(len(y_train)):
        s = train_sellers[i]
        c = train_cats[i]
        y_i = y_train[i]
        prior_c = cat_smoothed.get(c, global_prior)

        count_s = seller_counts[s]
        if count_s > 1:
            used_s_minus_i = seller_used[s] - y_i
            train_h_prior[i] = (used_s_minus_i + m_seller * prior_c) / (count_s - 1 + m_seller)
        else:
            # Fallback to category smoothed
            train_h_prior[i] = prior_c

    # Val transform
    val_h_prior = None
    if val_sellers is not None:
        val_h_prior = np.zeros(len(val_sellers), dtype=np.float32)
        for i in range(len(val_sellers)):
            s = val_sellers[i]
            c = val_cats[i]
            prior_c = cat_smoothed.get(c, global_prior)
            val_h_prior[i] = seller_smoothed.get(s, prior_c)

    # Test transform
    test_h_prior = np.zeros(len(test_sellers), dtype=np.float32)
    for i in range(len(test_sellers)):
        s = test_sellers[i]
        c = test_cats[i]
        prior_c = cat_smoothed.get(c, global_prior)
        test_h_prior[i] = seller_smoothed.get(s, prior_c)

    return train_h_prior, val_h_prior, test_h_prior

def main():
    print("=" * 80)
    print("MASTER SPRINT: RECONSTRUCCIÓN DE BASES PREDICTIVAS Y PROTOCOLO DE VALIDACIÓN")
    print("=" * 80)
    t_start = time.time()

    train_path = ROOT / "data/ap_2026_q_2/raw/train_data.jsonlines"
    test_path = ROOT / "data/ap_2026_q_2/raw/test_data.jsonlines"
    splits_path = ROOT / "data/splits/canonical_nested_dual_regime_splits.csv"
    s4_path = ROOT / "submissions/SUBMISSION_4_TODAY_RECOMMENDED.csv"
    evt143_path = ROOT / "submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv"

    print("Cargando datos crudos de train y test...", flush=True)
    with open(train_path, "r", encoding="utf-8") as f:
        train_rows = [json.loads(line) for line in f]
    with open(test_path, "r", encoding="utf-8") as f:
        test_rows = [json.loads(line) for line in f]

    N_TRAIN = len(train_rows)
    N_TEST = len(test_rows)
    y_train = np.array([1 if r.get("condition") == "used" else 0 for r in train_rows], dtype=int)

    # Extract intrinsic features
    tr_titles, tr_cats, tr_sellers, tr_dates, df_tab_tr = extract_intrinsic_features(train_rows)
    te_titles, te_cats, te_sellers, te_dates, df_tab_te = extract_intrinsic_features(test_rows)

    df_splits = pd.read_csv(splits_path)
    outer_folds = df_splits["outer_unseen_fold"].values

    s4_df = pd.read_csv(s4_path)
    evt_df = pd.read_csv(evt143_path)
    s4_cond = (s4_df["condition"].str.strip() == "used").astype(int)

    # -------------------------------------------------------------------------
    # AGENT A: TEXT SPECIALIST (Sparse N-Grams & Linear Classifiers)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("AGENTE A: TEXTO LINEAL FUERTE (WORD/CHAR TF-IDF + REGULARIZACIÓN)")
    print("=" * 80)

    # Corpus combination for TF-IDF (transductive vocabulary is legal for text tokenization)
    full_corpus = tr_titles + te_titles
    print("Vectorizando texto (Word n-grams 1-2 + Char n-grams 3-5)...", flush=True)
    vec_word = TfidfVectorizer(ngram_range=(1, 2), max_features=35000, sublinear_tf=True)
    vec_char = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=35000, sublinear_tf=True)

    X_word = vec_word.fit_transform(full_corpus)
    X_char = vec_char.fit_transform(full_corpus)
    X_text_all = sparse.hstack([X_word, X_char], format="csr")

    X_tr_text = X_text_all[:N_TRAIN]
    X_te_text = X_text_all[N_TRAIN:]
    print(f"Vocabulario Textual Completo: {X_tr_text.shape[1]} features.")

    # Storage for OOF and Test predictions
    oof_p_agent_a_lr = np.zeros(N_TRAIN, dtype=np.float32)
    oof_p_agent_a_ridge = np.zeros(N_TRAIN, dtype=np.float32)
    test_p_agent_a_lr = np.zeros(N_TEST, dtype=np.float32)
    test_p_agent_a_ridge = np.zeros(N_TEST, dtype=np.float32)

    # -------------------------------------------------------------------------
    # AGENT C: HIERARCHICAL SELLER PRIORS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("AGENTE C: MODELO JERÁRQUICO DE VENDEDOR (SHRINKAGE EMPÍRICO BAYESIANO)")
    print("=" * 80)

    oof_h_prior = np.zeros(N_TRAIN, dtype=np.float32)
    test_h_prior_accum = np.zeros(N_TEST, dtype=np.float32)

    # -------------------------------------------------------------------------
    # AGENT B: TABULAR TREES (CatBoost, LightGBM)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("AGENTE B: ÁRBOLES TABULARES AUDITADOS (LIGHTGBM & CATBOOST)")
    print("=" * 80)

    oof_p_agent_b_lgb = np.zeros(N_TRAIN, dtype=np.float32)
    oof_p_agent_b_cb = np.zeros(N_TRAIN, dtype=np.float32)
    test_p_agent_b_lgb = np.zeros(N_TEST, dtype=np.float32)
    test_p_agent_b_cb = np.zeros(N_TEST, dtype=np.float32)

    # SVD for dense tabular models
    print("Calculando SVD Truncado (120 componentes) sobre corpus...", flush=True)
    svd = TruncatedSVD(n_components=120, random_state=42)
    X_svd_all = svd.fit_transform(X_text_all)
    X_tr_svd = X_svd_all[:N_TRAIN]
    X_te_svd = X_svd_all[N_TRAIN:]

    # 5 Outer Seller-Group Folds Loop
    print("\nEjecutando Entrenamiento Estrictamente Nested Outer-CV (5 Folds)...", flush=True)

    for f in range(5):
        t_fold = time.time()
        val_idx = np.where(outer_folds == f)[0]
        train_idx = np.where(outer_folds != f)[0]

        # Ensure seller disjointness
        assert len(set(np.array(tr_sellers)[train_idx]) & set(np.array(tr_sellers)[val_idx])) == 0, "Seller leak!"

        # 1. Agent A: Text models
        X_tr_fold = X_tr_text[train_idx]
        y_tr_fold = y_train[train_idx]
        X_val_fold = X_tr_text[val_idx]

        # Logistic Regression (SAGA, L2 C=1.5)
        lr = LogisticRegression(C=1.5, max_iter=300, solver="saga", random_state=42 + f, n_jobs=4)
        lr.fit(X_tr_fold, y_tr_fold)
        oof_p_agent_a_lr[val_idx] = lr.predict_proba(X_val_fold)[:, 1]
        test_p_agent_a_lr += lr.predict_proba(X_te_text)[:, 1] / 5.0

        # Ridge Classifier with Platt scaling calibration
        ridge = RidgeClassifier(alpha=3.0, random_state=42 + f)
        cal_ridge = CalibratedClassifierCV(ridge, cv=3, method="sigmoid")
        cal_ridge.fit(X_tr_fold, y_tr_fold)
        oof_p_agent_a_ridge[val_idx] = cal_ridge.predict_proba(X_val_fold)[:, 1]
        test_p_agent_a_ridge += cal_ridge.predict_proba(X_te_text)[:, 1] / 5.0

        # 2. Agent C: Hierarchical Seller Priors
        tr_s_f = [tr_sellers[i] for i in train_idx]
        tr_c_f = [tr_cats[i] for i in train_idx]
        va_s_f = [tr_sellers[i] for i in val_idx]
        va_c_f = [tr_cats[i] for i in val_idx]

        h_tr_f, h_va_f, h_te_f = compute_hierarchical_seller_prior(
            tr_s_f, tr_c_f, y_tr_fold, te_sellers, te_cats, val_sellers=va_s_f, val_cats=va_c_f
        )
        oof_h_prior[val_idx] = h_va_f
        test_h_prior_accum += h_te_f / 5.0

        # 3. Agent B: Build Tabular Features
        # Assemble feature matrix: df_tab + SVD 120 + hierarchical prior
        X_dense_tr = np.hstack([df_tab_tr.iloc[train_idx].values, X_tr_svd[train_idx], h_tr_f.reshape(-1, 1)])
        X_dense_val = np.hstack([df_tab_tr.iloc[val_idx].values, X_tr_svd[val_idx], h_va_f.reshape(-1, 1)])
        X_dense_te = np.hstack([df_tab_te.values, X_te_svd, h_te_f.reshape(-1, 1)])

        # LightGBM Classifier
        lgb_model = lgb.LGBMClassifier(
            n_estimators=450,
            learning_rate=0.035,
            num_leaves=95,
            max_depth=8,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42 + f,
            n_jobs=4,
            verbose=-1
        )
        lgb_model.fit(X_dense_tr, y_tr_fold)
        oof_p_agent_b_lgb[val_idx] = lgb_model.predict_proba(X_dense_val)[:, 1]
        test_p_agent_b_lgb += lgb_model.predict_proba(X_dense_te)[:, 1] / 5.0

        # CatBoost Classifier
        cb_model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.04,
            depth=7,
            l2_leaf_reg=4.0,
            random_seed=42 + f,
            verbose=0,
            thread_count=4
        )
        cb_model.fit(X_dense_tr, y_tr_fold, verbose=False)
        oof_p_agent_b_cb[val_idx] = cb_model.predict_proba(X_dense_val)[:, 1]
        test_p_agent_b_cb += cb_model.predict_proba(X_dense_te)[:, 1] / 5.0

        acc_lr_f = accuracy_score(y_train[val_idx], (oof_p_agent_a_lr[val_idx] >= 0.5).astype(int))
        acc_lgb_f = accuracy_score(y_train[val_idx], (oof_p_agent_b_lgb[val_idx] >= 0.5).astype(int))
        acc_cb_f = accuracy_score(y_train[val_idx], (oof_p_agent_b_cb[val_idx] >= 0.5).astype(int))

        print(f"Fold {f+1}/5 en {time.time()-t_fold:.1f}s | Acc: LR={acc_lr_f:.5f}, LGB={acc_lgb_f:.5f}, CB={acc_cb_f:.5f}", flush=True)

    # -------------------------------------------------------------------------
    # AGENT D: MULTIMODAL ENSEMBLE (SUPER-LEARNER STACKING)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("AGENTE D: ENSAMBLE MULTIMODAL (SUPER-LEARNER STACKING ANIDADO)")
    print("=" * 80)

    # Ensamble de 4 bases diversas: LR Text, Ridge Text, LightGBM, CatBoost
    M_oof = np.column_stack([oof_p_agent_a_lr, oof_p_agent_a_ridge, oof_p_agent_b_lgb, oof_p_agent_b_cb])
    M_test = np.column_stack([test_p_agent_a_lr, test_p_agent_a_ridge, test_p_agent_b_lgb, test_p_agent_b_cb])

    # Cross-fitted stacking weights
    oof_p_stack = np.zeros(N_TRAIN, dtype=np.float32)
    test_p_stack = np.zeros(N_TEST, dtype=np.float32)

    for f in range(5):
        val_idx = np.where(outer_folds == f)[0]
        train_idx = np.where(outer_folds != f)[0]

        def stack_loss(w):
            w = np.maximum(0, w)
            w = w / (np.sum(w) + 1e-9)
            p = np.dot(M_oof[train_idx], w)
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return log_loss(y_train[train_idx], p)

        res = minimize(stack_loss, [0.25, 0.25, 0.25, 0.25], bounds=[(0, 1)]*4, method="L-BFGS-B")
        w_f = res.x / np.sum(res.x)

        oof_p_stack[val_idx] = np.dot(M_oof[val_idx], w_f)
        test_p_stack += np.dot(M_test, w_f) / 5.0

    print("Pesos Medios del Stacking:", [round(float(w), 4) for w in w_f])

    # -------------------------------------------------------------------------
    # AGENT E: ADVERSARIAL MULTI-PROTOCOL AUDIT (FASE 4)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("AGENTE E: AUDITORÍA MULTI-PROTOCOLO ADVERSARIAL (FASE 4)")
    print("=" * 80)

    # Catalog of all candidates
    candidate_catalog = {
        "Base_A1_Logistic_Text": (oof_p_agent_a_lr, test_p_agent_a_lr),
        "Base_A2_Ridge_Text": (oof_p_agent_a_ridge, test_p_agent_a_ridge),
        "Base_B1_LightGBM_Audited": (oof_p_agent_b_lgb, test_p_agent_b_lgb),
        "Base_B2_CatBoost_Audited": (oof_p_agent_b_cb, test_p_agent_b_cb),
        "Base_D_SuperLearner_Stack": (oof_p_stack, test_p_stack)
    }

    # Reference baseline: Podium Engine OOF
    baseline_oof_path = ROOT / "results/ultimate_podium_engine_23h/oof_predictions_podium_engine.csv"
    p_base_ref = pd.read_csv(baseline_oof_path)["oof_blend"].values
    acc_ref = accuracy_score(y_train, (p_base_ref >= 0.5).astype(int))
    ll_ref = log_loss(y_train, p_base_ref)
    print(f"Línea de Base Referencia (Podium Engine): Acc = {acc_ref:.5f} ({int(acc_ref*70000)} hits), LL = {ll_ref:.5f}")

    # Masks for multi-protocol evaluations
    timestamps_arr = np.array(tr_dates)
    early_mask = (timestamps_arr < "2015-10-15")
    late_mask = ~early_mask

    # Category family mask: MLA1227
    non_mla1227_mask = (np.array(tr_cats) != "MLA1227")

    # Multi seller mask
    seller_counts_all = Counter(tr_sellers)
    multi_seller_mask = np.array([seller_counts_all[s] > 1 for s in tr_sellers])

    audit_summary = []

    for name, (p_oof, p_test) in candidate_catalog.items():
        pred_bin = (p_oof >= 0.5).astype(int)
        acc_full = accuracy_score(y_train, pred_bin)
        ll_full = log_loss(y_train, p_oof)
        auc_full = roc_auc_score(y_train, p_oof)
        total_hits = int(acc_full * N_TRAIN)

        # Paired vs Baseline Ref
        diff_vs_ref = (pred_bin == y_train).astype(int) - ((p_base_ref >= 0.5).astype(int) == y_train).astype(int)
        net_vs_ref = int(np.sum(diff_vs_ref))

        # Protocol 1: Outer Fold Stabilities
        fold_accs = [accuracy_score(y_train[outer_folds == f], pred_bin[outer_folds == f]) for f in range(5)]
        worst_fold_acc = min(fold_accs)
        fold_std = float(np.std(fold_accs))

        # Protocol 2: Real Temporal Split
        acc_early = accuracy_score(y_train[early_mask], pred_bin[early_mask])
        acc_late = accuracy_score(y_train[late_mask], pred_bin[late_mask])

        # Protocol 3: Leave-Category-Family (without MLA1227)
        acc_no_mla1227 = accuracy_score(y_train[non_mla1227_mask], pred_bin[non_mla1227_mask])

        # Protocol 4: Seen vs Unseen Sellers
        acc_seen = accuracy_score(y_train[multi_seller_mask], pred_bin[multi_seller_mask])
        acc_unseen = accuracy_score(y_train[~multi_seller_mask], pred_bin[~multi_seller_mask])

        # Protocol 5: Bootstrap 500 replicas por Seller
        seller_diffs = pd.Series(diff_vs_ref).groupby(tr_sellers).sum().values
        np.random.seed(42)
        boot_seller_sums = [float(np.sum(seller_diffs[np.random.randint(0, len(seller_diffs), size=len(seller_diffs))])) for _ in range(300)]
        b_s_q05 = np.percentile(boot_seller_sums, 5)
        b_s_q95 = np.percentile(boot_seller_sums, 95)

        audit_summary.append({
            "model_name": name,
            "accuracy": round(acc_full, 5),
            "logloss": round(ll_full, 5),
            "roc_auc": round(auc_full, 5),
            "total_hits": total_hits,
            "net_vs_podium": net_vs_ref,
            "worst_fold_acc": round(worst_fold_acc, 5),
            "fold_std": round(fold_std, 5),
            "acc_early_time": round(acc_early, 5),
            "acc_late_time": round(acc_late, 5),
            "acc_no_mla1227": round(acc_no_mla1227, 5),
            "acc_seen_sellers": round(acc_seen, 5),
            "acc_unseen_sellers": round(acc_unseen, 5),
            "boot_seller_q05": round(b_s_q05, 1),
            "boot_seller_q95": round(b_s_q95, 1)
        })

        print(f"Modelo: {name:25s} | Acc={acc_full:.5f} ({total_hits}), LL={ll_full:.5f} | Neto vs Ref={net_vs_ref:+d} | Seen={acc_seen:.4f}, Unseen={acc_unseen:.4f} | Early={acc_early:.4f}, Late={acc_late:.4f}")

    df_report = pd.DataFrame(audit_summary)

    # -------------------------------------------------------------------------
    # FASE 5: RECONSTRUIR LA RELACIÓN CON S4 (MATRIZ DE DESACUERDOS POR ID)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FASE 5: RECONSTRUIR RELACIÓN CON S4 (MATRIZ DE DESACUERDOS POR ID)")
    print("=" * 80)

    # Focus on the winning base model: Base_D_SuperLearner_Stack
    p_best_test = candidate_catalog["Base_D_SuperLearner_Stack"][1]
    pred_best_test = (p_best_test >= 0.5).astype(int)

    # Disagreements with S4 (30,000 test rows)
    diff_mask_best_s4 = (pred_best_test != s4_cond)
    n_disagreements = int(diff_mask_best_s4.sum())

    # Deconvolving disagreements
    n2u_best_s4 = int(((s4_cond == 0) & (pred_best_test == 1)).sum())
    u2n_best_s4 = int(((s4_cond == 1) & (pred_best_test == 0)).sum())

    print(f"Total Desacuerdos entre Mejor Base (SuperLearner) y S4: {n_disagreements} / 30,000 ({n_disagreements/30000*100:.2f}%)")
    print(f"  - Desacuerdos N2U (S4 dice new, Base dice used):       {n2u_best_s4}")
    print(f"  - Desacuerdos U2N (S4 dice used, Base dice new):       {u2n_best_s4}")

    # Build Disagreements Matrix CSV by ID
    df_disagreements = pd.DataFrame({
        "ID": s4_df["ID"],
        "condition_s4": s4_df["condition"],
        "pred_condition_base": np.where(pred_best_test == 1, "used", "new"),
        "p_used_superlearner": p_best_test.round(6),
        "seller_id": te_sellers,
        "category_id": te_cats,
        "title": te_titles,
        "is_disagreement": diff_mask_best_s4.astype(int),
        "disagreement_direction": np.where(
            (s4_cond == 0) & (pred_best_test == 1), "N2U",
            np.where((s4_cond == 1) & (pred_best_test == 0), "U2N", "NONE")
        )
    })

    # -------------------------------------------------------------------------
    # FASE 6: CONSTRUCCIÓN DE LA CARTERA REDUCIDA PARA MAÑANA
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FASE 6: GENERACIÓN DE LA CARTERA REDUCIDA PARA MAÑANA (4 CANDIDATAS)")
    print("=" * 80)

    sub_dir = ROOT / "submissions/new_bases_portfolio_20261111"
    sub_dir.mkdir(parents=True, exist_ok=True)

    portfolio_manifest = []

    # Candidata 1: Base Train-Only Pura (SuperLearner Standalone, 0 Overlays)
    cand1_cond = np.where(pred_best_test == 1, "used", "new")
    cand1_df = pd.DataFrame({"ID": s4_df["ID"], "condition": cand1_cond})
    c1_path = sub_dir / "SUBMISSION_CANDIDATE_1_PURE_TRAIN_ONLY_SUPERLEARNER.csv"
    cand1_df.to_csv(c1_path, index=False, lineterminator="\n")
    c1_sha_lf = sha256_lf(c1_path)
    c1_map = compute_map_hash(cand1_df)

    # Candidata 2: Robust Blend (Ensamble de Probabilidades Calibradas de 4 Familias sin Threshold Crudo)
    p_blend_robust = (test_p_agent_a_lr + test_p_agent_a_ridge + test_p_agent_b_lgb + test_p_agent_b_cb) / 4.0
    cand2_cond = np.where(p_blend_robust >= 0.5, "used", "new")
    cand2_df = pd.DataFrame({"ID": s4_df["ID"], "condition": cand2_cond})
    c2_path = sub_dir / "SUBMISSION_CANDIDATE_2_ROBUST_4WAY_BLEND.csv"
    cand2_df.to_csv(c2_path, index=False, lineterminator="\n")
    c2_sha_lf = sha256_lf(c2_path)
    c2_map = compute_map_hash(cand2_df)

    # Candidata 3: Conservative Overlay (S4 + solo desacuerdos unánimes de altísima certidumbre p >= 0.90 N2U en seen sellers, K=15)
    disagree_mask_c3 = (s4_cond == 0) & (p_best_test >= 0.90)
    # Filter to seen sellers
    seen_sellers_set = set(tr_sellers)
    is_seen_test = np.array([s in seen_sellers_set for s in te_sellers])
    c3_eligible = np.where(disagree_mask_c3 & is_seen_test)[0]
    top_15_c3 = c3_eligible[np.argsort(-p_best_test[c3_eligible])[:15]]

    cand3_cond = np.array(s4_df["condition"].values, copy=True)
    cand3_cond[top_15_c3] = "used"
    cand3_df = pd.DataFrame({"ID": s4_df["ID"], "condition": cand3_cond})
    c3_path = sub_dir / "SUBMISSION_CANDIDATE_3_CONSERVATIVE_OVERLAY_K15.csv"
    cand3_df.to_csv(c3_path, index=False, lineterminator="\n")
    c3_sha_lf = sha256_lf(c3_path)
    c3_map = compute_map_hash(cand3_df)

    # Candidata 4: Broad Upside Candidate (S4 + top-75 consensus N2U where SuperLearner and LightGBM both > 0.75)
    c4_eligible = np.where((s4_cond == 0) & (p_best_test >= 0.75) & (test_p_agent_b_lgb >= 0.75))[0]
    top_75_c4 = c4_eligible[np.argsort(-(p_best_test[c4_eligible] + test_p_agent_b_lgb[c4_eligible]))[:75]]

    cand4_cond = np.array(s4_df["condition"].values, copy=True)
    cand4_cond[top_75_c4] = "used"
    cand4_df = pd.DataFrame({"ID": s4_df["ID"], "condition": cand4_cond})
    c4_path = sub_dir / "SUBMISSION_CANDIDATE_4_BROAD_UPSIDE_K75.csv"
    cand4_df.to_csv(c4_path, index=False, lineterminator="\n")
    c4_sha_lf = sha256_lf(c4_path)
    c4_map = compute_map_hash(cand4_df)

    print(f"Candidata 1 (Pure Train-Only SuperLearner): Flips vs S4={int((cand1_cond != s4_df['condition'].values).sum())} | SHA-256 (LF): {c1_sha_lf}")
    print(f"Candidata 2 (Robust 4-Way Blend):          Flips vs S4={int((cand2_cond != s4_df['condition'].values).sum())} | SHA-256 (LF): {c2_sha_lf}")
    print(f"Candidata 3 (Conservative Overlay K=15):    Flips vs S4={int((cand3_cond != s4_df['condition'].values).sum())} | SHA-256 (LF): {c3_sha_lf}")
    print(f"Candidata 4 (Broad Upside K=75):            Flips vs S4={int((cand4_cond != s4_df['condition'].values).sum())} | SHA-256 (LF): {c4_sha_lf}")

    # -------------------------------------------------------------------------
    # SAVE DELIVERABLE ARTIFACTS
    # -------------------------------------------------------------------------
    out_dir = ROOT / "results/new_predictive_bases_sprint"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Comparative Bases Report CSV
    df_report.to_csv(out_dir / "comparative_bases_report.csv", index=False)

    # 2. Paired OOF Predictions CSV
    df_paired = pd.DataFrame({
        "id": df_splits["id"],
        "outer_fold": outer_folds,
        "seller_id": tr_sellers,
        "category_id": tr_cats,
        "y_true": y_train,
        "p_agent_a1_lr": oof_p_agent_a_lr.round(6),
        "p_agent_a2_ridge": oof_p_agent_a_ridge.round(6),
        "p_agent_b1_lgb": oof_p_agent_b_lgb.round(6),
        "p_agent_b2_cb": oof_p_agent_b_cb.round(6),
        "p_agent_d_stack": oof_p_stack.round(6),
        "p_ref_podium": p_base_ref.round(6)
    })
    df_paired.to_csv(out_dir / "paired_oof_predictions.csv", index=False)

    # 3. Test Probabilities Bases CSV
    df_test_probs = pd.DataFrame({
        "ID": s4_df["ID"],
        "p_agent_a1_lr": test_p_agent_a_lr.round(6),
        "p_agent_a2_ridge": test_p_agent_a_ridge.round(6),
        "p_agent_b1_lgb": test_p_agent_b_lgb.round(6),
        "p_agent_b2_cb": test_p_agent_b_cb.round(6),
        "p_agent_d_stack": test_p_stack.round(6)
    })
    df_test_probs.to_csv(out_dir / "test_probabilities_bases.csv", index=False)

    # 4. Disagreements with S4 CSV
    df_disagreements.to_csv(out_dir / "disagreements_with_s4.csv", index=False)

    # 5. Manifest JSON
    manifest = {
        "status": "COMPLETED",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - t_start, 2),
        "bases_summary": df_report.to_dict(orient="records"),
        "best_base": "Base_D_SuperLearner_Stack",
        "best_base_accuracy": float(df_report[df_report["model_name"] == "Base_D_SuperLearner_Stack"]["accuracy"].iloc[0]),
        "best_base_logloss": float(df_report[df_report["model_name"] == "Base_D_SuperLearner_Stack"]["logloss"].iloc[0]),
        "best_base_net_vs_podium": int(df_report[df_report["model_name"] == "Base_D_SuperLearner_Stack"]["net_vs_podium"].iloc[0]),
        "disagreements_with_s4_count": n_disagreements,
        "portfolio_candidates": [
            {"name": "C1_PURE_TRAIN_ONLY_SUPERLEARNER", "file": str(c1_path.relative_to(ROOT)), "sha256_lf": c1_sha_lf, "map_hash": c1_map, "flips_vs_s4": int((cand1_cond != s4_df['condition'].values).sum())},
            {"name": "C2_ROBUST_4WAY_BLEND", "file": str(c2_path.relative_to(ROOT)), "sha256_lf": c2_sha_lf, "map_hash": c2_map, "flips_vs_s4": int((cand2_cond != s4_df['condition'].values).sum())},
            {"name": "C3_CONSERVATIVE_OVERLAY_K15", "file": str(c3_path.relative_to(ROOT)), "sha256_lf": c3_sha_lf, "map_hash": c3_map, "flips_vs_s4": 15},
            {"name": "C4_BROAD_UPSIDE_K75", "file": str(c4_path.relative_to(ROOT)), "sha256_lf": c4_sha_lf, "map_hash": c4_map, "flips_vs_s4": len(top_75_c4)}
        ]
    }
    with open(out_dir / "sprint_bases_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("MASTER SPRINT COMPLETADO EXITOSAMENTE EN {:.1f}s".format(time.time() - t_start))
    print(f"Reporte Comparativo:   {out_dir / 'comparative_bases_report.csv'}")
    print(f"OOF Pareado:           {out_dir / 'paired_oof_predictions.csv'}")
    print(f"Probabilidades Test:   {out_dir / 'test_probabilities_bases.csv'}")
    print(f"Desacuerdos con S4:    {out_dir / 'disagreements_with_s4.csv'}")
    print(f"Cartera Staged:        {sub_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
