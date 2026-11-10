#!/usr/bin/env python3
"""Calibrated Neural Tabular Network (MLPClassifier) on Item-Intrinsic Features & SVD Text Embeddings.

Mission:
- Strict outer StratifiedGroupKFold(5) grouped by seller_id on 70,000 train items.
- Exactly ZERO seller identity features, ZERO seller target encoding (100% item-intrinsic).
- Fit TfidfVectorizer + TruncatedSVD(128, random_state=42) strictly on outer training fold text.
- Fit StandardScaler strictly on outer training fold tabular features.
- Train MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', alpha=0.01,
                      batch_size=256, learning_rate_init=0.001, max_iter=40,
                      early_stopping=True, validation_fraction=0.10, random_state=42 + fold).
- Generate out-of-fold predictions and test set probabilities.
- Resource boundary: Peak RAM strictly < 3.2 GB via sequential fold processing and gc.collect().

Output Artifacts:
- results/agent_neural_tabular_20261110/oof_neural_tabular.csv (cols: id, p_used, target_condition_used, seller_id)
- results/agent_neural_tabular_20261110/test_probabilities_neural_tabular.csv (cols: id, p_used)
- results/agent_neural_tabular_20261110/summary_report.json
"""

import os
import sys
import gc
import json
import time
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

# Paths
ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "ap_2026_q_2" / "raw" / "train_data.jsonlines"
TEST_PATH = ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines"
OUT_DIR = ROOT / "results" / "agent_neural_tabular_20261110"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Decisive item-intrinsic lexical cue regexes
LEXICAL_CUES = [
    ("cue_0km", re.compile(r"\b0\s*km\b|\bcero\s*km\b|\bsin\s+rodar\b|\b0km\b", re.I)),
    ("cue_caja_sellada", re.compile(r"\bcaja\s+sellada\b|\bcerrad[oa]\s+de\s+f[aá]brica\b|\bprecintad[oa]\b|\ben\s+blister\b|\bnuevo\s+sellado\b|\bnuevo\s+cerrado\b", re.I)),
    ("cue_nuevo", re.compile(r"\bnuev[oa]s?\b|\ba\s+estrenar\b", re.I)),
    ("cue_usado", re.compile(r"\b(usad[oa]s?|segunda\s+mano)\b", re.I)),
    ("cue_repuesto", re.compile(r"\bpara\s+repuestos?\b|\bpara\s+desarme\b|\bno\s+funciona\b|\brot[oa]s?\b|\bpara\s+reparar\b|\bcon\s+falla\b", re.I)),
    ("cue_poco_uso", re.compile(r"\b(un|1)\s+solo\s+uso\b|\bpoco\s+uso\b|\bpoqu[ií]simo\s+uso\b", re.I)),
    ("cue_como_nuevo", re.compile(r"\bcomo\s+nuev[oa]\b|\bigual\s+a\s+nuev[oa]\b|\bseminuevo\b", re.I)),
    ("cue_impecable", re.compile(r"\bimpecable\b|\bde\s+vitrina\b|\bexcelente\s+estado\b|\bmuy\s+buen\s+estado\b", re.I)),
    ("cue_antiguo", re.compile(r"\bantigu[oa]s?\b|\bde\s+[eé]poca\b|\bvintage\b|\bcolecci[oó]n\b|\bretro\b", re.I)),
    ("cue_digital", re.compile(r"\b(libro\s+digital|ebook|pdf|descarga\s+digital)\b", re.I)),
    ("cue_sin_uso", re.compile(r"\bsin\s+uso\b|\bnunca\s+se\s+us[oó]\b|\bnueva\s+sin\s+uso\b", re.I)),
    ("cue_original", re.compile(r"\boriginal\b|\bleg[ií]tim[oa]\b", re.I)),
    ("cue_detalles", re.compile(r"\bcon\s+detalles?\b|\bmarcas?\s+de\s+uso\b|\brayones?\b|\bdesgaste\b", re.I)),
    ("cue_garantia", re.compile(r"\bcon\s+garant[ií]a\b|\bgarant[ií]a\s+oficial\b|\bgarant[ií]a\s+escrita\b", re.I)),
    ("cue_factura", re.compile(r"\bfactura\s+[ab]\b|\bhacemos\s+factura\b", re.I)),
    ("cue_reacondicionado", re.compile(r"\breacondicionad[oa]\b|\brefurbished\b", re.I)),
]


def extract_item_intrinsics(rows):
    """Extract item-intrinsic features and item text corpus.
    
    100% item-intrinsic: ZERO seller identity features, ZERO seller target encoding.
    """
    corpus = []
    tab_features = []
    raw_ids = []
    sellers = []
    categories = []
    conditions = []

    for r in rows:
        raw_ids.append(str(r.get("id") or ""))
        sellers.append(str(r.get("seller_id") or "UNKNOWN"))
        categories.append(str(r.get("category_id") or "UNKNOWN"))
        conditions.append(1 if r.get("condition") == "used" else 0)

        title = str(r.get("title") or "").strip()
        warranty = str(r.get("warranty") or "").strip()
        attrs = r.get("attributes") or []
        attr_vals = [str(a.get("value_name") or a.get("name") or "") for a in attrs if isinstance(a, dict)]
        attr_str = " ".join(attr_vals)

        # Rich item text
        item_text = f"{title} WARRANTY {warranty} ATTR {attr_str}".strip()
        corpus.append(item_text)

        # Tabular Item-Intrinsic Features
        price = float(r.get("price") or 0.0)
        log_price = np.log1p(max(0.0, price))
        title_len = float(len(title))
        title_words = float(len(title.split()))
        digit_ratio = sum(c.isdigit() for c in title) / max(len(title), 1)
        uppercase_ratio = sum(c.isupper() for c in title) / max(len(title), 1)
        qmark_count = float(title.count("?") + title.count("¿"))
        excl_count = float(title.count("!") + title.count("¡"))

        # Extra clean structural item intrinsics
        orig_price = float(r.get("original_price") or 0.0)
        has_orig = 1.0 if orig_price > 0 else 0.0
        log_orig = np.log1p(max(0.0, orig_price))
        iq = np.log1p(max(0.0, float(r.get("initial_quantity") or 0.0)))
        aq = np.log1p(max(0.0, float(r.get("available_quantity") or 0.0)))
        sq = np.log1p(max(0.0, float(r.get("sold_quantity") or 0.0)))
        mp = 1.0 if r.get("accepts_mercadopago") else 0.0
        fs = 1.0 if (r.get("shipping") or {}).get("free_shipping") else 0.0
        pic_count = float(len(r.get("pictures") or []))
        attr_count = float(len(attrs))
        has_warr = 1.0 if len(warranty) > 0 else 0.0
        warr_len = float(len(warranty))

        row = [
            price,
            log_price,
            title_len,
            title_words,
            digit_ratio,
            uppercase_ratio,
            qmark_count,
            excl_count,
            has_orig,
            log_orig,
            iq,
            aq,
            sq,
            mp,
            fs,
            pic_count,
            attr_count,
            has_warr,
            warr_len,
        ]

        # Lexical cue indicator flags
        for _, pat in LEXICAL_CUES:
            row.append(1.0 if pat.search(title) else 0.0)

        tab_features.append(row)

    return (
        np.array(raw_ids),
        np.array(sellers),
        np.array(categories),
        np.array(conditions, dtype=np.int32),
        corpus,
        np.array(tab_features, dtype=np.float32),
    )


def main():
    print("=" * 80)
    print("CALIBRATED NEURAL TABULAR NETWORK (MLPCLASSIFIER)")
    print("Intrinsic Tabular + SVD(128) Text Embeddings | Zero Seller Leakage")
    print("=" * 80)

    start_time = time.time()

    # 1. Load Data
    print(f"\n[1/5] Loading datasets from {TRAIN_PATH} and {TEST_PATH}...", flush=True)
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        train_rows = [json.loads(line) for line in f]
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_rows = [json.loads(line) for line in f]

    n_train = len(train_rows)
    n_test = len(test_rows)
    print(f"Loaded {n_train} train rows and {n_test} test rows.", flush=True)

    # 2. Extract 100% Item-Intrinsic Features
    print("\n[2/5] Extracting item-intrinsic features and text corpora...", flush=True)
    (
        train_ids,
        train_sellers,
        train_categories,
        y_train,
        train_corpus,
        train_tab_base,
    ) = extract_item_intrinsics(train_rows)

    (
        test_ids,
        test_sellers,
        test_categories,
        _,
        test_corpus,
        test_tab_base,
    ) = extract_item_intrinsics(test_rows)

    # Free raw rows to conserve RAM
    del train_rows, test_rows
    gc.collect()

    n_base_tab = train_tab_base.shape[1]
    print(f"Base tabular features: {n_base_tab} per item. Text corpus size: {len(train_corpus)} train, {len(test_corpus)} test.")

    # 3. Setup Outer StratifiedGroupKFold(5) grouped by seller_id
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    outer_splits = list(sgkf.split(train_corpus, y_train, groups=train_sellers))

    print("\n[3/5] Verifying zero seller leakage across 5 outer folds:")
    for fold_i, (tr_idx, val_idx) in enumerate(outer_splits):
        tr_s = set(train_sellers[tr_idx])
        val_s = set(train_sellers[val_idx])
        overlap = tr_s & val_s
        assert len(overlap) == 0, f"FATAL: Seller overlap in fold {fold_i + 1}: {len(overlap)} sellers!"
        print(f"  Fold {fold_i + 1}: outer_train={len(tr_idx)} items ({len(tr_s)} sellers), outer_val={len(val_idx)} items ({len(val_s)} sellers), overlap={len(overlap)}")

    # OOF and test probability containers
    oof_p = np.zeros(n_train, dtype=np.float32)
    test_p_folds = np.zeros((5, n_test), dtype=np.float32)
    fold_reports = []

    print("\n[4/5] Executing 5-Fold Sequential MLP Training Loop...", flush=True)

    for fold_i, (tr_idx, val_idx) in enumerate(outer_splits):
        f_start = time.time()
        print(f"\n{'=' * 40}\n>>> START FOLD {fold_i + 1} / 5\n{'=' * 40}")

        # Step 4a: Category frequency strictly fit on outer training fold
        cat_counts = Counter(train_categories[tr_idx])
        tr_cat_cnt = np.array([cat_counts.get(c, 0) for c in train_categories[tr_idx]], dtype=np.float32)[:, None]
        val_cat_cnt = np.array([cat_counts.get(c, 0) for c in train_categories[val_idx]], dtype=np.float32)[:, None]
        te_cat_cnt = np.array([cat_counts.get(c, 0) for c in test_categories], dtype=np.float32)[:, None]

        tr_cat_freq_log1p = np.log1p(tr_cat_cnt)
        val_cat_freq_log1p = np.log1p(val_cat_cnt)
        te_cat_freq_log1p = np.log1p(te_cat_cnt)

        tr_cat_freq_ratio = tr_cat_cnt / float(len(tr_idx))
        val_cat_freq_ratio = val_cat_cnt / float(len(tr_idx))
        te_cat_freq_ratio = te_cat_cnt / float(len(tr_idx))

        # Assemble full tabular features for this fold
        X_tr_tab = np.column_stack([train_tab_base[tr_idx], tr_cat_freq_log1p, tr_cat_freq_ratio])
        X_val_tab = np.column_stack([train_tab_base[val_idx], val_cat_freq_log1p, val_cat_freq_ratio])
        X_te_tab = np.column_stack([test_tab_base, te_cat_freq_log1p, te_cat_freq_ratio])

        # Step 4b: Scale tabular features using StandardScaler fit ONLY on outer training fold
        scaler = StandardScaler()
        X_tr_tab_scaled = scaler.fit_transform(X_tr_tab).astype(np.float32)
        X_val_tab_scaled = scaler.transform(X_val_tab).astype(np.float32)
        X_te_tab_scaled = scaler.transform(X_te_tab).astype(np.float32)

        # Step 4c: Fit TfidfVectorizer + TruncatedSVD(128, random_state=42) strictly on outer training text
        t_svd = time.time()
        print("Fitting TfidfVectorizer and TruncatedSVD(128) strictly on outer training text...", flush=True)
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=40000, sublinear_tf=True, min_df=2)
        X_tr_tfidf = vec.fit_transform([train_corpus[i] for i in tr_idx])
        X_val_tfidf = vec.transform([train_corpus[i] for i in val_idx])
        X_te_tfidf = vec.transform(test_corpus)

        svd = TruncatedSVD(n_components=128, random_state=42)
        X_tr_svd = svd.fit_transform(X_tr_tfidf).astype(np.float32)
        X_val_svd = svd.transform(X_val_tfidf).astype(np.float32)
        X_te_svd = svd.transform(X_te_tfidf).astype(np.float32)
        print(f"SVD text representations computed in {time.time() - t_svd:.2f}s. TruncatedSVD variance explained: {svd.explained_variance_ratio_.sum():.4f}")

        # Combine tabular and SVD features
        X_tr = np.column_stack([X_tr_tab_scaled, X_tr_svd])
        X_val = np.column_stack([X_val_tab_scaled, X_val_svd])
        X_te = np.column_stack([X_te_tab_scaled, X_te_svd])
        print(f"Total features per item: {X_tr.shape[1]} ({X_tr_tab_scaled.shape[1]} tabular + {X_tr_svd.shape[1]} SVD)")

        # Step 4d: Train MLPClassifier
        t_mlp = time.time()
        seed_fold = 42 + fold_i
        print(f"Training MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=40, seed={seed_fold})...", flush=True)
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            alpha=0.01,
            batch_size=256,
            learning_rate_init=0.001,
            max_iter=40,
            early_stopping=True,
            validation_fraction=0.10,
            random_state=seed_fold,
        )
        clf.fit(X_tr, y_train[tr_idx])
        mlp_time = time.time() - t_mlp
        print(f"MLP fit completed in {mlp_time:.2f}s (stopped at iter {clf.n_iter_}).")

        # Step 4e: Predict validation and test fold probabilities
        p_val_fold = clf.predict_proba(X_val)[:, 1].astype(np.float32)
        p_te_fold = clf.predict_proba(X_te)[:, 1].astype(np.float32)

        oof_p[val_idx] = p_val_fold
        test_p_folds[fold_i] = p_te_fold

        acc_f = accuracy_score(y_train[val_idx], (p_val_fold >= 0.5).astype(int))
        auc_f = roc_auc_score(y_train[val_idx], p_val_fold)
        ll_f = log_loss(y_train[val_idx], p_val_fold)

        fold_duration = time.time() - f_start
        print(f"Fold {fold_i + 1} Metrics -> Accuracy: {acc_f:.5f}, ROC-AUC: {auc_f:.5f}, LogLoss: {ll_f:.5f} [Time: {fold_duration:.1f}s]")

        fold_reports.append({
            "fold": fold_i + 1,
            "accuracy": float(acc_f),
            "roc_auc": float(auc_f),
            "logloss": float(ll_f),
            "n_iterations": int(clf.n_iter_),
            "fit_time_seconds": float(mlp_time),
            "total_fold_seconds": float(fold_duration),
        })

        # Step 4f: Explicit memory release to maintain RAM < 3.2 GB
        del vec, svd, scaler, clf
        del X_tr_tfidf, X_val_tfidf, X_te_tfidf
        del X_tr_svd, X_val_svd, X_te_svd
        del X_tr_tab, X_val_tab, X_te_tab
        del X_tr_tab_scaled, X_val_tab_scaled, X_te_tab_scaled
        del X_tr, X_val, X_te
        gc.collect()

    # 5. Global Out-of-Fold Evaluation
    global_acc = accuracy_score(y_train, (oof_p >= 0.5).astype(int))
    global_auc = roc_auc_score(y_train, oof_p)
    global_ll = log_loss(y_train, oof_p)
    hits = int(((oof_p >= 0.5).astype(int) == y_train).sum())
    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("GLOBAL LEAK-FREE OUT-OF-FOLD METRICS (CALIBRATED NEURAL TABULAR):")
    print(f"  Global Accuracy: {global_acc:.5f} ({hits} / {n_train} correct)")
    print(f"  Global ROC-AUC : {global_auc:.5f}")
    print(f"  Global LogLoss : {global_ll:.5f}")
    print(f"  Total Duration : {total_time:.1f} seconds ({total_time / 60:.2f} minutes)")
    print("=" * 80)

    # 6. Save Artifacts
    print("\n[5/5] Saving results and artifacts...", flush=True)

    # 6a. OOF Predictions
    oof_df = pd.DataFrame({
        "id": np.arange(1, n_train + 1),
        "p_used": oof_p,
        "target_condition_used": y_train,
        "seller_id": train_sellers,
        "item_id": train_ids,
    })
    oof_path = OUT_DIR / "oof_neural_tabular.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"Saved OOF predictions to {oof_path} ({len(oof_df)} rows).")

    # 6b. Test Predictions
    test_p_mean = test_p_folds.mean(axis=0)
    test_df = pd.DataFrame({
        "id": np.arange(1, n_test + 1),
        "p_used": test_p_mean,
        "item_id": test_ids,
    })
    test_path = OUT_DIR / "test_probabilities_neural_tabular.csv"
    test_df.to_csv(test_path, index=False)
    print(f"Saved test probabilities to {test_path} ({len(test_df)} rows).")

    # 6c. Summary Report JSON
    summary = {
        "model_name": "Calibrated Neural Tabular Network (MLPClassifier 256-128-64)",
        "architecture": {
            "hidden_layer_sizes": [256, 128, 64],
            "activation": "relu",
            "alpha": 0.01,
            "batch_size": 256,
            "learning_rate_init": 0.001,
            "max_iter": 40,
            "early_stopping": True,
            "validation_fraction": 0.10,
        },
        "cv_protocol": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) grouped by seller_id",
        "zero_seller_leakage": True,
        "item_intrinsic_features_only": True,
        "n_train": n_train,
        "n_test": n_test,
        "n_tabular_features": int(n_base_tab + 2),
        "n_svd_features": 128,
        "total_features": int(n_base_tab + 2 + 128),
        "global_accuracy": float(global_acc),
        "global_roc_auc": float(global_auc),
        "global_logloss": float(global_ll),
        "hits": hits,
        "total_duration_seconds": float(total_time),
        "fold_metrics": fold_reports,
        "output_files": {
            "oof_csv": str(oof_path),
            "test_csv": str(test_path),
        },
        "leak_free_audit": {
            "seller_overlap_across_folds": 0,
            "seller_identity_features": 0,
            "seller_target_encoding": 0,
            "scaler_fitted_strictly_on_train": True,
            "svd_fitted_strictly_on_train": True,
            "category_freq_fitted_strictly_on_train": True,
        },
    }

    report_path = OUT_DIR / "summary_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary report to {report_path}")

    print("\nTraining and evaluation pipeline completed successfully!")


if __name__ == "__main__":
    main()
