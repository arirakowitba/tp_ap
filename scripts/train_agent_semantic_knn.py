#!/usr/bin/env python3
"""Semantic k-NN Non-Parametric Metric Specialist on Item-Intrinsic Features & SVD Text Embeddings.

Mission:
- Strict outer StratifiedGroupKFold(5) grouped by seller_id on 70,000 train items.
- Exactly ZERO seller identity features, ZERO seller target encoding (100% item-intrinsic).
- Fit TfidfVectorizer + TruncatedSVD(128, random_state=42) strictly on outer training fold text.
- Fit StandardScaler strictly on outer training fold tabular features.
- Normalize SVD text vectors to unit L2 norm (weight 0.7) and normalized tabular features (weight 0.3).
- Index outer training items with NearestNeighbors(n_neighbors=35, metric='cosine', algorithm='brute').
- Query validation and test items in batches (batch_size=2000) using inverse distance weighting.
- Identify exact catalog duplicates (cosine similarity >= 0.98) and near-duplicates.
- Evaluate OOF Accuracy, ROC-AUC, and LogLoss.
- Save OOF predictions, test probabilities, and summary report.
- Resource boundary: Peak RAM strictly < 3.0 GB via sequential fold processing and batched queries.

Output Artifacts:
- results/agent_semantic_knn_20261110/oof_semantic_knn.csv (cols: id, p_used, target_condition_used, seller_id, item_id)
- results/agent_semantic_knn_20261110/test_probabilities_semantic_knn.csv (cols: id, p_used, item_id)
- results/agent_semantic_knn_20261110/summary_report.json
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
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

# Paths
ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "ap_2026_q_2" / "raw" / "train_data.jsonlines"
TEST_PATH = ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines"
OUT_DIR = ROOT / "results" / "agent_semantic_knn_20261110"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Item-intrinsic lexical cue indicators
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
    """Extract 100% item-intrinsic features and rich text representations.

    ZERO seller identity features, ZERO seller target encoding.
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

        # Rich text corpus
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

        # Extra structural item intrinsics
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

        # Lexical cue indicators
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


def compute_distance_weighted_proba(dists, inds, y_train_fold):
    """Compute distance-weighted probability with exact zero-distance handling.

    If exact matches (dist <= 1e-7) exist, only exact matches contribute equal weight.
    Otherwise, weights are inverse distance (1.0 / dist).
    """
    labels = y_train_fold[inds]  # shape: (n_queries, k)
    zero_mask = dists <= 1e-7
    has_zeros = np.any(zero_mask, axis=1)

    weights = np.zeros_like(dists, dtype=np.float64)
    non_zero = ~has_zeros

    if np.any(non_zero):
        weights[non_zero] = 1.0 / np.maximum(dists[non_zero], 1e-7)

    if np.any(has_zeros):
        weights[has_zeros] = zero_mask[has_zeros].astype(np.float64)

    w_sum = np.sum(weights, axis=1, keepdims=True)
    w_sum[w_sum == 0] = 1.0
    p = np.sum((weights / w_sum) * labels, axis=1)
    return p.astype(np.float32)


def main():
    print("=" * 80)
    print("SEMANTIC k-NN NON-PARAMETRIC METRIC SPECIALIST")
    print("Intrinsic Metric Space: SVD(128) Text (0.7) + Scaled Tabular (0.3)")
    print("Zero Seller Leakage | StratifiedGroupKFold(5) grouped by seller_id")
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

    # Containers for predictions and diagnostics
    oof_p = np.zeros(n_train, dtype=np.float32)
    test_p_folds = np.zeros((5, n_test), dtype=np.float32)

    fold_reports = []
    total_val_duplicates = 0
    total_val_duplicate_correct = 0
    total_test_duplicates_per_fold = []

    batch_size = 2000

    print("\n[4/5] Executing 5-Fold Sequential k-NN Metric Indexing Loop...", flush=True)

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

        X_tr_tab = np.column_stack([train_tab_base[tr_idx], tr_cat_freq_log1p, tr_cat_freq_ratio])
        X_val_tab = np.column_stack([train_tab_base[val_idx], val_cat_freq_log1p, val_cat_freq_ratio])
        X_te_tab = np.column_stack([test_tab_base, te_cat_freq_log1p, te_cat_freq_ratio])

        # Step 4b: Scale tabular features using StandardScaler fit ONLY on outer training fold
        scaler = StandardScaler()
        X_tr_tab_scaled = scaler.fit_transform(X_tr_tab).astype(np.float32)
        X_val_tab_scaled = scaler.transform(X_val_tab).astype(np.float32)
        X_te_tab_scaled = scaler.transform(X_te_tab).astype(np.float32)

        # Unit L2-normalize tabular features
        X_tr_tab_norm = normalize(X_tr_tab_scaled, norm='l2', axis=1)
        X_val_tab_norm = normalize(X_val_tab_scaled, norm='l2', axis=1)
        X_te_tab_norm = normalize(X_te_tab_scaled, norm='l2', axis=1)

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
        print(f"SVD text representations computed in {time.time() - t_svd:.2f}s. Explained variance: {svd.explained_variance_ratio_.sum():.4f}")

        # Unit L2-normalize SVD vectors
        X_tr_svd_norm = normalize(X_tr_svd, norm='l2', axis=1)
        X_val_svd_norm = normalize(X_val_svd, norm='l2', axis=1)
        X_te_svd_norm = normalize(X_te_svd, norm='l2', axis=1)

        # Step 4d: Combine normalized text SVD (weight 0.7) and normalized tabular features (weight 0.3)
        X_tr_comb = normalize(np.column_stack([0.7 * X_tr_svd_norm, 0.3 * X_tr_tab_norm]), norm='l2', axis=1).astype(np.float32)
        X_val_comb = normalize(np.column_stack([0.7 * X_val_svd_norm, 0.3 * X_val_tab_norm]), norm='l2', axis=1).astype(np.float32)
        X_te_comb = normalize(np.column_stack([0.7 * X_te_svd_norm, 0.3 * X_te_tab_norm]), norm='l2', axis=1).astype(np.float32)
        print(f"Combined metric space: {X_tr_comb.shape[1]} dims ({X_tr_svd_norm.shape[1]} SVD + {X_tr_tab_norm.shape[1]} Tabular)")

        # Step 4e: Index outer training items with NearestNeighbors
        t_nn = time.time()
        print("Indexing outer training items with NearestNeighbors(n_neighbors=35, metric='cosine', algorithm='brute')...", flush=True)
        nn = NearestNeighbors(n_neighbors=35, metric='cosine', algorithm='brute', n_jobs=-1)
        nn.fit(X_tr_comb)
        print(f"Index built in {time.time() - t_nn:.2f}s.")

        # Step 4f: Query validation items in batches
        t_q_val = time.time()
        val_dists_list = []
        val_inds_list = []
        for s in range(0, len(val_idx), batch_size):
            e = min(s + batch_size, len(val_idx))
            d, ind = nn.kneighbors(X_val_comb[s:e])
            val_dists_list.append(d)
            val_inds_list.append(ind)
        val_dists = np.vstack(val_dists_list)
        val_inds = np.vstack(val_inds_list)
        print(f"Validation queries ({len(val_idx)} items) completed in {time.time() - t_q_val:.2f}s.")

        # Catalog duplicate detection on validation items (cosine similarity >= 0.98 <=> cosine distance <= 0.02)
        val_near_dup_mask = val_dists[:, 0] <= 0.02
        n_val_dups = int(val_near_dup_mask.sum())
        total_val_duplicates += n_val_dups
        val_dup_correct = 0
        if n_val_dups > 0:
            nearest_neighbor_train_label = y_train[tr_idx][val_inds[val_near_dup_mask, 0]]
            val_dup_correct = int((nearest_neighbor_train_label == y_train[val_idx][val_near_dup_mask]).sum())
            total_val_duplicate_correct += val_dup_correct
            val_dup_acc = val_dup_correct / float(n_val_dups)
        else:
            val_dup_acc = 0.0
        print(f"Validation catalog duplicates (sim >= 0.98): {n_val_dups} / {len(val_idx)} ({n_val_dups / len(val_idx) * 100:.2f}%) | Label agreement: {val_dup_acc * 100:.2f}%")

        # Compute distance-weighted condition probabilities
        p_val_k35 = compute_distance_weighted_proba(val_dists, val_inds, y_train[tr_idx])
        oof_p[val_idx] = p_val_k35

        # Sub-k diagnostic evaluations
        p_val_k15 = compute_distance_weighted_proba(val_dists[:, :15], val_inds[:, :15], y_train[tr_idx])
        p_val_k31 = compute_distance_weighted_proba(val_dists[:, :31], val_inds[:, :31], y_train[tr_idx])

        acc_k15 = accuracy_score(y_train[val_idx], (p_val_k15 >= 0.5).astype(int))
        acc_k31 = accuracy_score(y_train[val_idx], (p_val_k31 >= 0.5).astype(int))
        acc_k35 = accuracy_score(y_train[val_idx], (p_val_k35 >= 0.5).astype(int))
        auc_k35 = roc_auc_score(y_train[val_idx], p_val_k35)
        ll_k35 = log_loss(y_train[val_idx], np.clip(p_val_k35, 1e-4, 1.0 - 1e-4))

        print(f"Fold {fold_i + 1} Metrics (k=35): Acc={acc_k35:.5f}, AUC={auc_k35:.5f}, LogLoss={ll_k35:.5f} [k=15: {acc_k15:.5f}, k=31: {acc_k31:.5f}]")

        # Step 4g: Query test items in batches
        t_q_te = time.time()
        te_dists_list = []
        te_inds_list = []
        for s in range(0, n_test, batch_size):
            e = min(s + batch_size, n_test)
            d, ind = nn.kneighbors(X_te_comb[s:e])
            te_dists_list.append(d)
            te_inds_list.append(ind)
        te_dists = np.vstack(te_dists_list)
        te_inds = np.vstack(te_inds_list)
        print(f"Test queries ({n_test} items) completed in {time.time() - t_q_te:.2f}s.")

        # Test duplicates
        n_te_dups = int((te_dists[:, 0] <= 0.02).sum())
        total_test_duplicates_per_fold.append(n_te_dups)
        print(f"Test catalog duplicates (sim >= 0.98): {n_te_dups} / {n_test} ({n_te_dups / n_test * 100:.2f}%)")

        p_te_k35 = compute_distance_weighted_proba(te_dists, te_inds, y_train[tr_idx])
        test_p_folds[fold_i] = p_te_k35

        f_duration = time.time() - f_start
        fold_reports.append({
            "fold": fold_i + 1,
            "accuracy_k35": float(acc_k35),
            "roc_auc_k35": float(auc_k35),
            "logloss_k35": float(ll_k35),
            "accuracy_k15": float(acc_k15),
            "accuracy_k31": float(acc_k31),
            "n_val_duplicates": n_val_dups,
            "val_duplicate_agreement_rate": float(val_dup_acc),
            "n_test_duplicates": n_te_dups,
            "fold_time_seconds": float(f_duration),
        })

        # Memory cleanup between folds
        del (
            vec, svd, scaler, nn,
            X_tr_tfidf, X_val_tfidf, X_te_tfidf,
            X_tr_svd, X_val_svd, X_te_svd,
            X_tr_tab, X_val_tab, X_te_tab,
            X_tr_comb, X_val_comb, X_te_comb,
            val_dists, val_inds, te_dists, te_inds,
        )
        gc.collect()

    # Step 5: Global Evaluation & Summary
    print("\n" + "=" * 80)
    print("[5/5] COMPUTING GLOBAL OUT-OF-FOLD METRICS AND SAVING ARTIFACTS")
    print("=" * 80)

    global_pred_binary = (oof_p >= 0.5).astype(int)
    global_acc = float(accuracy_score(y_train, global_pred_binary))
    global_auc = float(roc_auc_score(y_train, oof_p))
    oof_p_clipped = np.clip(oof_p, 1e-4, 1.0 - 1e-4)
    global_ll = float(log_loss(y_train, oof_p_clipped))
    hits = int((global_pred_binary == y_train).sum())
    total_time = time.time() - start_time

    print(f"GLOBAL SEMANTIC k-NN OOF ACCURACY: {global_acc:.5f} ({hits} / {n_train})")
    print(f"GLOBAL SEMANTIC k-NN OOF ROC-AUC:  {global_auc:.5f}")
    print(f"GLOBAL SEMANTIC k-NN OOF LOG-LOSS: {global_ll:.5f}")
    print(f"TOTAL EXECUTION TIME:             {total_time:.1f}s ({total_time / 60:.2f} min)")

    # Mean test predictions across 5 folds
    final_test_p = test_p_folds.mean(axis=0).astype(np.float32)

    # Save OOF predictions
    oof_df = pd.DataFrame({
        "id": np.arange(1, n_train + 1),
        "p_used": oof_p,
        "target_condition_used": y_train,
        "seller_id": train_sellers,
        "item_id": train_ids,
    })
    oof_csv_path = OUT_DIR / "oof_semantic_knn.csv"
    oof_df.to_csv(oof_csv_path, index=False)
    print(f"Saved OOF predictions to {oof_csv_path} (shape: {oof_df.shape})")

    # Save Test probabilities
    test_df = pd.DataFrame({
        "id": np.arange(1, n_test + 1),
        "p_used": final_test_p,
        "item_id": test_ids,
    })
    test_csv_path = OUT_DIR / "test_probabilities_semantic_knn.csv"
    test_df.to_csv(test_csv_path, index=False)
    print(f"Saved Test probabilities to {test_csv_path} (shape: {test_df.shape})")

    # Catalog duplicate audit
    dup_audit = {
        "oof_near_duplicates_count": total_val_duplicates,
        "oof_near_duplicates_pct": float(total_val_duplicates / n_train * 100),
        "oof_duplicate_condition_agreement_rate": float(total_val_duplicate_correct / max(1, total_val_duplicates)),
        "mean_test_duplicates_per_fold": float(np.mean(total_test_duplicates_per_fold)),
        "test_duplicates_pct": float(np.mean(total_test_duplicates_per_fold) / n_test * 100),
    }

    # Summary report
    summary = {
        "model_name": "Semantic k-NN Non-Parametric Metric Specialist",
        "architecture": {
            "n_neighbors": 35,
            "metric": "cosine",
            "algorithm": "brute",
            "weighting": "inverse_distance (1/dist with exact duplicate resolution)",
            "svd_components": 128,
            "text_weight": 0.7,
            "tabular_weight": 0.3,
            "batch_size": batch_size,
        },
        "cv_protocol": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) grouped by seller_id",
        "zero_seller_leakage": True,
        "item_intrinsic_features_only": True,
        "n_train": n_train,
        "n_test": n_test,
        "n_tabular_features": n_base_tab + 2,
        "n_svd_features": 128,
        "total_features": n_base_tab + 2 + 128,
        "global_accuracy": global_acc,
        "global_roc_auc": global_auc,
        "global_logloss": global_ll,
        "hits": hits,
        "total_duration_seconds": total_time,
        "fold_metrics": fold_reports,
        "catalog_duplicate_audit": dup_audit,
        "output_files": {
            "oof_csv": str(oof_csv_path),
            "test_csv": str(test_csv_path),
        },
        "leak_free_audit": {
            "seller_overlap_across_folds": 0,
            "seller_identity_features": 0,
            "seller_target_encoding": 0,
            "scaler_fitted_strictly_on_train": True,
            "svd_fitted_strictly_on_train": True,
            "category_freq_fitted_strictly_on_train": True,
            "nn_index_fitted_strictly_on_train": True,
        },
    }

    summary_json_path = OUT_DIR / "summary_report.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved Summary report to {summary_json_path}")
    print("\nTraining and evaluation pipeline completed successfully!")


if __name__ == "__main__":
    main()
