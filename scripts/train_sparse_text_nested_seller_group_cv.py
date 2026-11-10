#!/usr/bin/env python3
"""Outer StratifiedGroupKFold Seller CV for Sparse Text & Quad-Titan Synthesis.

Implements Lineament 5 of CX-0143:
- High-capacity sparse n-gram text modeling (word 1-2 + char_wb 3-5, ~150k features).
- Strictly nested outer StratifiedGroupKFold(n_splits=5) grouped by seller_id.
- Feature extraction and LogisticRegression fit strictly inside outer-train.
- Paired blend with Tri-Titan intrinsic tree ensemble on true outer OOF.
- Seller-group bootstrap (1000 resamples, 95% CI).
- Full 70,000 item out-of-fold evaluation and 30,000 test set predictions.
"""

import os
import sys
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def main():
    print("=" * 80)
    print("SPARSE TEXT CLASSIFIER & QUAD-TITAN ENSEMBLE (OUTER SELLER-GROUP CV)")
    print("Strict Zero Seller Overlap Nested Protocol (CX-0143 Lineament 5)")
    print("=" * 80)

    start_total = time.time()
    out_dir = Path("results/sparse_text_seller_group_cv_20261110")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    train_path = "data/ap_2026_q_2/raw/train_data.jsonlines"
    test_path = "data/ap_2026_q_2/raw/test_data.jsonlines"

    print("\n[1/4] Loading train and test data...", flush=True)
    with open(train_path, "r", encoding="utf-8") as f:
        train_rows = [json.loads(line) for line in f]
    with open(test_path, "r", encoding="utf-8") as f:
        test_rows = [json.loads(line) for line in f]

    n_train = len(train_rows)
    n_test = len(test_rows)
    print(f"Loaded {n_train} train rows and {n_test} test rows.", flush=True)

    y_train = np.array([1 if r.get("condition") == "used" else 0 for r in train_rows], dtype=int)
    train_sellers = np.array([str(r.get("seller_id") or "UNKNOWN") for r in train_rows])
    train_ids = np.array([r.get("id") for r in train_rows])
    test_ids = np.array([r.get("id") for r in test_rows])

    # Build rich text corpus
    train_corpus = []
    for r in train_rows:
        title = str(r.get("title") or "").strip()
        warr = str(r.get("warranty") or "").strip()
        attrs = [str(a.get("value_name") or a.get("name") or "") for a in (r.get("attributes") or []) if isinstance(a, dict)]
        train_corpus.append(f"{title} WARRANTY {warr} ATTR {' '.join(attrs)}")

    test_corpus = []
    for r in test_rows:
        title = str(r.get("title") or "").strip()
        warr = str(r.get("warranty") or "").strip()
        attrs = [str(a.get("value_name") or a.get("name") or "") for a in (r.get("attributes") or []) if isinstance(a, dict)]
        test_corpus.append(f"{title} WARRANTY {warr} ATTR {' '.join(attrs)}")

    del train_rows, test_rows
    gc.collect()

    # Load Tri-Titan tree OOF
    tri_titan_oof_path = "results/seller_group_validation_20261110/oof_predictions_all_models.csv"
    print(f"Loading Tri-Titan tree OOF from {tri_titan_oof_path}...", flush=True)
    df_tri_titan = pd.read_csv(tri_titan_oof_path)
    p_tri_titan_oof = df_tri_titan["tri_titan_arm_a"].values
    acc_tri_titan = accuracy_score(y_train, (p_tri_titan_oof >= 0.5).astype(int))
    ll_tri_titan = log_loss(y_train, p_tri_titan_oof)
    print(f"Loaded Tri-Titan OOF: Acc = {acc_tri_titan:.5f}, LogLoss = {ll_tri_titan:.5f}")

    # 2. Outer StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261110)
    outer_splits = list(sgkf.split(train_corpus, y_train, groups=train_sellers))

    oof_p_text = np.zeros(n_train, dtype=float)
    fold_reports = []

    print("\n[2/4] Executing 5-Fold Nested Sparse Text Training Loop...", flush=True)
    for fold_i, (tr_idx, val_idx) in enumerate(outer_splits):
        f_start = time.time()
        print(f"\n--- FOLD {fold_i + 1} / 5 ---")
        tr_s = set(train_sellers[tr_idx])
        val_s = set(train_sellers[val_idx])
        assert len(tr_s & val_s) == 0, f"Seller overlap in fold {fold_i + 1}"
        print(f"Outer train: {len(tr_idx)} items ({len(tr_s)} sellers), Outer val: {len(val_idx)} items ({len(val_s)} sellers)")

        t_feat = time.time()
        # Word TF-IDF
        vec_word = TfidfVectorizer(ngram_range=(1, 2), max_features=75000, sublinear_tf=True, min_df=2)
        X_tr_w = vec_word.fit_transform([train_corpus[i] for i in tr_idx])
        X_val_w = vec_word.transform([train_corpus[i] for i in val_idx])

        # Char TF-IDF
        vec_char = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=75000, sublinear_tf=True, min_df=3)
        X_tr_c = vec_char.fit_transform([train_corpus[i] for i in tr_idx])
        X_val_c = vec_char.transform([train_corpus[i] for i in val_idx])

        X_tr = sparse.hstack([X_tr_w, X_tr_c], format="csr")
        X_val = sparse.hstack([X_val_w, X_val_c], format="csr")
        feat_time = time.time() - t_feat
        print(f"Features extracted in {feat_time:.1f}s. Sparse matrix shape: {X_tr.shape}")

        t_train = time.time()
        clf = LogisticRegression(C=2.0, max_iter=200, solver="saga", random_state=42)
        clf.fit(X_tr, y_train[tr_idx])
        p_val_fold = clf.predict_proba(X_val)[:, 1]
        train_time = time.time() - t_train
        print(f"Classifier fit in {train_time:.1f}s.")

        oof_p_text[val_idx] = p_val_fold

        acc_f = accuracy_score(y_train[val_idx], (p_val_fold >= 0.5).astype(int))
        ll_f = log_loss(y_train[val_idx], p_val_fold)
        auc_f = roc_auc_score(y_train[val_idx], p_val_fold)
        print(f"Fold {fold_i + 1} Sparse Text: Acc = {acc_f:.5f}, LogLoss = {ll_f:.5f}, AUC = {auc_f:.5f}")

        # Fold blend with Tri-Titan
        p_tri_val = p_tri_titan_oof[val_idx]
        p_blend_val = 0.35 * p_val_fold + 0.65 * p_tri_val
        acc_b = accuracy_score(y_train[val_idx], (p_blend_val >= 0.5).astype(int))
        ll_b = log_loss(y_train[val_idx], p_blend_val)
        delta_acc = acc_b - accuracy_score(y_train[val_idx], (p_tri_val >= 0.5).astype(int))
        print(f"Fold {fold_i + 1} Quad-Titan Blend: Acc = {acc_b:.5f}, LogLoss = {ll_b:.5f} (Δ vs Trees = {delta_acc:+.5f})")

        fold_reports.append({
            "fold": fold_i + 1,
            "text_acc": acc_f,
            "text_ll": ll_f,
            "text_auc": auc_f,
            "quad_acc": acc_b,
            "quad_ll": ll_b,
            "delta_acc_vs_trees": delta_acc,
            "time_s": time.time() - f_start,
        })

        del X_tr_w, X_val_w, X_tr_c, X_val_c, X_tr, X_val, clf
        gc.collect()

    # 3. Global 70,000 Item OOF Metrics & Blend Optimization
    print("\n" + "=" * 80)
    print("[3/4] GLOBAL 70,000-ITEM OUT-OF-FOLD EVALUATION & BLEND SYNTHESIS")
    print("=" * 80)

    acc_text_global = accuracy_score(y_train, (oof_p_text >= 0.5).astype(int))
    ll_text_global = log_loss(y_train, oof_p_text)
    auc_text_global = roc_auc_score(y_train, oof_p_text)
    hits_text = int(np.sum((oof_p_text >= 0.5).astype(int) == y_train))
    print(f"Sparse Text Standalone: Acc = {acc_text_global:.5f} ({hits_text}/{n_train}), LogLoss = {ll_text_global:.5f}, ROC-AUC = {auc_text_global:.5f}")

    # Search optimal convex weight on true OOF
    print("\nEvaluating convex combinations: w * Text + (1 - w) * Tri-Titan Trees:")
    best_w = 0.35
    best_acc = 0.0
    best_ll = 1.0
    blend_results = []
    for w in np.linspace(0.0, 1.0, 21):
        p_comb = w * oof_p_text + (1.0 - w) * p_tri_titan_oof
        pred_comb = (p_comb >= 0.5).astype(int)
        acc_c = accuracy_score(y_train, pred_comb)
        ll_c = log_loss(y_train, p_comb)
        auc_c = roc_auc_score(y_train, p_comb)
        hits_c = int(np.sum(pred_comb == y_train))
        delta_hits = hits_c - int(np.sum((p_tri_titan_oof >= 0.5).astype(int) == y_train))
        blend_results.append({
            "w_text": round(float(w), 2),
            "accuracy": float(acc_c),
            "logloss": float(ll_c),
            "roc_auc": float(auc_c),
            "hits": hits_c,
            "delta_hits_vs_trees": delta_hits,
        })
        if acc_c > best_acc:
            best_acc = acc_c
            best_w = round(float(w), 2)
            best_ll = ll_c
        print(f"  w_text = {w:4.2f}: Acc = {acc_c:.5f} ({hits_c}/{n_train}), LogLoss = {ll_c:.5f}, ROC-AUC = {auc_c:.5f} (ΔHits = {delta_hits:+d})")

    # Selected Quad-Titan Synthesis
    p_quad_oof = best_w * oof_p_text + (1.0 - best_w) * p_tri_titan_oof
    pred_quad_oof = (p_quad_oof >= 0.5).astype(int)
    quad_acc = accuracy_score(y_train, pred_quad_oof)
    quad_ll = log_loss(y_train, p_quad_oof)
    quad_auc = roc_auc_score(y_train, p_quad_oof)
    quad_hits = int(np.sum(pred_quad_oof == y_train))
    tree_hits = int(np.sum((p_tri_titan_oof >= 0.5).astype(int) == y_train))

    # Paired fixes and breaks vs pure trees
    correct_tree = ((p_tri_titan_oof >= 0.5).astype(int) == y_train)
    correct_quad = (pred_quad_oof == y_train)
    fixes_quad = int(np.sum((~correct_tree) & correct_quad))
    breaks_quad = int(np.sum(correct_tree & (~correct_quad)))
    net_fixes = fixes_quad - breaks_quad

    print("\n" + "*" * 80)
    print(f"SELECTED QUAD-TITAN ENSEMBLE (w_text = {best_w}):")
    print(f"  Accuracy: {quad_acc:.5f} ({quad_hits} / {n_train})")
    print(f"  LogLoss:  {quad_ll:.5f}")
    print(f"  ROC-AUC:  {quad_auc:.5f}")
    print(f"  PAIRED GAIN VS TREES: Fixes = {fixes_quad}, Breaks = {breaks_quad}, Net = {net_fixes:+d} ({quad_acc - acc_tri_titan:+.5f} ΔAcc)")
    print("*" * 80)

    # Seller-Group Bootstrap (1000 resamples)
    print("\nComputing Seller-Group Bootstrap on Quad-Titan (1000 resamples, 95% CI)...", flush=True)
    rng_bs = np.random.default_rng(20261110)
    u_sellers_all = np.unique(train_sellers)
    seller_map = {}
    for i, s_id in enumerate(train_sellers):
        seller_map.setdefault(s_id, []).append(i)

    bs_accs = []
    for _ in range(1000):
        sample_sellers = rng_bs.choice(u_sellers_all, size=len(u_sellers_all), replace=True)
        bs_idx = []
        for s_id in sample_sellers:
            bs_idx.extend(seller_map[s_id])
        bs_accs.append(accuracy_score(y_train[bs_idx], pred_quad_oof[bs_idx]))

    ci_low = float(np.percentile(bs_accs, 2.5))
    ci_high = float(np.percentile(bs_accs, 97.5))
    print(f"  Quad-Titan Bootstrap 95% CI: [{ci_low:.5f}, {ci_high:.5f}]")

    # 4. Fit Full Sparse Text Model on 100% Train & Generate Test Predictions
    print("\n[4/4] Fitting Full Sparse Text Model on 100% of Train for Test Inference...", flush=True)
    t_full = time.time()
    vec_word_full = TfidfVectorizer(ngram_range=(1, 2), max_features=75000, sublinear_tf=True, min_df=2)
    X_full_w = vec_word_full.fit_transform(train_corpus)
    X_test_w = vec_word_full.transform(test_corpus)

    vec_char_full = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=75000, sublinear_tf=True, min_df=3)
    X_full_c = vec_char_full.fit_transform(train_corpus)
    X_test_c = vec_char_full.transform(test_corpus)

    X_full = sparse.hstack([X_full_w, X_full_c], format="csr")
    X_test = sparse.hstack([X_test_w, X_test_c], format="csr")

    clf_full = LogisticRegression(C=2.0, max_iter=200, solver="saga", random_state=42)
    clf_full.fit(X_full, y_train)
    p_test_text = clf_full.predict_proba(X_test)[:, 1]
    print(f"Full text model fit and test projected in {time.time() - t_full:.1f}s.")

    # Load Tri-Titan test predictions
    tri_test_path = "results/seller_group_validation_20261110/test_probabilities_all_models.csv"
    df_tri_test = pd.read_csv(tri_test_path)
    p_tri_test = df_tri_test["p_used_tri_titan_arm_a"].values

    # Quad-Titan test blend
    p_quad_test = best_w * p_test_text + (1.0 - best_w) * p_tri_test

    # Save OOF and Test artifacts
    oof_df = pd.DataFrame({
        "id": train_ids,
        "seller_id": train_sellers,
        "target_condition_used": y_train,
        "p_used_sparse_text": oof_p_text,
        "p_used_tri_titan_trees": p_tri_titan_oof,
        "p_used_quad_titan": p_quad_oof,
        "pred_label_quad_titan": np.where(p_quad_oof >= 0.5, "used", "new"),
    })
    oof_file = out_dir / "oof_predictions_sparse_text_and_quad_titan.csv"
    oof_df.to_csv(oof_file, index=False)

    test_df = pd.DataFrame({
        "id": test_ids,
        "p_used_sparse_text": p_test_text,
        "p_used_tri_titan_trees": p_tri_test,
        "p_used_quad_titan": p_quad_test,
        "pred_label_quad_titan": np.where(p_quad_test >= 0.5, "used", "new"),
    })
    test_file = out_dir / "test_probabilities_quad_titan.csv"
    test_df.to_csv(test_file, index=False)

    summary_file = out_dir / "summary_report.json"
    report = {
        "title": "Sparse Text Classifier and Quad-Titan Ensemble (Outer Seller-Group CV)",
        "protocol": "CX-0143 Lineament 5 (Strict Nested Seller-Group Protocol)",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sparse_text_standalone": {
            "accuracy": float(acc_text_global),
            "logloss": float(ll_text_global),
            "roc_auc": float(auc_text_global),
            "hits": hits_text,
            "total": n_train,
        },
        "tri_titan_trees_baseline": {
            "accuracy": float(acc_tri_titan),
            "logloss": float(ll_tri_titan),
            "hits": tree_hits,
            "total": n_train,
        },
        "quad_titan_ensemble": {
            "optimal_w_text": best_w,
            "accuracy": float(quad_acc),
            "logloss": float(quad_ll),
            "roc_auc": float(quad_auc),
            "hits": quad_hits,
            "total": n_train,
            "paired_fixes_vs_trees": fixes_quad,
            "paired_breaks_vs_trees": breaks_quad,
            "paired_net_fixes_vs_trees": net_fixes,
            "delta_accuracy_vs_trees": float(quad_acc - acc_tri_titan),
            "bootstrap_95_ci": [ci_low, ci_high],
        },
        "blend_grid": blend_results,
        "fold_reports": fold_reports,
        "artifacts": {
            "oof_predictions": str(oof_file),
            "test_probabilities": str(test_file),
        }
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved artifacts:")
    print(f"  OOF predictions:    {oof_file}")
    print(f"  Test probabilities: {test_file}")
    print(f"  Summary report:     {summary_file}")
    print("\n" + "=" * 80)
    print(f"QUAD-TITAN PROTOCOL COMPLETED IN {time.time() - start_total:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
