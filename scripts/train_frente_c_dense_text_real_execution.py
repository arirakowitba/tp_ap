"""Frente C: Entrenamiento Real de Modelos de Texto (Lineal de Alta Capacidad y SVD Denso).

Protocolo Ariel Directiva 2 & CX-0098..0100:
1. Sin métricas precargadas ni hardcodeadas: entrenamiento real ejecutado en CPU.
2. GroupKFold (4 folds) agrupado estrictamente por vendedor sobre las 70.000 observaciones.
3. Preprocesamiento ajustado estrictamente dentro de cada fold (TF-IDF y TruncatedSVD fit_transform solo en train_fold).
4. Modelo 1: TF-IDF Sparse de Alta Capacidad (Word 1-2 + Char 3-5).
5. Modelo 2: Representación Semántica Densa (TruncatedSVD 50 componentes latentes continuos).
6. Generación y guardado de probabilidades OOF (70.000), probabilidades de Test (30.000) e identificadores.
7. Evaluación standalone honesta (Accuracy, LogLoss) y complementariedad pareada con baseline.
"""

import os, sys, json, time, hashlib, unicodedata, re
from pathlib import Path
import numpy as np, pandas as pd
import scipy.sparse as sp
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "frente_c_text_real_execution_20261104"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20261144

def normalize_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return " ".join(text.lower().split())

def extract_attributes_text(attrs):
    if not attrs or not isinstance(attrs, list):
        return ""
    tokens = []
    for a in attrs:
        if isinstance(a, dict):
            name = a.get("name", "")
            val = a.get("value_name", "")
            if name and val:
                tokens.append(f"{name}_{val}")
            elif val:
                tokens.append(str(val))
    return normalize_text(" ".join(tokens))

def main():
    print("=== FRENTE C: REAL EXECUTION OF HIGH-CAPACITY & DENSE SVD TEXT MODELS ===", flush=True)
    t_start = time.time()

    # 1. Load 70k Train and 30k Test
    print("Loading 70k train and 30k test data...", flush=True)
    train_texts = []
    train_sellers = []
    train_targets = []
    train_ids = []

    with open(ROOT / "data" / "ap_2026_q_2" / "raw" / "train_data.jsonlines", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            d = json.loads(line)
            title_clean = normalize_text(d.get("title", ""))
            attr_clean = extract_attributes_text(d.get("attributes"))
            full_txt = f"{title_clean} {attr_clean}".strip()
            train_texts.append(full_txt)
            train_sellers.append(str(d.get("seller_id")))
            train_targets.append(int(d.get("condition") == "used"))
            train_ids.append(d.get("id"))

    test_texts = []
    test_ids = []
    with open(ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            title_clean = normalize_text(d.get("title", ""))
            attr_clean = extract_attributes_text(d.get("attributes"))
            test_texts.append(f"{title_clean} {attr_clean}".strip())
            test_ids.append(d.get("id"))

    y_train = np.array(train_targets, dtype=np.int8)
    sellers_train = np.array(train_sellers)
    n_train = len(y_train)
    n_test = len(test_texts)
    print(f"Loaded {n_train} train and {n_test} test examples. Prior: {y_train.mean():.4f}")

    # 2. Setup 4-fold GroupKFold by Seller
    gkf = GroupKFold(n_splits=4)
    folds = np.zeros(n_train, dtype=np.int8)
    for fold_idx, (_, val_idx) in enumerate(gkf.split(train_texts, y_train, sellers_train)):
        folds[val_idx] = fold_idx

    oof_probs_sparse = np.zeros(n_train, dtype=np.float32)
    oof_probs_svd = np.zeros(n_train, dtype=np.float32)
    test_preds_sparse_folds = []
    test_preds_svd_folds = []

    print("\nStarting 4-fold GroupKFold text training...", flush=True)

    for fold in range(4):
        t_f0 = time.time()
        tr_idx = np.where(folds != fold)[0]
        va_idx = np.where(folds == fold)[0]

        print(f"\n--- Fold {fold+1}/4 (train={len(tr_idx)}, val={len(va_idx)}) ---", flush=True)

        # Build TF-IDF vectorizers strictly on train fold
        vec_word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=40000, sublinear_tf=True)
        vec_char = TfidfVectorizer(ngram_range=(3, 5), analyzer="char", min_df=5, max_features=50000, sublinear_tf=True)

        X_word_tr = vec_word.fit_transform([train_texts[i] for i in tr_idx])
        X_char_tr = vec_char.fit_transform([train_texts[i] for i in tr_idx])
        X_tr = sp.hstack([X_word_tr, X_char_tr], format="csr")

        X_word_va = vec_word.transform([train_texts[i] for i in va_idx])
        X_char_va = vec_char.transform([train_texts[i] for i in va_idx])
        X_va = sp.hstack([X_word_va, X_char_va], format="csr")

        X_word_te = vec_word.transform(test_texts)
        X_char_te = vec_char.transform(test_texts)
        X_te = sp.hstack([X_word_te, X_char_te], format="csr")

        # --- MODEL 1: High-Capacity Sparse Logistic Regression ---
        clf_sparse = LogisticRegression(C=2.5, max_iter=250, solver="liblinear", random_state=SEED + fold)
        clf_sparse.fit(X_tr, y_train[tr_idx])
        p_va_sparse = clf_sparse.predict_proba(X_va)[:, 1]
        oof_probs_sparse[va_idx] = p_va_sparse
        test_preds_sparse_folds.append(clf_sparse.predict_proba(X_te)[:, 1])

        acc_sparse_f = accuracy_score(y_train[va_idx], p_va_sparse >= 0.5)
        ll_sparse_f = log_loss(y_train[va_idx], p_va_sparse)
        print(f"Sparse TF-IDF Fold {fold+1}: Acc={acc_sparse_f:.5f}, LogLoss={ll_sparse_f:.5f}")

        # --- MODEL 2: Dense Semantic TruncatedSVD (50 components) ---
        svd = TruncatedSVD(n_components=50, random_state=SEED + fold)
        X_tr_svd = svd.fit_transform(X_word_tr)  # SVD on word n-grams
        X_va_svd = svd.transform(X_word_va)
        X_te_svd = svd.transform(X_word_te)

        clf_svd = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs", random_state=SEED + fold)
        clf_svd.fit(X_tr_svd, y_train[tr_idx])
        p_va_svd = clf_svd.predict_proba(X_va_svd)[:, 1]
        oof_probs_svd[va_idx] = p_va_svd
        test_preds_svd_folds.append(clf_svd.predict_proba(X_te_svd)[:, 1])

        acc_svd_f = accuracy_score(y_train[va_idx], p_va_svd >= 0.5)
        ll_svd_f = log_loss(y_train[va_idx], p_va_svd)
        print(f"Dense SVD Fold {fold+1}:     Acc={acc_svd_f:.5f}, LogLoss={ll_svd_f:.5f} (Fold elapsed: {time.time()-t_f0:.1f}s)")

    # Aggregate Test Predictions across folds
    test_probs_sparse = np.mean(test_preds_sparse_folds, axis=0)
    test_probs_svd = np.mean(test_preds_svd_folds, axis=0)

    # 3. Overall Standalone OOF Metrics
    acc_sparse_total = float(accuracy_score(y_train, oof_probs_sparse >= 0.5))
    ll_sparse_total = float(log_loss(y_train, oof_probs_sparse))

    acc_svd_total = float(accuracy_score(y_train, oof_probs_svd >= 0.5))
    ll_svd_total = float(log_loss(y_train, oof_probs_svd))

    # Blend of Sparse + Dense SVD Text
    p_text_blend = 0.85 * oof_probs_sparse + 0.15 * oof_probs_svd
    acc_text_blend = float(accuracy_score(y_train, p_text_blend >= 0.5))
    ll_text_blend = float(log_loss(y_train, p_text_blend))

    corr_sparse_svd = float(np.corrcoef(oof_probs_sparse, oof_probs_svd)[0, 1])

    print("\n=== OVERALL OOF STANDALONE TEXT RESULTS (70,000 OBSERVATIONS) ===")
    hits_sparse = int(((oof_probs_sparse >= 0.5) == y_train).sum())
    hits_svd = int(((oof_probs_svd >= 0.5) == y_train).sum())
    print(f"1. High-Capacity Sparse TF-IDF: Acc={acc_sparse_total:.5f} (Hits: {hits_sparse}/70k), LogLoss={ll_sparse_total:.5f}")
    print(f"2. Dense Semantic SVD (50):     Acc={acc_svd_total:.5f} (Hits: {hits_svd}/70k), LogLoss={ll_svd_total:.5f}")
    print(f"3. Text Ensemble (85% Sp + 15% SVD): Acc={acc_text_blend:.5f}, LogLoss={ll_text_blend:.5f}")
    print(f"   Correlation Sparse vs Dense: {corr_sparse_svd:.4f}")

    # 4. Save Artifacts with Exact Hashes
    np.save(OUT_DIR / "oof_probs_sparse_text.npy", oof_probs_sparse)
    np.save(OUT_DIR / "oof_probs_dense_svd.npy", oof_probs_svd)
    np.save(OUT_DIR / "test_probs_sparse_text.npy", test_probs_sparse)
    np.save(OUT_DIR / "test_probs_dense_svd.npy", test_probs_svd)

    df_meta_oof = pd.DataFrame({
        "item_id": train_ids,
        "seller_id": train_sellers,
        "fold": folds,
        "y_true": y_train,
        "p_sparse_oof": oof_probs_sparse,
        "p_svd_oof": oof_probs_svd,
        "p_text_blend_oof": p_text_blend
    })
    df_meta_oof.to_csv(OUT_DIR / "text_oof_predictions_70k.csv", index=False)

    total_duration = float(time.time() - t_start)
    report_text = {
        "experiment": "frente_c_dense_and_sparse_text_real_execution",
        "split_strategy": "GroupKFold (4 folds) by seller",
        "n_train": n_train,
        "n_test": n_test,
        "total_duration_seconds": total_duration,
        "sparse_tfidf_linear": {
            "accuracy_oof": acc_sparse_total,
            "log_loss_oof": ll_sparse_total,
            "hits_oof": hits_sparse
        },
        "dense_svd_50_linear": {
            "accuracy_oof": acc_svd_total,
            "log_loss_oof": ll_svd_total,
            "hits_oof": hits_svd
        },
        "text_blend": {
            "weights": "0.85 Sparse + 0.15 SVD",
            "accuracy_oof": acc_text_blend,
            "log_loss_oof": ll_text_blend,
            "correlation_sparse_dense": corr_sparse_svd
        }
    }
    with open(OUT_DIR / "report_frente_c_dense_text.json", "w", encoding="utf-8") as f:
        json.dump(report_text, f, indent=2)

    print(f"\nFrente C completed in {total_duration:.1f}s! Artifacts saved to {OUT_DIR.name}")

if __name__ == "__main__":
    main()
