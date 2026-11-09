"""Intensive Multi-Model Specialist Training (AP-2026-Q-2).

Trains 5 orthogonal high-capacity models on the full 70,000 dataset:
1. High-Capacity LightGBM (1000 trees, depth 9, 63 leaves)
2. High-Capacity CatBoost (1000 iterations, depth 8)
3. High-Dimensional Sparse Ridge Text (70,000 n-grams)
4. Calibrated Logistic Regression L2 on Enriched Text
5. Complement Naive Bayes on Enriched Text

Saves full test consensus predictions to results/intensive_training_20261109/.
"""

import json
import time
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
from catboost import CatBoostClassifier

ROOT = Path(".")
DATA_DIR = ROOT / "data/ap_2026_q_2/raw"
OUT_DIR = ROOT / "results/intensive_training_20261109"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_enriched(row):
    parts = [str(row.get("title") or "")]
    w = row.get("warranty")
    if w and str(w).strip():
        parts.append(f"garantia_{w}")
    tags = row.get("tags") or []
    for t in tags:
        parts.append(f"tag_{t}")
    attrs = row.get("attributes") or []
    for a in attrs:
        if isinstance(a, dict):
            name = a.get("id") or a.get("name") or ""
            val = a.get("value_name") or a.get("value_id") or ""
            if name or val:
                parts.append(f"attr_{name}_{val}")
    return " ".join(parts).lower()


def extract_features(df, cat_medians=None):
    feats = pd.DataFrame(index=df.index)
    titles = df["title"].fillna("").astype(str)
    titles_lower = titles.str.lower()

    # Title length metrics
    feats["title_len"] = titles.str.len()
    feats["title_words"] = titles.str.split().str.len()
    feats["title_digits"] = titles.str.count(r"\d")
    feats["title_caps"] = titles.str.findall(r"[A-Z]").str.len()

    # Price features
    price = df["price"].fillna(0).astype(float).values
    feats["log_price"] = np.log1p(np.maximum(price, 0))

    if cat_medians is not None:
        cat_med = df["category_id"].map(cat_medians).fillna(np.median(price)).values
        feats["price_ratio_cat"] = np.log1p(price) - np.log1p(cat_med)

    # Condition regex cues
    feats["cue_como_nuevo"] = titles_lower.str.contains(r"\bcomo nuevo\b").astype(float)
    feats["cue_impecable"] = titles_lower.str.contains(r"\bimpecable\b").astype(float)
    feats["cue_excelente_estado"] = titles_lower.str.contains(r"\bexcelente estado\b").astype(float)
    feats["cue_muy_buen_estado"] = titles_lower.str.contains(r"\bmuy buen estado\b").astype(float)
    feats["cue_para_repuesto"] = titles_lower.str.contains(r"\bpara repuesto\b").astype(float)
    feats["cue_usado"] = titles_lower.str.contains(r"\busad[oa]\b").astype(float)
    feats["cue_caja_cerrada"] = titles_lower.str.contains(r"\bcaja cerrada\b").astype(float)
    feats["cue_nuevo_sellado"] = titles_lower.str.contains(r"\bnuevo sellado\b").astype(float)
    feats["cue_garantia_fabrica"] = titles_lower.str.contains(r"\bgarantia de fabrica\b").astype(float)

    # Tags
    tags_str = df["tags"].apply(lambda t: " ".join(t) if isinstance(t, list) else "")
    feats["tag_free_relist"] = tags_str.str.contains("free_relist").astype(float)
    feats["tag_dragged_visits"] = tags_str.str.contains("dragged_visits").astype(float)
    feats["tag_good_thumb"] = tags_str.str.contains("good_quality_thumbnail").astype(float)

    # Listing type & quantities
    feats["is_gold"] = (df["listing_type_id"] == "gold_special").astype(float)
    feats["is_bronze"] = (df["listing_type_id"] == "bronze").astype(float)
    feats["initial_quantity"] = df["initial_quantity"].fillna(1).astype(float)
    feats["sold_quantity"] = df["sold_quantity"].fillna(0).astype(float)
    feats["sold_ratio"] = feats["sold_quantity"] / (feats["initial_quantity"] + 1)

    return feats.fillna(0)


def main():
    print("=" * 85)
    print("STARTING INTENSIVE SPECIALIST MULTI-MODEL TRAINING (16 CORES)")
    print("=" * 85, flush=True)

    t0 = time.time()

    # 1. Load Data
    print("\n[1/5] Loading 70k Train and 30k Test Data...")
    df_train = pd.read_json(DATA_DIR / "train_data.jsonlines", lines=True)
    df_test = pd.read_json(DATA_DIR / "test_data.jsonlines", lines=True)
    y_train = (df_train["condition"] == "used").astype(np.int8).values
    print(f"Train size: {len(df_train)} | Test size: {len(df_test)}")

    # 2. Extract Tabular Features
    print("\n[2/5] Engineering Tabular Features...")
    cat_medians = df_train.groupby("category_id")["price"].median().to_dict()
    X_tab_train = extract_features(df_train, cat_medians)
    X_tab_test = extract_features(df_test, cat_medians)
    print(f"Tabular features dimension: {X_tab_train.shape[1]}")

    # 3. Extract High-Dimensional Enriched Text Features (70,000 n-grams)
    print("\n[3/5] Extracting Enriched Multimodal Text Representations (70k n-grams)...")
    texts_train = df_train.apply(build_enriched, axis=1)
    texts_test = df_test.apply(build_enriched, axis=1)

    vec_word = TfidfVectorizer(max_features=40000, ngram_range=(1, 2), sublinear_tf=True)
    vec_char = TfidfVectorizer(max_features=30000, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)

    print("  Fitting TF-IDF Word Vectorizer...")
    w_tr = vec_word.fit_transform(texts_train)
    w_te = vec_word.transform(texts_test)

    print("  Fitting TF-IDF Char Vectorizer...")
    c_tr = vec_char.fit_transform(texts_train)
    c_te = vec_char.transform(texts_test)

    X_text_tr = sparse.hstack([w_tr, c_tr], format="csr")
    X_text_te = sparse.hstack([w_te, c_te], format="csr")
    print(f"Sparse text dimension: {X_text_tr.shape[1]}")

    # 4. Train Orthogonal Models
    print("\n[4/5] Training Orthogonal Specialist Models...")

    # Model 1: High-Capacity Sparse Ridge (L2 text classifier)
    print("  Training Model 1: High-Capacity Sparse Ridge Text (alpha=1.5)...")
    m_ridge = RidgeClassifier(alpha=1.5, random_state=42)
    m_ridge.fit(X_text_tr, y_train)
    p_ridge_test = 1.0 / (1.0 + np.exp(-m_ridge.decision_function(X_text_te)))

    # Model 2: Logistic Regression L2 on Text
    print("  Training Model 2: Calibrated Logistic Regression L2 (C=1.0)...")
    m_lr = LogisticRegression(C=1.0, max_iter=250, random_state=42, n_jobs=-1)
    m_lr.fit(X_text_tr, y_train)
    p_lr_test = m_lr.predict_proba(X_text_te)[:, 1]

    # Model 3: Complement Naive Bayes on Text
    print("  Training Model 3: Complement Naive Bayes (alpha=0.5)...")
    m_cnb = ComplementNB(alpha=0.5)
    m_cnb.fit(X_text_tr, y_train)
    p_cnb_test = m_cnb.predict_proba(X_text_te)[:, 1]

    # Model 4: High-Capacity LightGBM on Tabular Features
    print("  Training Model 4: High-Capacity LightGBM (1000 estimators, depth 9, 63 leaves)...")
    m_lgb = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=9,
        colsample_bytree=0.80,
        subsample=0.85,
        subsample_freq=1,
        reg_alpha=0.5,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    m_lgb.fit(X_tab_train, y_train)
    p_lgb_test = m_lgb.predict_proba(X_tab_test)[:, 1]

    # Model 5: High-Capacity CatBoost on Tabular Features
    print("  Training Model 5: High-Capacity CatBoost (1000 iterations, depth 8)...")
    m_cat = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.04,
        depth=8,
        l2_leaf_reg=5.0,
        random_seed=42,
        thread_count=-1,
        verbose=0
    )
    m_cat.fit(X_tab_train, y_train)
    p_cat_test = m_cat.predict_proba(X_tab_test)[:, 1]

    # 5. Compute Unified Multi-Model Consensus on Test
    print("\n[5/5] Computing Multi-Model Certainty Index & Cross-Referencing...")
    probs_matrix = np.column_stack([p_ridge_test, p_lr_test, p_cnb_test, p_lgb_test, p_cat_test])
    mean_p = np.mean(probs_matrix, axis=1)
    min_p = np.min(probs_matrix, axis=1)
    std_p = np.std(probs_matrix, axis=1)

    # Save predictions
    npz_out = OUT_DIR / "intensive_specialist_test_predictions.npz"
    np.savez_compressed(
        npz_out,
        p_ridge_test=p_ridge_test,
        p_lr_test=p_lr_test,
        p_cnb_test=p_cnb_test,
        p_lgb_test=p_lgb_test,
        p_cat_test=p_cat_test,
        mean_p=mean_p,
        min_p=min_p,
        std_p=std_p
    )
    print(f"  Saved test predictions to: {npz_out}")

    # Compare with EVT-132
    p_evt = ROOT / "submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv"
    df_evt = pd.read_csv(p_evt)
    y_evt = (df_evt["condition"] == "used").astype(int).values

    # Find items where EVT says 'new' (0), but ALL 5 models unanimously say 'used' with extreme confidence (min_p >= 0.80)
    unanimous_errors = (y_evt == 0) & (min_p >= 0.80) & (std_p <= 0.08)
    n_unanimous = np.sum(unanimous_errors)
    print(f"\nTotal Test Items where EVT-132 said NEW, but ALL 5 Models Unanimously say USED (min_p >= 0.80): {n_unanimous}")

    df_errors = df_test[unanimous_errors].copy()
    df_errors["min_p"] = min_p[unanimous_errors]
    df_errors["mean_p"] = mean_p[unanimous_errors]
    df_errors["std_p"] = std_p[unanimous_errors]
    df_errors = df_errors.sort_values(by=["min_p", "mean_p"], ascending=[False, False])

    csv_out = OUT_DIR / "unanimous_evt132_error_candidates.csv"
    df_errors[["id", "title", "price", "category_id", "min_p", "mean_p", "std_p"]].to_csv(csv_out, index=False)
    print(f"  Saved candidate errors to: {csv_out}")

    print("\nTop 15 Absolute Certainty Fixes:")
    for idx, (_, r) in enumerate(df_errors.head(15).iterrows(), 1):
        print(f"  #{idx:2d} | ID {r['id']:15s} | Min_P: {r['min_p']:.3f} | Mean_P: {r['mean_p']:.3f} | {r['title'][:65]}")

    print(f"\nIntensive training and cross-verification completed in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
