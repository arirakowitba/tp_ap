"""High-Dimensional Sub-Word N-Gram Sparse Linear Models (Ridge + Logistic Regression).
Trains 5-fold Stratified K-Fold across 250,000 word & character n-grams.
Completely leak-free (zero seller-id target encoding).
"""
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from scipy.special import expit
import os

start_time = time.time()
print("=== PARALLEL STREAM 1: HIGH-DIMENSIONAL SPARSE LINEAR MODELS ===")

# 1. Load data
print("Loading train and test datasets...")
with open("data/ap_2026_q_2/raw/train_data.jsonlines", encoding="utf-8") as f:
    train_data = [json.loads(line) for line in f]

with open("data/ap_2026_q_2/raw/test_data.jsonlines", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f]

train_titles = [d.get("title", "") for d in train_data]
test_titles = [d.get("title", "") for d in test_data]
y_train = np.array([1 if d["condition"] == "new" else 0 for d in train_data], dtype=np.int8)

print(f"Train samples: {len(train_titles)}, Test samples: {len(test_titles)}")

# 2. Build massive TF-IDF Feature Union:
# Word n-grams (1, 3) + Character n-grams (3, 5)
print("Building 250,000-feature word + character TF-IDF union...")
word_vec = TfidfVectorizer(
    ngram_range=(1, 3),
    min_df=3,
    max_features=120000,
    sublinear_tf=True,
    strip_accents="unicode"
)

char_vec = TfidfVectorizer(
    ngram_range=(3, 5),
    analyzer="char_wb",
    min_df=5,
    max_features=130000,
    sublinear_tf=True,
    strip_accents="unicode"
)

union = FeatureUnion([
    ("word", word_vec),
    ("char", char_vec)
], n_jobs=8)

X_train_sparse = union.fit_transform(train_titles)
X_test_sparse = union.transform(test_titles)
print(f"Sparse Feature Matrix: {X_train_sparse.shape}")

# 3. 5-Fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lr = np.zeros(len(train_titles), dtype=np.float32)
test_lr = np.zeros(len(test_titles), dtype=np.float32)

oof_ridge = np.zeros(len(train_titles), dtype=np.float32)
test_ridge = np.zeros(len(test_titles), dtype=np.float32)

print("\nTraining 5-Fold Logistic Regression & Ridge Classifier...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_sparse, y_train)):
    f_start = time.time()
    X_tr, y_tr = X_train_sparse[train_idx], y_train[train_idx]
    X_va, y_val = X_train_sparse[val_idx], y_train[val_idx]
    
    # Model 1: Logistic Regression L2 (SAGA / L-BFGS)
    clf_lr = LogisticRegression(C=2.0, max_iter=200, solver="lbfgs", n_jobs=4, random_state=42+fold)
    clf_lr.fit(X_tr, y_tr)
    val_probs_lr = clf_lr.predict_proba(X_va)[:, 1]
    oof_lr[val_idx] = val_probs_lr
    test_lr += clf_lr.predict_proba(X_test_sparse)[:, 1] / 5.0
    
    # Model 2: Ridge Classifier (calibrated via decision_function sigmoid)
    clf_ridge = RidgeClassifier(alpha=1.5, random_state=42+fold)
    clf_ridge.fit(X_tr, y_tr)
    # Calibrate Ridge decision function to pseudo-probabilities via Platt scaling
    val_df = clf_ridge.decision_function(X_va)
    # Fit simple 1D logistic on validation decision function
    calibrator = LogisticRegression(C=1e5, solver="lbfgs")
    calibrator.fit(val_df.reshape(-1, 1), y_val)
    oof_ridge[val_idx] = calibrator.predict_proba(val_df.reshape(-1, 1))[:, 1]
    
    test_df = clf_ridge.decision_function(X_test_sparse)
    test_ridge += calibrator.predict_proba(test_df.reshape(-1, 1))[:, 1] / 5.0
    
    acc_lr = accuracy_score(y_val, (val_probs_lr >= 0.5).astype(int))
    acc_rg = accuracy_score(y_val, (oof_ridge[val_idx] >= 0.5).astype(int))
    print(f"Fold {fold+1}/5 ({time.time()-f_start:.1f}s) | LR Acc: {acc_lr:.5f} | Ridge Acc: {acc_rg:.5f}")

# Overall OOF Metrics
acc_lr_tot = accuracy_score(y_train, (oof_lr >= 0.5).astype(int))
acc_rg_tot = accuracy_score(y_train, (oof_ridge >= 0.5).astype(int))
auc_lr = roc_auc_score(y_train, oof_lr)
auc_rg = roc_auc_score(y_train, oof_ridge)

blend_oof = 0.5 * oof_lr + 0.5 * oof_ridge
blend_test = 0.5 * test_lr + 0.5 * test_ridge
acc_blend = accuracy_score(y_train, (blend_oof >= 0.5).astype(int))

print("\n=== SPARSE LINEAR OVERALL RESULTS ===")
print(f"Logistic Regression OOF Acc: {acc_lr_tot:.5f} (AUC: {auc_lr:.5f})")
print(f"Ridge Classifier    OOF Acc: {acc_rg_tot:.5f} (AUC: {auc_rg:.5f})")
print(f"Sparse Blend        OOF Acc: {acc_blend:.5f} ({int(acc_blend*70000)}/70000 correct)")

# Save predictions
os.makedirs("results/deep_parallel_20261109", exist_ok=True)
np.savez_compressed(
    "results/deep_parallel_20261109/sparse_linear_predictions.npz",
    oof_lr=oof_lr,
    test_lr=test_lr,
    oof_ridge=oof_ridge,
    test_ridge=test_ridge,
    blend_oof=blend_oof,
    blend_test=blend_test
)
print(f"Saved artifacts to results/deep_parallel_20261109/sparse_linear_predictions.npz ({time.time()-start_time:.1f}s)")
