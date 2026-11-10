"""Deep Multi-Layer Perceptron (MLP) Neural Network Specialist.
Trains a 3-layer neural network (512 -> 256 -> 128) with Adam optimizer,
early stopping, and Stratified 5-Fold CV on 200 TruncatedSVD components + normalized price features.
Strictly leak-free (zero seller target encoding).
"""
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import os

start_time = time.time()
print("=== PARALLEL STREAM 2: DEEP MULTI-LAYER PERCEPTRON (MLP) ===")

# 1. Load data
print("Loading train and test datasets...")
with open("data/ap_2026_q_2/raw/train_data.jsonlines", encoding="utf-8") as f:
    train_data = [json.loads(line) for line in f]

with open("data/ap_2026_q_2/raw/test_data.jsonlines", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f]

train_titles = [d.get("title", "") for d in train_data]
test_titles = [d.get("title", "") for d in test_data]
y_train = np.array([1 if d["condition"] == "new" else 0 for d in train_data], dtype=np.int8)

train_prices = np.array([float(d.get("price", 0)) for d in train_data])
test_prices = np.array([float(d.get("price", 0)) for d in test_data])

train_cats = [d.get("category_id", "") for d in train_data]
test_cats = [d.get("category_id", "") for d in test_data]

# 2. Extract 200 TruncatedSVD Text Features
print("Computing 200-component TruncatedSVD over 100,000 TF-IDF features...")
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100000, sublinear_tf=True)
X_all_text = tfidf.fit_transform(train_titles + test_titles)

svd = TruncatedSVD(n_components=200, random_state=42)
X_all_svd = svd.fit_transform(X_all_text)
print(f"Explained Variance Ratio sum (200 SVD): {svd.explained_variance_ratio_.sum():.4f}")

X_train_svd = X_all_svd[:len(train_titles)]
X_test_svd = X_all_svd[len(train_titles):]

# 3. Engineer Price & Metadata Features (Normalized)
print("Engineering price, warranty, and listing type features...")
# Category median price dict
cat_medians = {}
df_tr = pd.DataFrame({"cat": train_cats, "price": train_prices})
cat_medians = df_tr.groupby("cat")["price"].median().to_dict()

def extract_meta(data, prices, cats):
    feats = []
    for d, p, cat in zip(data, prices, cats):
        log_p = np.log1p(max(0, p))
        med = cat_medians.get(cat, 300.0)
        rel_p = (p - med) / (med + 1.0)
        has_war = 1.0 if d.get("warranty") else 0.0
        free_ship = 1.0 if d.get("shipping", {}).get("free_shipping") else 0.0
        is_gold = 1.0 if "gold" in d.get("listing_type_id", "") else 0.0
        is_bronze = 1.0 if "bronze" in d.get("listing_type_id", "") else 0.0
        is_free = 1.0 if "free" in d.get("listing_type_id", "") else 0.0
        feats.append([log_p, rel_p, has_war, free_ship, is_gold, is_bronze, is_free])
    return np.array(feats, dtype=np.float32)

meta_tr = extract_meta(train_data, train_prices, train_cats)
meta_te = extract_meta(test_data, test_prices, test_cats)

scaler = StandardScaler()
meta_tr_sc = scaler.fit_transform(meta_tr)
meta_te_sc = scaler.transform(meta_te)

X_train_dense = np.hstack([X_train_svd, meta_tr_sc])
X_test_dense = np.hstack([X_test_svd, meta_te_sc])
print(f"Dense Feature Matrix: {X_train_dense.shape}")

# 4. 5-Fold Stratified Cross-Validation with MLPClassifier
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_mlp = np.zeros(len(train_titles), dtype=np.float32)
test_mlp = np.zeros(len(test_titles), dtype=np.float32)

print("\nTraining 5-Fold Deep Multi-Layer Perceptron (512, 256, 128)...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_dense, y_train)):
    f_start = time.time()
    X_tr, y_tr = X_train_dense[train_idx], y_train[train_idx]
    X_va, y_val = X_train_dense[val_idx], y_train[val_idx]
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        max_iter=150,
        random_state=42 + fold
    )
    mlp.fit(X_tr, y_tr)
    
    val_probs = mlp.predict_proba(X_va)[:, 1]
    oof_mlp[val_idx] = val_probs
    test_mlp += mlp.predict_proba(X_test_dense)[:, 1] / 5.0
    
    acc = accuracy_score(y_val, (val_probs >= 0.5).astype(int))
    auc = roc_auc_score(y_val, val_probs)
    print(f"Fold {fold+1}/5 ({time.time()-f_start:.1f}s) | MLP Acc: {acc:.5f} | AUC: {auc:.5f} | Iterations: {mlp.n_iter_}")

acc_tot = accuracy_score(y_train, (oof_mlp >= 0.5).astype(int))
auc_tot = roc_auc_score(y_train, oof_mlp)
print("\n=== MLP NEURAL SPECIALIST OVERALL RESULTS ===")
print(f"MLP OOF Accuracy: {acc_tot:.5f} ({int(acc_tot*70000)}/70000 correct) | AUC: {auc_tot:.5f}")

# Save predictions
os.makedirs("results/deep_parallel_20261109", exist_ok=True)
np.savez_compressed(
    "results/deep_parallel_20261109/mlp_predictions.npz",
    oof_mlp=oof_mlp,
    test_mlp=test_mlp
)
print(f"Saved artifacts to results/deep_parallel_20261109/mlp_predictions.npz ({time.time()-start_time:.1f}s)")
