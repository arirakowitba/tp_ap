"""Prototype for Deep MLP and ExtraTrees on outer seller-group split."""

import json, time, gc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

print("Loading train data...", flush=True)
with open('data/ap_2026_q_2/raw/train_data.jsonlines', encoding='utf-8') as f:
    train_rows = [json.loads(line) for line in f]

y = np.array([1 if r['condition'] == 'used' else 0 for r in train_rows], dtype=np.int32)
sellers = np.array([r.get('seller_id') or 'unknown' for r in train_rows])
titles = [str(r.get('title') or '') for r in train_rows]
prices = np.array([float(r.get('price') or 0.0) for r in train_rows], dtype=np.float32)

print(f"Loaded {len(train_rows)} items, target mean: {y.mean():.4f}")

# Test on 1 fold first
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(sgkf.split(titles, y, groups=sellers))

print(f"Fold 1 train: {len(train_idx)}, val: {len(val_idx)}")

# Extract text features strictly in outer-train
t0 = time.time()
vec = TfidfVectorizer(ngram_range=(1, 2), max_features=25000, sublinear_tf=True)
X_tr_tfidf = vec.fit_transform([titles[i] for i in train_idx])
X_val_tfidf = vec.transform([titles[i] for i in val_idx])

svd = TruncatedSVD(n_components=80, random_state=42)
X_tr_svd = svd.fit_transform(X_tr_tfidf)
X_val_svd = svd.transform(X_val_tfidf)

# Scale
scaler = StandardScaler()
X_tr_dense = scaler.fit_transform(np.column_stack([np.log1p(prices[train_idx]), X_tr_svd]))
X_val_dense = scaler.transform(np.column_stack([np.log1p(prices[val_idx]), X_val_svd]))
print(f"Feature prep done in {time.time()-t0:.1f}s. Dense shape: {X_tr_dense.shape}")

# Test MLP
t0 = time.time()
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=25, early_stopping=True, n_iter_no_change=3, random_state=42)
mlp.fit(X_tr_dense, y[train_idx])
p_mlp_val = mlp.predict_proba(X_val_dense)[:, 1]
acc_mlp = accuracy_score(y[val_idx], (p_mlp_val >= 0.5).astype(int))
auc_mlp = roc_auc_score(y[val_idx], p_mlp_val)
print(f"MLP (128, 64) in {time.time()-t0:.1f}s -> Val Acc: {acc_mlp:.5f}, AUC: {auc_mlp:.5f}")

# Test HistGB
t0 = time.time()
hgb = HistGradientBoostingClassifier(max_iter=150, min_samples_leaf=20, random_state=42)
hgb.fit(X_tr_dense, y[train_idx])
p_hgb_val = hgb.predict_proba(X_val_dense)[:, 1]
acc_hgb = accuracy_score(y[val_idx], (p_hgb_val >= 0.5).astype(int))
auc_hgb = roc_auc_score(y[val_idx], p_hgb_val)
print(f"HistGB in {time.time()-t0:.1f}s -> Val Acc: {acc_hgb:.5f}, AUC: {auc_hgb:.5f}")

# Test blend
p_blend = 0.5 * p_mlp_val + 0.5 * p_hgb_val
acc_blend = accuracy_score(y[val_idx], (p_blend >= 0.5).astype(int))
auc_blend = roc_auc_score(y[val_idx], p_blend)
print(f"Blend (MLP + HistGB) -> Val Acc: {acc_blend:.5f}, AUC: {auc_blend:.5f}")
