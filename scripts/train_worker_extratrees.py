"""Worker: ExtraTrees Bagging Specialist.

Provides randomized bagging diversity completely decorrelated from gradient boosting:
- 120 trees with max_depth=22, min_samples_split=12.
- Fits strictly on outer-train under StratifiedGroupKFold(5).
- Generates OOF and test probabilities.
"""

import os, sys, json, time, gc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

out_dir = 'results/worker_extratrees_20261110'
os.makedirs(out_dir, exist_ok=True)

print("Loading train and test data for ExtraTrees...", flush=True)
with open('data/ap_2026_q_2/raw/train_data.jsonlines', encoding='utf-8') as f:
    train_rows = [json.loads(line) for line in f]

with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    test_rows = [json.loads(line) for line in f]

n_train = len(train_rows)
n_test = len(test_rows)

y_train = np.array([1 if r['condition'] == 'used' else 0 for r in train_rows], dtype=np.int32)
sellers_train = np.array([r.get('seller_id') or 'unknown' for r in train_rows])
train_titles = [str(r.get('title') or '') for r in train_rows]
test_titles = [str(r.get('title') or '') for r in test_rows]

train_prices = np.array([float(r.get('price') or 0.0) for r in train_rows], dtype=np.float32)
test_prices = np.array([float(r.get('price') or 0.0) for r in test_rows], dtype=np.float32)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

oof_p = np.zeros(n_train, dtype=np.float32)
test_p_folds = np.zeros((5, n_test), dtype=np.float32)

for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_titles, y_train, groups=sellers_train)):
    t0 = time.time()
    print(f"\n--- FOLD {fold + 1}/5 (Train: {len(train_idx)}, Val: {len(val_idx)}) ---", flush=True)
    
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=25000, sublinear_tf=True, min_df=2)
    X_tr_tfidf = vec.fit_transform([train_titles[i] for i in train_idx])
    X_val_tfidf = vec.transform([train_titles[i] for i in val_idx])
    X_te_tfidf = vec.transform(test_titles)
    
    svd = TruncatedSVD(n_components=60, random_state=42)
    X_tr_svd = svd.fit_transform(X_tr_tfidf)
    X_val_svd = svd.transform(X_val_tfidf)
    X_te_svd = svd.transform(X_te_tfidf)
    
    X_tr = np.column_stack([np.log1p(train_prices[train_idx]), X_tr_svd])
    X_val = np.column_stack([np.log1p(train_prices[val_idx]), X_val_svd])
    X_te = np.column_stack([np.log1p(test_prices), X_te_svd])
    
    et = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=15,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    )
    et.fit(X_tr, y_train[train_idx])
    
    p_val = et.predict_proba(X_val)[:, 1]
    oof_p[val_idx] = p_val
    test_p_folds[fold] = et.predict_proba(X_te)[:, 1]
    
    acc = accuracy_score(y_train[val_idx], (p_val >= 0.5).astype(int))
    auc = roc_auc_score(y_train[val_idx], p_val)
    print(f"Fold {fold + 1} finished in {time.time()-t0:.1f}s -> Acc: {acc:.5f}, AUC: {auc:.5f}")
    
    del X_tr_tfidf, X_val_tfidf, X_te_tfidf, X_tr, X_val, X_te, et
    gc.collect()

global_acc = accuracy_score(y_train, (oof_p >= 0.5).astype(int))
global_auc = roc_auc_score(y_train, oof_p)
n_corr = int(((oof_p >= 0.5).astype(int) == y_train).sum())
print("\n" + "=" * 60)
print(f"GLOBAL EXTRATREES OOF ACC: {global_acc:.5f} ({n_corr}/{n_train}) | AUC: {global_auc:.5f}")
print("=" * 60)

# Save
pd.DataFrame({
    'id': np.arange(1, n_train + 1),
    'p_used_extratrees': oof_p
}).to_csv(os.path.join(out_dir, 'oof_extratrees.csv'), index=False)

pd.DataFrame({
    'id': np.arange(1, n_test + 1),
    'p_used_extratrees': test_p_folds.mean(axis=0)
}).to_csv(os.path.join(out_dir, 'test_probabilities_extratrees.csv'), index=False)

print(f"Saved ExtraTrees artifacts to {out_dir}")
