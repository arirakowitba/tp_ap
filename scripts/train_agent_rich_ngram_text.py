"""Agent Worker: Rich N-gram Text Specialist (Words 1-3 + Chars 2-6).

Trains ultra-rich text representations under outer StratifiedGroupKFold(5):
- 60,000 word n-grams (1-3) + 80,000 character n-grams (2-6).
- Sublinear TF-IDF scaling + L2 regularized Logistic Regression.
- 100% item-intrinsic features (ZERO seller leakage).
- Fits strictly inside outer-train.
- Generates OOF predictions and test probabilities for all 30,000 items.
"""

import os, sys, json, time, gc
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

out_dir = 'results/agent_rich_ngram_text_20261110'
os.makedirs(out_dir, exist_ok=True)

print("Loading train and test data for Rich N-gram Text...", flush=True)
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

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

oof_p = np.zeros(n_train, dtype=np.float32)
test_p_folds = np.zeros((5, n_test), dtype=np.float32)
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_titles, y_train, groups=sellers_train)):
    t0 = time.time()
    print(f"\n--- FOLD {fold + 1}/5 (Train: {len(train_idx)}, Val: {len(val_idx)}) ---", flush=True)
    
    # 1. Word n-grams (1-3)
    vec_word = TfidfVectorizer(ngram_range=(1, 3), max_features=60000, sublinear_tf=True, min_df=2)
    X_tr_w = vec_word.fit_transform([train_titles[i] for i in train_idx])
    X_val_w = vec_word.transform([train_titles[i] for i in val_idx])
    X_te_w = vec_word.transform(test_titles)
    
    # 2. Char n-grams (2-6)
    vec_char = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 6), max_features=80000, sublinear_tf=True, min_df=2)
    X_tr_c = vec_char.fit_transform([train_titles[i] for i in train_idx])
    X_val_c = vec_char.transform([train_titles[i] for i in val_idx])
    X_te_c = vec_char.transform(test_titles)
    
    # Stack sparse
    X_tr = sparse.hstack([X_tr_w, X_tr_c], format='csr')
    X_val = sparse.hstack([X_val_w, X_val_c], format='csr')
    X_te = sparse.hstack([X_te_w, X_te_c], format='csr')
    
    clf = LogisticRegression(C=2.0, solver='saga', max_iter=45, random_state=42)
    clf.fit(X_tr, y_train[train_idx])
    
    p_val = clf.predict_proba(X_val)[:, 1]
    oof_p[val_idx] = p_val
    test_p_folds[fold] = clf.predict_proba(X_te)[:, 1]
    
    acc = accuracy_score(y_train[val_idx], (p_val >= 0.5).astype(int))
    auc = roc_auc_score(y_train[val_idx], p_val)
    ll = log_loss(y_train[val_idx], p_val)
    fold_metrics.append({'fold': fold + 1, 'accuracy': acc, 'roc_auc': auc, 'logloss': ll})
    print(f"Fold {fold + 1} finished in {time.time()-t0:.1f}s -> Acc: {acc:.5f}, AUC: {auc:.5f}, LL: {ll:.5f}")
    
    del vec_word, vec_char, X_tr_w, X_val_w, X_te_w, X_tr_c, X_val_c, X_te_c, X_tr, X_val, X_te, clf
    gc.collect()

global_acc = accuracy_score(y_train, (oof_p >= 0.5).astype(int))
global_auc = roc_auc_score(y_train, oof_p)
global_ll = log_loss(y_train, oof_p)
n_corr = int(((oof_p >= 0.5).astype(int) == y_train).sum())

print("\n" + "=" * 60)
print(f"GLOBAL RICH N-GRAM TEXT OOF ACC: {global_acc:.5f} ({n_corr}/{n_train})")
print(f"GLOBAL ROC-AUC: {global_auc:.5f} | LOGLOSS: {global_ll:.5f}")
print("=" * 60)

pd.DataFrame({
    'id': np.arange(1, n_train + 1),
    'seller_id': sellers_train,
    'target_condition_used': y_train,
    'p_used_rich_ngram': oof_p
}).to_csv(os.path.join(out_dir, 'oof_rich_ngram.csv'), index=False)

pd.DataFrame({
    'id': np.arange(1, n_test + 1),
    'p_used_rich_ngram': test_p_folds.mean(axis=0)
}).to_csv(os.path.join(out_dir, 'test_probabilities_rich_ngram.csv'), index=False)

summary = {
    'model_name': 'Rich N-gram Text Specialist (Words 1-3 [60k] + Chars 2-6 [80k])',
    'n_outer_folds': 5,
    'global_accuracy': float(global_acc),
    'global_roc_auc': float(global_auc),
    'global_logloss': float(global_ll),
    'fold_metrics': fold_metrics
}
with open(os.path.join(out_dir, 'summary_report.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Saved artifacts to {out_dir}")
