"""Worker: Subword Character N-Gram Specialist (Morphological Text Model).

Captures subword morphological cues (3-6 char n-grams) across all 70,000 items:
- Tolerates misspellings, plurals, compound words, technical codes.
- Outer StratifiedGroupKFold(n_splits=5) grouped by seller_id (zero seller leakage).
- Fits strictly inside outer-train.
- Generates OOF predictions and test probabilities for all 30,000 items.
"""

import os, sys, json, time, gc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

out_dir = 'results/worker_char_morphology_20261110'
os.makedirs(out_dir, exist_ok=True)

print("Loading train data...", flush=True)
with open('data/ap_2026_q_2/raw/train_data.jsonlines', encoding='utf-8') as f:
    train_rows = [json.loads(line) for line in f]

with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    test_rows = [json.loads(line) for line in f]

n_train = len(train_rows)
n_test = len(test_rows)
print(f"Loaded {n_train} train and {n_test} test items.")

y_train = np.array([1 if r['condition'] == 'used' else 0 for r in train_rows], dtype=np.int32)
sellers_train = np.array([r.get('seller_id') or 'unknown' for r in train_rows])
titles_train = [str(r.get('title') or '') for r in train_rows]
titles_test = [str(r.get('title') or '') for r in test_rows]

# StratifiedGroupKFold outer validation
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

oof_p = np.zeros(n_train, dtype=np.float32)
test_p_folds = np.zeros((5, n_test), dtype=np.float32)

fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(sgkf.split(titles_train, y_train, groups=sellers_train)):
    t0 = time.time()
    print(f"\n--- FOLD {fold + 1}/5 (Train: {len(train_idx)}, Val: {len(val_idx)}) ---", flush=True)
    
    # 1. Morphological subword char n-grams (3-5 chars)
    vec = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        max_features=80000,
        sublinear_tf=True,
        min_df=2
    )
    X_tr = vec.fit_transform([titles_train[i] for i in train_idx])
    X_val = vec.transform([titles_train[i] for i in val_idx])
    X_te = vec.transform(titles_test)
    
    # 2. Logistic Regression with saga / lbfgs
    clf = LogisticRegression(C=2.0, solver='saga', max_iter=40, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_train[train_idx])
    
    p_val = clf.predict_proba(X_val)[:, 1]
    oof_p[val_idx] = p_val
    test_p_folds[fold] = clf.predict_proba(X_te)[:, 1]
    
    acc = accuracy_score(y_train[val_idx], (p_val >= 0.5).astype(int))
    auc = roc_auc_score(y_train[val_idx], p_val)
    ll = log_loss(y_train[val_idx], p_val)
    fold_metrics.append({'fold': fold + 1, 'accuracy': acc, 'roc_auc': auc, 'logloss': ll})
    print(f"Fold {fold + 1} finished in {time.time()-t0:.1f}s -> Acc: {acc:.5f}, AUC: {auc:.5f}, LL: {ll:.5f}")
    
    del X_tr, X_val, X_te, clf, vec
    gc.collect()

global_acc = accuracy_score(y_train, (oof_p >= 0.5).astype(int))
global_auc = roc_auc_score(y_train, oof_p)
global_ll = log_loss(y_train, oof_p)

print("\n" + "=" * 60)
n_corr = int(((oof_p >= 0.5).astype(int) == y_train).sum())
print(f"GLOBAL CHAR MORPHOLOGY OOF ACC: {global_acc:.5f} ({n_corr}/{n_train})")
print(f"GLOBAL ROC-AUC: {global_auc:.5f} | LOGLOSS: {global_ll:.5f}")
print("=" * 60)

# Save OOF predictions
df_oof = pd.DataFrame({
    'id': np.arange(1, n_train + 1),
    'seller_id': sellers_train,
    'target_condition_used': y_train,
    'p_used_char_morphology': oof_p
})
df_oof.to_csv(os.path.join(out_dir, 'oof_char_morphology.csv'), index=False)

# Save Test predictions
test_p_mean = test_p_folds.mean(axis=0)
df_test = pd.DataFrame({
    'id': np.arange(1, n_test + 1),
    'p_used_char_morphology': test_p_mean
})
df_test.to_csv(os.path.join(out_dir, 'test_probabilities_char_morphology.csv'), index=False)

summary = {
    'model_name': 'Subword Character N-Gram Specialist (char_wb 3-5)',
    'n_outer_folds': 5,
    'global_accuracy': float(global_acc),
    'global_roc_auc': float(global_auc),
    'global_logloss': float(global_ll),
    'fold_metrics': fold_metrics
}
with open(os.path.join(out_dir, 'summary_report.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Saved artifacts to {out_dir}")
