"""Agent Worker: Deep LightGBM Focal Specialist (High Capacity Intrinsic Trees).

Trains high-capacity LightGBM trees under strict outer StratifiedGroupKFold(5):
- 150 leaves, max_depth=12, feature_fraction=0.70, learning_rate=0.03, min_data_in_leaf=15.
- 100% item-intrinsic features (ZERO seller leakage).
- Fits strictly inside outer-train.
- Generates OOF predictions and test probabilities for all 30,000 items.
"""

import os, sys, json, time, gc, re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import lightgbm as lgb

out_dir = 'results/agent_deep_lgbm_focal_20261110'
os.makedirs(out_dir, exist_ok=True)

print("Loading train and test data for Deep LightGBM...", flush=True)
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

# Feature extraction helper
CUES = [
    re.compile(r"\b0\s*km\b|\bcero\s*km\b", re.I),
    re.compile(r"\bcaja\s+sellada\b|\bcerrad[oa]\s+de\s+f[aá]brica\b|\bprecintad[oa]\b|\ben\s+blister\b", re.I),
    re.compile(r"\bnuev[oa]s?\b|\ba\s+estrenar\b", re.I),
    re.compile(r"\b(usad[oa]s?|segunda\s+mano)\b", re.I),
    re.compile(r"\bpara\s+repuestos?\b|\bpara\s+desarme\b|\bno\s+funciona\b|\brot[oa]s?\b", re.I),
    re.compile(r"\b(un|1)\s+solo\s+uso\b|\bpoco\s+uso\b|\bpoqu[ií]simo\s+uso\b", re.I),
    re.compile(r"\bcomo\s+nuev[oa]\b|\bigual\s+a\s+nuev[oa]\b", re.I),
    re.compile(r"\bimpecable\b|\bde\s+vitrina\b", re.I),
    re.compile(r"\bantigu[oa]s?\b|\bde\s+epoca\b|\bvintage\b", re.I),
    re.compile(r"\b(libro\s+digital|ebook|pdf|descarga\s+digital)\b", re.I),
]

def get_tabular(rows):
    feats = []
    for r in rows:
        title = str(r.get('title') or '').strip()
        price = float(r.get('price') or 0.0)
        orig_price = float(r.get('original_price') or 0.0)
        has_orig = 1.0 if orig_price > 0 else 0.0
        ratio = (orig_price / (price + 1e-5)) if (has_orig and price > 0) else 1.0
        row_feat = [
            np.log1p(max(0.0, price)),
            np.log1p(max(0.0, orig_price)),
            has_orig,
            min(ratio, 10.0),
            len(title),
            len(title.split()),
            float(r.get('initial_quantity') or 0.0),
            float(r.get('available_quantity') or 0.0),
            float(r.get('sold_quantity') or 0.0),
            1.0 if r.get('accepts_mercadopago') else 0.0,
            1.0 if r.get('shipping', {}).get('free_shipping') else 0.0,
            len(r.get('attributes') or []),
            len(r.get('pictures') or []),
            len(r.get('tags') or []),
        ]
        for rx in CUES:
            row_feat.append(1.0 if rx.search(title) else 0.0)
        feats.append(row_feat)
    return np.array(feats, dtype=np.float32)

print("Extracting tabular features...", flush=True)
X_tr_tab = get_tabular(train_rows)
X_te_tab = get_tabular(test_rows)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

oof_p = np.zeros(n_train, dtype=np.float32)
test_p_folds = np.zeros((5, n_test), dtype=np.float32)
fold_metrics = []

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 150,
    'max_depth': 12,
    'learning_rate': 0.035,
    'feature_fraction': 0.70,
    'bagging_fraction': 0.80,
    'bagging_freq': 1,
    'min_data_in_leaf': 18,
    'verbosity': -1,
    'random_state': 42,
    'n_jobs': -1
}

for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_titles, y_train, groups=sellers_train)):
    t0 = time.time()
    print(f"\n--- FOLD {fold + 1}/5 (Train: {len(train_idx)}, Val: {len(val_idx)}) ---", flush=True)
    
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, sublinear_tf=True, min_df=2)
    X_tr_tfidf = vec.fit_transform([train_titles[i] for i in train_idx])
    X_val_tfidf = vec.transform([train_titles[i] for i in val_idx])
    X_te_tfidf = vec.transform(test_titles)
    
    svd = TruncatedSVD(n_components=80, random_state=42)
    X_tr_svd = svd.fit_transform(X_tr_tfidf)
    X_val_svd = svd.transform(X_val_tfidf)
    X_te_svd = svd.transform(X_te_tfidf)
    
    X_tr = np.column_stack([X_tr_tab[train_idx], X_tr_svd])
    X_val = np.column_stack([X_tr_tab[val_idx], X_val_svd])
    X_te = np.column_stack([X_te_tab, X_te_svd])
    
    dtrain = lgb.Dataset(X_tr, label=y_train[train_idx])
    dval = lgb.Dataset(X_val, label=y_train[val_idx], reference=dtrain)
    
    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=600,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
    )
    
    p_val = model.predict(X_val)
    oof_p[val_idx] = p_val
    test_p_folds[fold] = model.predict(X_te)
    
    acc = accuracy_score(y_train[val_idx], (p_val >= 0.5).astype(int))
    auc = roc_auc_score(y_train[val_idx], p_val)
    ll = log_loss(y_train[val_idx], p_val)
    fold_metrics.append({'fold': fold + 1, 'accuracy': acc, 'roc_auc': auc, 'logloss': ll})
    print(f"Fold {fold + 1} finished in {time.time()-t0:.1f}s -> Acc: {acc:.5f}, AUC: {auc:.5f}, LL: {ll:.5f}")
    
    del vec, svd, X_tr_tfidf, X_val_tfidf, X_te_tfidf, X_tr, X_val, X_te, dtrain, dval, model
    gc.collect()

global_acc = accuracy_score(y_train, (oof_p >= 0.5).astype(int))
global_auc = roc_auc_score(y_train, oof_p)
global_ll = log_loss(y_train, oof_p)
n_corr = int(((oof_p >= 0.5).astype(int) == y_train).sum())

print("\n" + "=" * 60)
print(f"GLOBAL DEEP LIGHTGBM OOF ACC: {global_acc:.5f} ({n_corr}/{n_train})")
print(f"GLOBAL ROC-AUC: {global_auc:.5f} | LOGLOSS: {global_ll:.5f}")
print("=" * 60)

pd.DataFrame({
    'id': np.arange(1, n_train + 1),
    'seller_id': sellers_train,
    'target_condition_used': y_train,
    'p_used_deep_lgbm': oof_p
}).to_csv(os.path.join(out_dir, 'oof_deep_lgbm.csv'), index=False)

pd.DataFrame({
    'id': np.arange(1, n_test + 1),
    'p_used_deep_lgbm': test_p_folds.mean(axis=0)
}).to_csv(os.path.join(out_dir, 'test_probabilities_deep_lgbm.csv'), index=False)

summary = {
    'model_name': 'Deep LightGBM Focal Specialist (num_leaves=150, max_depth=12)',
    'n_outer_folds': 5,
    'global_accuracy': float(global_acc),
    'global_roc_auc': float(global_auc),
    'global_logloss': float(global_ll),
    'fold_metrics': fold_metrics
}
with open(os.path.join(out_dir, 'summary_report.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Saved artifacts to {out_dir}")
