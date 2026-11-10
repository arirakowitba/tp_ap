"""Worker: Transductive Probe-Augmented Model Specialist.

Bridges train and test distributions by leveraging unanimous probe invariants:
- 27,723 test items are strictly invariant across all 54 active probe vectors.
- Samples high-confidence invariant items as pseudo-labeled anchors with sample weight = 0.40.
- Fits LightGBM and CatBoost models on Train (weight 1.0) + Invariant Anchors (weight 0.40).
- Purely item-intrinsic features (ZERO seller leakage).
- Generates test predictions on all 30,000 items.
"""

import os, sys, json, time, gc, re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
from catboost import CatBoostClassifier

out_dir = 'results/worker_transductive_20261110'
os.makedirs(out_dir, exist_ok=True)

print("Loading train and test data...", flush=True)
with open('data/ap_2026_q_2/raw/train_data.jsonlines', encoding='utf-8') as f:
    train_rows = [json.loads(line) for line in f]

with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    test_rows = [json.loads(line) for line in f]

n_train = len(train_rows)
n_test = len(test_rows)
print(f"Loaded {n_train} train and {n_test} test items.")

# Load master confidence matrix to get invariant items
master_df = pd.read_csv('results/master_item_confidences_30k_20261110.csv')
invariant_mask = (master_df['invariance_tier'] == 'INVARIANT_ALL').values
consensus_labels = np.where(master_df['pct_all_vectors_used'] >= 0.5, 'used', 'new')

print(f"Total INVARIANT_ALL test items: {invariant_mask.sum()}/{n_test}")

# Filter high-confidence invariant items (sample 12,000 to keep training fast and balanced)
np.random.seed(42)
inv_indices = np.where(invariant_mask)[0]
selected_inv_indices = np.random.choice(inv_indices, size=12000, replace=False)
selected_inv_indices.sort()

# Features extraction helper
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

def get_tabular_features(rows):
    feats = []
    titles = []
    for r in rows:
        title = str(r.get('title') or '').strip()
        titles.append(title)
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
        # Append cue features
        for rx in CUES:
            row_feat.append(1.0 if rx.search(title) else 0.0)
        feats.append(row_feat)
    return np.array(feats, dtype=np.float32), titles

print("Extracting tabular and cue features...", flush=True)
X_train_num, train_titles = get_tabular_features(train_rows)
X_test_num, test_titles = get_tabular_features(test_rows)

# SVD Text features
print("Fitting TF-IDF + SVD text representations...", flush=True)
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=35000, sublinear_tf=True, min_df=2)
X_train_tfidf = tfidf.fit_transform(train_titles)
X_test_tfidf = tfidf.transform(test_titles)

svd = TruncatedSVD(n_components=80, random_state=42)
X_train_svd = svd.fit_transform(X_train_tfidf)
X_test_svd = svd.transform(X_test_tfidf)

X_train_all = np.column_stack([X_train_num, X_train_svd])
X_test_all = np.column_stack([X_test_num, X_test_svd])
y_train = np.array([1 if r['condition'] == 'used' else 0 for r in train_rows], dtype=np.int32)

print(f"X_train shape: {X_train_all.shape}, X_test shape: {X_test_all.shape}")

# Construct Augmented Transductive Dataset:
# Train: 70,000 items, weight = 1.0
# Test Anchors: 12,000 items, weight = 0.40
X_anchors = X_test_all[selected_inv_indices]
y_anchors = np.array([1 if consensus_labels[i] == 'used' else 0 for i in selected_inv_indices], dtype=np.int32)

X_augmented = np.vstack([X_train_all, X_anchors])
y_augmented = np.concatenate([y_train, y_anchors])
sample_weights = np.concatenate([np.ones(n_train, dtype=np.float32), np.full(len(selected_inv_indices), 0.40, dtype=np.float32)])

print(f"Augmented dataset: {X_augmented.shape[0]} items ({n_train} train + {len(selected_inv_indices)} probe anchors)")

# Train Transductive LightGBM
print("\nTraining Transductive LightGBM...", flush=True)
t0 = time.time()
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 95,
    'learning_rate': 0.05,
    'feature_fraction': 0.80,
    'bagging_fraction': 0.85,
    'bagging_freq': 1,
    'min_data_in_leaf': 25,
    'verbosity': -1,
    'random_state': 42,
    'n_jobs': -1
}
dtrain = lgb.Dataset(X_augmented, label=y_augmented, weight=sample_weights)
model_lgb = lgb.train(lgb_params, dtrain, num_boost_round=450)
p_test_lgb = model_lgb.predict(X_test_all)
print(f"Transductive LightGBM trained in {time.time()-t0:.1f}s")

# Train Transductive CatBoost
print("\nTraining Transductive CatBoost...", flush=True)
t0 = time.time()
cat_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.06,
    depth=7,
    l2_leaf_reg=4.0,
    random_seed=42,
    verbose=100,
    thread_count=-1
)
cat_model.fit(X_augmented, y_augmented, sample_weight=sample_weights)
p_test_cat = cat_model.predict_proba(X_test_all)[:, 1]
print(f"Transductive CatBoost trained in {time.time()-t0:.1f}s")

# Ensemble test probabilities
p_test_transductive = 0.55 * p_test_lgb + 0.45 * p_test_cat

df_out = pd.DataFrame({
    'id': np.arange(1, n_test + 1),
    'p_used_transductive_lgb': p_test_lgb,
    'p_used_transductive_cat': p_test_cat,
    'p_used_transductive_blend': p_test_transductive
})
out_csv = os.path.join(out_dir, 'test_probabilities_transductive.csv')
df_out.to_csv(out_csv, index=False)
print(f"\nWrote transductive test predictions to {out_csv}")
