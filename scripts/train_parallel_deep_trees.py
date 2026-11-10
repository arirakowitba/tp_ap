"""High-Intensity Parallel Gradient Boosted Trees Specialist (LightGBM, CatBoost, XGBoost).
Utilizes all 16 cores with OpenMP/native multithreading.
Evaluates 5-fold Stratified K-Fold across 70,000 items.
Features: 150 TruncatedSVD components + 35 Tabular/Lexical Features (Strictly Zero-Seller-Leak).
"""
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import os

start_time = time.time()
print("=== PARALLEL STREAM 3: HIGH-CAPACITY GRADIENT BOOSTED TREES (LGBM, CATBOOST, XGB) ===")

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

# 2. Text Features (150 TruncatedSVD components)
print("Extracting 150 TruncatedSVD components from 120,000 TF-IDF features...")
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=120000, sublinear_tf=True)
X_all_text = tfidf.fit_transform(train_titles + test_titles)

svd = TruncatedSVD(n_components=150, random_state=42)
X_all_svd = svd.fit_transform(X_all_text)
print(f"Explained Variance Ratio sum (150 SVD): {svd.explained_variance_ratio_.sum():.4f}")

X_tr_svd = X_all_svd[:len(train_titles)]
X_te_svd = X_all_svd[len(train_titles):]

# 3. Tabular & Lexical Features
print("Engineering 35 leak-free tabular & lexical features...")
# Category price aggregations
df_cats = pd.DataFrame({"cat": train_cats, "price": train_prices})
cat_stats = df_cats.groupby("cat")["price"].agg(["median", "mean", "std"]).to_dict("index")

# Lexical indicators
used_cues = ["usad", "reacond", "buen estado", "impecable estado", "repuesto", "despiece", "partes", "detalles", "falla", "antigu", "retro", "vintage"]
new_cues = ["nuevo", "nueva", "nuevas", "nuevos", "blister", "sellad", "cerrad", "estrenar", "0km", "0 km", "original", "garantia", "caja sellada"]

def extract_features(data, titles, prices, cats):
    feats = []
    for d, t, p, cat in zip(data, titles, prices, cats):
        t_lower = t.lower()
        log_p = np.log1p(max(0, p))
        st = cat_stats.get(cat, {"median": 300.0, "mean": 600.0, "std": 800.0})
        med = st["median"]
        mean_p = st["mean"]
        std_p = st["std"] if not np.isnan(st["std"]) and st["std"] > 0 else 1.0
        
        diff_med = (p - med) / (med + 1.0)
        z_score = (p - mean_p) / std_p
        
        has_war = 1.0 if d.get("warranty") else 0.0
        free_ship = 1.0 if d.get("shipping", {}).get("free_shipping") else 0.0
        is_gold = 1.0 if "gold" in d.get("listing_type_id", "") else 0.0
        is_bronze = 1.0 if "bronze" in d.get("listing_type_id", "") else 0.0
        is_free = 1.0 if "free" in d.get("listing_type_id", "") else 0.0
        
        title_len = len(t)
        word_count = len(t.split())
        num_count = sum(c.isdigit() for c in t)
        upper_count = sum(c.isupper() for c in t)
        
        # Lexical flags
        u_flags = [1.0 if cue in t_lower else 0.0 for cue in used_cues]
        n_flags = [1.0 if cue in t_lower else 0.0 for cue in new_cues]
        
        row = [log_p, diff_med, z_score, has_war, free_ship, is_gold, is_bronze, is_free,
               title_len, word_count, num_count, upper_count] + u_flags + n_flags
        feats.append(row)
    return np.array(feats, dtype=np.float32)

tab_tr = extract_features(train_data, train_titles, train_prices, train_cats)
tab_te = extract_features(test_data, test_titles, test_prices, test_cats)

X_train_full = np.hstack([X_tr_svd, tab_tr])
X_test_full = np.hstack([X_te_svd, tab_te])
print(f"Total Combined Feature Matrix: {X_train_full.shape}")

# 4. 5-Fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(train_titles), dtype=np.float32)
test_lgb = np.zeros(len(test_titles), dtype=np.float32)

oof_xgb = np.zeros(len(train_titles), dtype=np.float32)
test_xgb = np.zeros(len(test_titles), dtype=np.float32)

oof_cat = np.zeros(len(train_titles), dtype=np.float32)
test_cat = np.zeros(len(test_titles), dtype=np.float32)

print("\nTraining 5-Fold LightGBM (1500 trees, 63 leaves, 16 threads)...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train)):
    f_start = time.time()
    X_tr, y_tr = X_train_full[train_idx], y_train[train_idx]
    X_va, y_val = X_train_full[val_idx], y_train[val_idx]
    
    # LightGBM
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42 + fold,
        n_jobs=16,
        verbose=-1
    )
    clf_lgb.fit(X_tr, y_tr)
    v_lgb = clf_lgb.predict_proba(X_va)[:, 1]
    oof_lgb[val_idx] = v_lgb
    test_lgb += clf_lgb.predict_proba(X_test_full)[:, 1] / 5.0
    
    # XGBoost (Hist)
    clf_xgb = xgb.XGBClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.7,
        tree_method="hist",
        n_jobs=16,
        random_state=42 + fold,
        eval_metric="logloss"
    )
    clf_xgb.fit(X_tr, y_tr)
    v_xgb = clf_xgb.predict_proba(X_va)[:, 1]
    oof_xgb[val_idx] = v_xgb
    test_xgb += clf_xgb.predict_proba(X_test_full)[:, 1] / 5.0
    
    # CatBoost
    clf_cat = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.04,
        depth=7,
        thread_count=16,
        random_seed=42 + fold,
        verbose=0
    )
    clf_cat.fit(X_tr, y_tr)
    v_cat = clf_cat.predict_proba(X_va)[:, 1]
    oof_cat[val_idx] = v_cat
    test_cat += clf_cat.predict_proba(X_test_full)[:, 1] / 5.0
    
    acc_lgb = accuracy_score(y_val, (v_lgb >= 0.5).astype(int))
    acc_xgb = accuracy_score(y_val, (v_xgb >= 0.5).astype(int))
    acc_cat = accuracy_score(y_val, (v_cat >= 0.5).astype(int))
    print(f"Fold {fold+1}/5 ({time.time()-f_start:.1f}s) | LGB: {acc_lgb:.5f} | XGB: {acc_xgb:.5f} | Cat: {acc_cat:.5f}")

acc_lgb_tot = accuracy_score(y_train, (oof_lgb >= 0.5).astype(int))
acc_xgb_tot = accuracy_score(y_train, (oof_xgb >= 0.5).astype(int))
acc_cat_tot = accuracy_score(y_train, (oof_cat >= 0.5).astype(int))

tree_blend_oof = (oof_lgb + oof_xgb + oof_cat) / 3.0
tree_blend_test = (test_lgb + test_xgb + test_cat) / 3.0
acc_tree_blend = accuracy_score(y_train, (tree_blend_oof >= 0.5).astype(int))

print("\n=== TREE SPECIALISTS OVERALL RESULTS ===")
print(f"LightGBM OOF Acc: {acc_lgb_tot:.5f} (AUC: {roc_auc_score(y_train, oof_lgb):.5f})")
print(f"XGBoost  OOF Acc: {acc_xgb_tot:.5f} (AUC: {roc_auc_score(y_train, oof_xgb):.5f})")
print(f"CatBoost OOF Acc: {acc_cat_tot:.5f} (AUC: {roc_auc_score(y_train, oof_cat):.5f})")
print(f"Tree Blend OOF Acc: {acc_tree_blend:.5f} ({int(acc_tree_blend*70000)}/70000 correct)")

# Save predictions
os.makedirs("results/deep_parallel_20261109", exist_ok=True)
np.savez_compressed(
    "results/deep_parallel_20261109/tree_predictions.npz",
    oof_lgb=oof_lgb, test_lgb=test_lgb,
    oof_xgb=oof_xgb, test_xgb=test_xgb,
    oof_cat=oof_cat, test_cat=test_cat,
    tree_blend_oof=tree_blend_oof,
    tree_blend_test=tree_blend_test
)
print(f"Saved artifacts to results/deep_parallel_20261109/tree_predictions.npz ({time.time()-start_time:.1f}s)")
