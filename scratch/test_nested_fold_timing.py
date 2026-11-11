import sys, os, time, json, gc
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, log_loss
import lightgbm as lgb
from catboost import CatBoostClassifier

from scripts.train_strictly_nested_dual_regime_cv import extract_raw_fields

t0 = time.time()
print("Loading data...")
with open("data/ap_2026_q_2/raw/train_data.jsonlines", "r", encoding="utf-8") as f:
    train_rows = [json.loads(line) for line in f]
df_splits = pd.read_csv("data/splits/canonical_nested_dual_regime_splits.csv")
y_train = np.array([1 if r.get("condition") == "used" else 0 for r in train_rows], dtype=int)

train_ids, train_corpus, df_train_base, train_cats = extract_raw_fields(train_rows)
print(f"Data prepared in {time.time() - t0:.2f}s")

# Test fold 0 outer
val_idx = np.where(df_splits["outer_unseen_fold"] == 0)[0]
tr_idx = np.where(df_splits["outer_unseen_fold"] != 0)[0]
print(f"Outer fold 0: Train {len(tr_idx)}, Val {len(val_idx)}")

# Inner fold 1
in_val_idx = np.where((df_splits["outer_unseen_fold"] == 1))[0]
in_tr_idx = np.where((df_splits["outer_unseen_fold"] != 0) & (df_splits["outer_unseen_fold"] != 1))[0]
print(f"Inner fold 1: Train {len(in_tr_idx)}, Val {len(in_val_idx)}")

t_fit = time.time()
fit_corpus = [train_corpus[i] for i in in_tr_idx]
val_corpus = [train_corpus[i] for i in in_val_idx]

vec_word = TfidfVectorizer(ngram_range=(1, 2), max_features=25000, sublinear_tf=True, min_df=3)
vec_char = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=20000, sublinear_tf=True, min_df=5)

X_w_tr = vec_word.fit_transform(fit_corpus)
X_c_tr = vec_char.fit_transform(fit_corpus)
X_tfidf_tr = sparse.hstack([X_w_tr, X_c_tr]).tocsr()

X_w_va = vec_word.transform(val_corpus)
X_c_va = vec_char.transform(val_corpus)
X_tfidf_va = sparse.hstack([X_w_va, X_c_va]).tocsr()

print(f"TFIDF transform in {time.time() - t_fit:.2f}s, shape: {X_tfidf_tr.shape}")

t_svd = time.time()
svd = TruncatedSVD(n_components=50, random_state=42)
X_svd_tr = svd.fit_transform(X_tfidf_tr).astype(np.float32)
X_svd_va = svd.transform(X_tfidf_va).astype(np.float32)
print(f"SVD in {time.time() - t_svd:.2f}s")

# Model A1: LogisticRegression
t_m = time.time()
lr = LogisticRegression(C=1.5, solver="lbfgs", max_iter=150, random_state=42)
lr.fit(X_tfidf_tr, y_train[in_tr_idx])
p_lr = lr.predict_proba(X_tfidf_va)[:, 1]
print(f"LR in {time.time() - t_m:.2f}s, Acc: {accuracy_score(y_train[in_val_idx], (p_lr>=0.5).astype(int)):.4f}, LL: {log_loss(y_train[in_val_idx], p_lr):.4f}")

# Model A2: Ridge
t_m = time.time()
ridge = RidgeClassifier(alpha=2.0, random_state=42)
ridge.fit(X_tfidf_tr, y_train[in_tr_idx])
dfunc_tr = ridge.decision_function(X_tfidf_tr)
dfunc_va = ridge.decision_function(X_tfidf_va)
# Calibrate Platt
scale = float(np.std(dfunc_tr))
p_ridge = expit(dfunc_va / (scale + 1e-6))
print(f"Ridge in {time.time() - t_m:.2f}s, Acc: {accuracy_score(y_train[in_val_idx], (p_ridge>=0.5).astype(int)):.4f}")

# Tabular features
df_tr_b = df_train_base.iloc[in_tr_idx].copy().reset_index(drop=True)
df_va_b = df_train_base.iloc[in_val_idx].copy().reset_index(drop=True)

# SVD dataframe
for i in range(50):
    df_tr_b[f"svd_{i}"] = X_svd_tr[:, i]
    df_va_b[f"svd_{i}"] = X_svd_va[:, i]

# Seller LOO
s_tr = train_cats["seller_id"][in_tr_idx]
s_va = train_cats["seller_id"][in_val_idx]
y_tr = y_train[in_tr_idx]
prior = float(np.mean(y_tr))
m = 10.0
stats = pd.DataFrame({"s": s_tr, "y": y_tr}).groupby("s")["y"].agg(["count", "sum"])
s_map = ((stats["sum"] + m * prior) / (stats["count"] + m)).to_dict()
s_ser = pd.Series(s_tr)
c_cnt = s_ser.map(stats["count"]).fillna(0.0).values
c_sum = s_ser.map(stats["sum"]).fillna(0.0).values
df_tr_b["seller_te_loo"] = np.where(c_cnt > 1.0, (c_sum - y_tr + m * prior) / (c_cnt - 1.0 + m), prior)
df_va_b["seller_te_loo"] = pd.Series(s_va).map(s_map).fillna(prior).values

# Model B1: LightGBM
t_m = time.time()
lgb_model = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=63, max_depth=8, min_child_samples=30, n_jobs=4, verbose=-1, random_state=42)
lgb_model.fit(df_tr_b, y_tr)
p_lgb = lgb_model.predict_proba(df_va_b)[:, 1]
print(f"LGBM in {time.time() - t_m:.2f}s, Acc: {accuracy_score(y_train[in_val_idx], (p_lgb>=0.5).astype(int)):.4f}, LL: {log_loss(y_train[in_val_idx], p_lgb):.4f}")

# Model B2: CatBoost
t_m = time.time()
cb_model = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=7, thread_count=4, verbose=0, random_seed=42)
cb_model.fit(df_tr_b, y_tr)
p_cb = cb_model.predict_proba(df_va_b)[:, 1]
print(f"CatBoost in {time.time() - t_m:.2f}s, Acc: {accuracy_score(y_train[in_val_idx], (p_cb>=0.5).astype(int)):.4f}, LL: {log_loss(y_train[in_val_idx], p_cb):.4f}")

# Blend
p_blend = 0.35 * p_lr + 0.15 * p_ridge + 0.25 * p_lgb + 0.25 * p_cb
print(f"Simple Blend Acc: {accuracy_score(y_train[in_val_idx], (p_blend>=0.5).astype(int)):.4f}, LL: {log_loss(y_train[in_val_idx], p_blend):.4f}")
