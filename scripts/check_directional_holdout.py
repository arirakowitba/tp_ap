"""Inspect directional flip precision on frozen dual-regime holdout."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedGroupKFold

OUT_DIR = Path("results/dual_regime_20261109")
DATA_DIR = Path("data/ap_2026_q_2/raw")

sub_data = np.load(OUT_DIR / "subagent_trees_predictions.npz")
oof_cat = sub_data["oof_cat"]
oof_lgb = sub_data["oof_lgb"]
y_fit = sub_data["y_fit"]

holdout_seen_cat = sub_data["holdout_seen_cat"]
holdout_seen_lgb = sub_data["holdout_seen_lgb"]
y_seen = sub_data["y_seen"]
base_pred_seen = sub_data["base_pred_seen"]

holdout_unseen_cat = sub_data["holdout_unseen_cat"]
holdout_unseen_lgb = sub_data["holdout_unseen_lgb"]
y_unseen = sub_data["y_unseen"]
base_pred_unseen = sub_data["base_pred_unseen"]

with open(OUT_DIR / "frozen_dual_regime_indices.json", "r") as f:
    idx_data = json.load(f)
fit_idx = np.array(idx_data["fit_idx"])
seen_ho_idx = np.array(idx_data["seen_holdout_idx"])
unseen_ho_idx = np.array(idx_data["unseen_holdout_idx"])

df_train = pd.read_json(DATA_DIR / "train_data.jsonlines", lines=True)
df_fit = df_train.iloc[fit_idx].reset_index(drop=True)
df_seen_ho = df_train.iloc[seen_ho_idx].reset_index(drop=True)
df_unseen_ho = df_train.iloc[unseen_ho_idx].reset_index(drop=True)

titles_fit = df_fit["title"].fillna("").astype(str).str.lower()
titles_seen = df_seen_ho["title"].fillna("").astype(str).str.lower()
titles_unseen = df_unseen_ho["title"].fillna("").astype(str).str.lower()

vec_word = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True)
vec_char = TfidfVectorizer(max_features=25000, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_ridge = np.zeros(len(df_fit), dtype=np.float32)
for fold, (tr_idx, va_idx) in enumerate(sgkf.split(df_fit, y_fit, df_fit["seller_id"].values)):
    w_tr = vec_word.fit_transform(titles_fit.iloc[tr_idx])
    c_tr = vec_char.fit_transform(titles_fit.iloc[tr_idx])
    X_tr = sparse.hstack([w_tr, c_tr], format="csr")
    w_va = vec_word.transform(titles_fit.iloc[va_idx])
    c_va = vec_char.transform(titles_fit.iloc[va_idx])
    X_va = sparse.hstack([w_va, c_va], format="csr")
    m_rdg = RidgeClassifier(alpha=2.0, random_state=42)
    m_rdg.fit(X_tr, y_fit[tr_idx])
    scores = m_rdg.decision_function(X_va)
    oof_ridge[va_idx] = 1.0 / (1.0 + np.exp(-scores))

w_fit = vec_word.fit_transform(titles_fit)
c_fit = vec_char.fit_transform(titles_fit)
X_fit_full = sparse.hstack([w_fit, c_fit], format="csr")
m_rdg_full = RidgeClassifier(alpha=2.0, random_state=42)
m_rdg_full.fit(X_fit_full, y_fit)

w_seen = vec_word.transform(titles_seen)
c_seen = vec_char.transform(titles_seen)
X_seen_sparse = sparse.hstack([w_seen, c_seen], format="csr")
holdout_seen_ridge = 1.0 / (1.0 + np.exp(-m_rdg_full.decision_function(X_seen_sparse)))

w_unseen = vec_word.transform(titles_unseen)
c_unseen = vec_char.transform(titles_unseen)
X_unseen_sparse = sparse.hstack([w_unseen, c_unseen], format="csr")
holdout_unseen_ridge = 1.0 / (1.0 + np.exp(-m_rdg_full.decision_function(X_unseen_sparse)))

seller_counts = df_fit["seller_id"].value_counts()
is_multi_seller = df_fit["seller_id"].map(seller_counts > 1).values
meta_X_fit = np.column_stack([oof_cat, oof_lgb, oof_ridge])

meta_seen = LogisticRegression(C=1.0, random_state=42).fit(meta_X_fit[is_multi_seller], y_fit[is_multi_seller])
meta_unseen = LogisticRegression(C=1.0, random_state=42).fit(meta_X_fit[~is_multi_seller], y_fit[~is_multi_seller])

meta_X_seen = np.column_stack([holdout_seen_cat, holdout_seen_lgb, holdout_seen_ridge])
p_seen_meta = meta_seen.predict_proba(meta_X_seen)[:, 1]

meta_X_unseen = np.column_stack([holdout_unseen_cat, holdout_unseen_lgb, holdout_unseen_ridge])
p_unseen_meta = meta_unseen.predict_proba(meta_X_unseen)[:, 1]

p_holdout = np.concatenate([p_seen_meta, p_unseen_meta])
base_holdout = np.concatenate([base_pred_seen, base_pred_unseen])
y_holdout = np.concatenate([y_seen, y_unseen])

print("=" * 80)
print("ANALYZING DIRECTIONAL FLIP PRECISION ON DUAL-REGIME HOLDOUT (14,000 items)")
print("=" * 80)

for delta in [0.00, 0.10, 0.15, 0.20, 0.25, 0.30]:
    cand = (p_holdout >= 0.5).astype(np.int8)
    active_mask = np.abs(p_holdout - 0.5) >= delta
    
    # new -> used flips (base=0, cand=1)
    n2u = (base_holdout == 0) & (cand == 1) & active_mask
    fixes_n2u = int(np.sum((base_holdout[n2u] != y_holdout[n2u]) & (cand[n2u] == y_holdout[n2u])))
    breaks_n2u = int(np.sum((base_holdout[n2u] == y_holdout[n2u]) & (cand[n2u] != y_holdout[n2u])))
    prec_n2u = (fixes_n2u / max(fixes_n2u + breaks_n2u, 1)) * 100.0
    
    # used -> new flips (base=1, cand=0)
    u2n = (base_holdout == 1) & (cand == 0) & active_mask
    fixes_u2n = int(np.sum((base_holdout[u2n] != y_holdout[u2n]) & (cand[u2n] == y_holdout[u2n])))
    breaks_u2n = int(np.sum((base_holdout[u2n] == y_holdout[u2n]) & (cand[u2n] != y_holdout[u2n])))
    prec_u2n = (fixes_u2n / max(fixes_u2n + breaks_u2n, 1)) * 100.0
    
    print(f"delta = {delta:.2f}:")
    print(f"  NEW -> USED: {int(np.sum(n2u)):4d} flips | Fixes: {fixes_n2u:3d}, Breaks: {breaks_n2u:3d}, Net: {fixes_n2u-breaks_n2u:+4d} | Prec: {prec_n2u:5.1f}%")
    print(f"  USED -> NEW: {int(np.sum(u2n)):4d} flips | Fixes: {fixes_u2n:3d}, Breaks: {breaks_u2n:3d}, Net: {fixes_u2n-breaks_u2n:+4d} | Prec: {prec_u2n:5.1f}%")
    print(f"  COMBINED   : {int(np.sum(n2u+u2n)):4d} flips | Fixes: {fixes_n2u+fixes_u2n:3d}, Breaks: {breaks_n2u+breaks_u2n:3d}, Net: {(fixes_n2u-breaks_n2u)+(fixes_u2n-breaks_u2n):+4d} | Prec: {((fixes_n2u+fixes_u2n)/max(fixes_n2u+fixes_u2n+breaks_n2u+breaks_u2n,1))*100.0:5.1f}%\n")
