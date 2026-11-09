"""Inspect high-certainty disagreements with EVT-132."""

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
test_cat = sub_data["test_cat"]
test_lgb = sub_data["test_lgb"]

with open(OUT_DIR / "frozen_dual_regime_indices.json", "r") as f:
    idx_data = json.load(f)
fit_idx = np.array(idx_data["fit_idx"])

df_train = pd.read_json(DATA_DIR / "train_data.jsonlines", lines=True)
df_test = pd.read_json(DATA_DIR / "test_data.jsonlines", lines=True)
df_fit = df_train.iloc[fit_idx].reset_index(drop=True)

titles_fit = df_fit["title"].fillna("").astype(str).str.lower()
titles_test = df_test["title"].fillna("").astype(str).str.lower()

vec_word = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True)
vec_char = TfidfVectorizer(max_features=25000, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)

w_fit = vec_word.fit_transform(titles_fit)
c_fit = vec_char.fit_transform(titles_fit)
X_fit_full = sparse.hstack([w_fit, c_fit], format="csr")

m_rdg_full = RidgeClassifier(alpha=2.0, random_state=42)
m_rdg_full.fit(X_fit_full, y_fit)

w_test = vec_word.transform(titles_test)
c_test = vec_char.transform(titles_test)
X_test_sparse = sparse.hstack([w_test, c_test], format="csr")
test_ridge = 1.0 / (1.0 + np.exp(-m_rdg_full.decision_function(X_test_sparse)))

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

seller_counts = df_fit["seller_id"].value_counts()
is_multi = df_fit["seller_id"].map(seller_counts > 1).values
meta_X_fit = np.column_stack([oof_cat, oof_lgb, oof_ridge])
meta_seen = LogisticRegression(C=1.0, random_state=42).fit(meta_X_fit[is_multi], y_fit[is_multi])
meta_unseen = LogisticRegression(C=1.0, random_state=42).fit(meta_X_fit[~is_multi], y_fit[~is_multi])

meta_X_test = np.column_stack([test_cat, test_lgb, test_ridge])
is_seen_test = np.isin(df_test["seller_id"].values, df_fit["seller_id"].unique())
p_test = np.zeros(len(df_test), dtype=np.float32)
p_test[is_seen_test] = meta_seen.predict_proba(meta_X_test[is_seen_test])[:, 1]
p_test[~is_seen_test] = meta_unseen.predict_proba(meta_X_test[~is_seen_test])[:, 1]

evt = pd.read_csv("submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv")
y_evt = (evt["condition"] == "used").astype(int).values

for delta in [0.20, 0.30, 0.35, 0.40, 0.42, 0.45]:
    disagree_mask = ((p_test >= 0.5).astype(int) != y_evt) & (np.abs(p_test - 0.5) >= delta)
    print(f"delta = {delta:.2f}: {np.sum(disagree_mask)} flips vs EVT-132")

disagree_mask = ((p_test >= 0.5).astype(int) != y_evt) & (np.abs(p_test - 0.5) >= 0.35)
df_dis = df_test[disagree_mask].copy()
df_dis["p_meta"] = p_test[disagree_mask]
df_dis["evt_pred"] = np.where(y_evt[disagree_mask] == 1, "used", "new")
df_dis["meta_pred"] = np.where(p_test[disagree_mask] >= 0.5, "used", "new")

print(f"\nBreakdown at delta=0.35 ({len(df_dis)} flips):")
print(df_dis["meta_pred"].value_counts())

print("\nSample flips where EVT-132 said NEW, but Meta says USED (p > 0.85):")
for _, row in df_dis[df_dis["meta_pred"] == "used"].head(10).iterrows():
    print(f"  ID {row['id']}: p={row['p_meta']:.3f} | {row['title']}")

print("\nSample flips where EVT-132 said USED, but Meta says NEW (p < 0.15):")
for _, row in df_dis[df_dis["meta_pred"] == "new"].head(10).iterrows():
    print(f"  ID {row['id']}: p={row['p_meta']:.3f} | {row['title']}")
