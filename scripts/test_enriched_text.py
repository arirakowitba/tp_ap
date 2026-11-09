"""Test enriched text representation on df_fit 5-fold CV."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold

OUT_DIR = Path("results/dual_regime_20261109")
DATA_DIR = Path("data/ap_2026_q_2/raw")

with open(OUT_DIR / "frozen_dual_regime_indices.json", "r") as f:
    idx_data = json.load(f)
fit_idx = np.array(idx_data["fit_idx"])

df_train = pd.read_json(DATA_DIR / "train_data.jsonlines", lines=True)
df_fit = df_train.iloc[fit_idx].reset_index(drop=True)
y_fit = (df_fit["condition"] == "used").astype(np.int8).values

# 1. Baseline: title only
print("Preparing title-only text...")
titles = df_fit["title"].fillna("").astype(str).str.lower()

# 2. Rich augmented text: title + warranty + tags + top attributes
print("Preparing enriched text...")
def build_enriched(row):
    parts = [str(row.get("title") or "")]
    w = row.get("warranty")
    if w and str(w).strip():
        parts.append(f"garantia_{w}")
    tags = row.get("tags") or []
    for t in tags:
        parts.append(f"tag_{t}")
    attrs = row.get("attributes") or []
    for a in attrs:
        if isinstance(a, dict):
            name = a.get("id") or a.get("name") or ""
            val = a.get("value_name") or a.get("value_id") or ""
            if name or val:
                parts.append(f"attr_{name}_{val}")
    return " ".join(parts).lower()

enriched_texts = df_fit.apply(build_enriched, axis=1)

# Compare 5-fold CV
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for name, text_series in [("Title-Only", titles), ("Enriched-Text (Title+Tags+Attrs+Warranty)", enriched_texts)]:
    vec_word = TfidfVectorizer(max_features=35000, ngram_range=(1, 2), sublinear_tf=True)
    vec_char = TfidfVectorizer(max_features=25000, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    
    oof_scores = np.zeros(len(df_fit), dtype=np.float32)
    
    for tr_idx, va_idx in sgkf.split(df_fit, y_fit, df_fit["seller_id"].values):
        w_tr = vec_word.fit_transform(text_series.iloc[tr_idx])
        c_tr = vec_char.fit_transform(text_series.iloc[tr_idx])
        X_tr = sparse.hstack([w_tr, c_tr], format="csr")
        
        w_va = vec_word.transform(text_series.iloc[va_idx])
        c_va = vec_char.transform(text_series.iloc[va_idx])
        X_va = sparse.hstack([w_va, c_va], format="csr")
        
        m = RidgeClassifier(alpha=2.0, random_state=42)
        m.fit(X_tr, y_fit[tr_idx])
        scores = m.decision_function(X_va)
        oof_scores[va_idx] = 1.0 / (1.0 + np.exp(-scores))
        
    acc = accuracy_score(y_fit, (oof_scores >= 0.5))
    loss = log_loss(y_fit, oof_scores)
    print(f"{name:45s} | OOF Acc: {acc:.5f} | LogLoss: {loss:.5f}")
