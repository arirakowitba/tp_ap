"""Granular Category Specialist Suite for Top 10 Contested Verticals.
Trains vertical-specific models with dedicated vocabulary and price distribution:
1. MLA1055 (Celulares y Teléfonos)
2. MLA1000 (Electrónica, Audio y Video)
3. MLA1227 (Libros, Revistas y Cómics)
4. MLA15171 (Música, Películas y Series)
5. MLA1743 (Autos, Motos y Otros)
6. MLA1430 (Ropa y Accesorios)
7. MLA1574 (Hogar, Muebles y Jardín)
8. MLA1144 (Consolas y Videojuegos)
9. MLA3025 (Coleccionables y Hobbies)
10. MLA1051 (Celulares y Smartphones)
Strictly leak-free (zero seller-id features).
"""
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score
import os

start_time = time.time()
print("=== CONTINUOUS OVERNIGHT PHASE 3: GRANULAR CATEGORY SPECIALISTS ===")

# 1. Load Data
with open("data/ap_2026_q_2/raw/train_data.jsonlines", encoding="utf-8") as f:
    train_data = [json.loads(line) for line in f]

with open("data/ap_2026_q_2/raw/test_data.jsonlines", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f]

train_titles = [d.get("title", "") for d in train_data]
test_titles = [d.get("title", "") for d in test_data]
y_train = np.array([1 if d["condition"] == "new" else 0 for d in train_data], dtype=np.int8)

train_cats = np.array([d.get("category_id", "") for d in train_data])
test_cats = np.array([d.get("category_id", "") for d in test_data])

train_prices = np.array([float(d.get("price", 0)) for d in train_data])
test_prices = np.array([float(d.get("price", 0)) for d in test_data])

# Top 10 categories
top_cats = [
    "MLA1055", "MLA1000", "MLA1227", "MLA15171", "MLA1743",
    "MLA1430", "MLA1574", "MLA1144", "MLA3025", "MLA1051"
]

oof_specialists = np.zeros(len(train_titles), dtype=np.float32)
test_specialists = np.zeros(len(test_titles), dtype=np.float32)

# Also train general fallback model
vec_gen = TfidfVectorizer(ngram_range=(1, 3), min_df=3, max_features=80000, sublinear_tf=True)
X_gen_tr = vec_gen.fit_transform(train_titles)
X_gen_te = vec_gen.transform(test_titles)

clf_gen = LogisticRegression(C=2.0, max_iter=200, solver="lbfgs", n_jobs=8)
clf_gen.fit(X_gen_tr, y_train)
fallback_test = clf_gen.predict_proba(X_gen_te)[:, 1]

for cat_id in top_cats:
    c_start = time.time()
    tr_mask = (train_cats == cat_id)
    te_mask = (test_cats == cat_id)
    
    tr_idx = np.where(tr_mask)[0]
    te_idx = np.where(te_mask)[0]
    
    n_tr = len(tr_idx)
    n_te = len(te_idx)
    
    if n_tr < 30:
        continue
        
    c_titles = [train_titles[i] for i in tr_idx]
    c_y = y_train[tr_idx]
    c_prices = train_prices[tr_idx]
    
    # TF-IDF on this vertical's exact vocabulary
    vec = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=25000, sublinear_tf=True)
    X_tr_vec = vec.fit_transform(c_titles)
    
    if n_te > 0:
        c_te_titles = [test_titles[i] for i in te_idx]
        X_te_vec = vec.transform(c_te_titles)
    else:
        X_te_vec = None
        
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    c_oof = np.zeros(n_tr, dtype=np.float32)
    c_test = np.zeros(n_te, dtype=np.float32) if n_te > 0 else None
    
    for fold, (f_tr, f_va) in enumerate(skf.split(X_tr_vec, c_y)):
        clf = LogisticRegression(C=2.5, max_iter=200, solver="lbfgs")
        clf.fit(X_tr_vec[f_tr], c_y[f_tr])
        c_oof[f_va] = clf.predict_proba(X_tr_vec[f_va])[:, 1]
        if X_te_vec is not None:
            c_test += clf.predict_proba(X_te_vec)[:, 1] / 5.0
            
    oof_specialists[tr_idx] = c_oof
    if n_te > 0:
        test_specialists[te_idx] = c_test
        
    acc = accuracy_score(c_y, (c_oof >= 0.5).astype(int))
    print(f"Vertical {cat_id:8s} ({n_tr:4d} train, {n_te:4d} test) in {time.time()-c_start:4.1f}s | OOF Acc: {acc:.5f}")

# Fill non-top-10 items with fallback
non_top_te = np.where(test_specialists == 0)[0]
test_specialists[non_top_te] = fallback_test[non_top_te]

# Save
os.makedirs("results/category_specialists_20261110", exist_ok=True)
np.savez_compressed(
    "results/category_specialists_20261110/granular_specialist_predictions.npz",
    oof_specialists=oof_specialists,
    test_specialists=test_specialists
)
print(f"Saved artifacts to results/category_specialists_20261110/ ({time.time()-start_time:.1f}s)")
