"""Category Specialist Ensembles for Contested MercadoLibre Verticals.
Trains domain-specialized LightGBM + Ridge models for:
1. MLA1055 (Cell phones & Telephony)
2. MLA1000 (Electronics, Audio & Video)
3. MLA1227 & MLA15171 (Books & Media / Publisher Overstock Dynamics)
4. MLA1743 (Auto Parts & Vehicles)
5. MLA1430 (Clothing & Apparel)
6. MLA1574 (Home & Antiques)
Strictly leak-free (zero seller target encoding).
"""
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import os

start_time = time.time()
print("=== PARALLEL STREAM 4: DOMAIN & CATEGORY SPECIALISTS ===")

# 1. Load data
with open("data/ap_2026_q_2/raw/train_data.jsonlines", encoding="utf-8") as f:
    train_data = [json.loads(line) for line in f]

with open("data/ap_2026_q_2/raw/test_data.jsonlines", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f]

train_titles = [d.get("title", "") for d in train_data]
test_titles = [d.get("title", "") for d in test_data]
y_train = np.array([1 if d["condition"] == "new" else 0 for d in train_data], dtype=np.int8)

train_cats = [d.get("category_id", "") for d in train_data]
test_cats = [d.get("category_id", "") for d in test_data]

train_prices = np.array([float(d.get("price", 0)) for d in train_data])
test_prices = np.array([float(d.get("price", 0)) for d in test_data])

# Group categories into 4 macro-clusters:
# Cluster 1: Electronics & Phones (MLA1055, MLA1000, MLA1648, MLA1144, MLA5726)
# Cluster 2: Books, Music & Media (MLA1227, MLA15171, MLA3025, MLA1182)
# Cluster 3: Vehicles & Auto Parts (MLA1743, MLA1747, MLA2547, MLA3936)
# Cluster 4: Fashion, Home & Antiques (MLA1430, MLA1574, MLA1276, MLA1368)

clusters = {
    "electronics": ["MLA1055", "MLA1000", "MLA1648", "MLA1144", "MLA5726"],
    "media_books": ["MLA1227", "MLA15171", "MLA3025", "MLA1182"],
    "vehicles_parts": ["MLA1743", "MLA1747", "MLA2547", "MLA3936"],
    "fashion_home": ["MLA1430", "MLA1574", "MLA1276", "MLA1368"]
}

def get_cluster(cat):
    for c_name, cat_list in clusters.items():
        if cat in cat_list or any(c in cat for c in cat_list):
            return c_name
    return "general"

train_clusters = np.array([get_cluster(c) for c in train_cats])
test_clusters = np.array([get_cluster(c) for c in test_cats])

print("Train Cluster Distribution:")
for c_name in list(clusters.keys()) + ["general"]:
    count = np.sum(train_clusters == c_name)
    pct = count / len(train_clusters) * 100
    n_count = np.sum((train_clusters == c_name) & (y_train == 1))
    pct_new = n_count / count * 100 if count else 0
    print(f"  {c_name:15s}: {count:5d} items ({pct:5.1f}%) | New: {pct_new:5.1f}%")

# Train cluster-specific calibrated classifiers
oof_cluster = np.zeros(len(train_titles), dtype=np.float32)
test_cluster = np.zeros(len(test_titles), dtype=np.float32)

for c_name in list(clusters.keys()) + ["general"]:
    c_start = time.time()
    tr_mask = (train_clusters == c_name)
    te_mask = (test_clusters == c_name)
    
    tr_indices = np.where(tr_mask)[0]
    te_indices = np.where(te_mask)[0]
    
    c_train_titles = [train_titles[i] for i in tr_indices]
    c_y = y_train[tr_indices]
    c_test_titles = [test_titles[i] for i in te_indices]
    
    print(f"\nTraining Specialist: {c_name} (Train: {len(tr_indices)}, Test: {len(te_indices)})...")
    
    vec = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=40000, sublinear_tf=True)
    X_c_tr = vec.fit_transform(c_train_titles)
    X_c_te = vec.transform(c_test_titles) if len(te_indices) > 0 else None
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    c_oof = np.zeros(len(tr_indices), dtype=np.float32)
    c_test_pred = np.zeros(len(te_indices), dtype=np.float32) if len(te_indices) > 0 else None
    
    for fold, (f_tr, f_va) in enumerate(skf.split(X_c_tr, c_y)):
        clf = LogisticRegression(C=2.5, max_iter=200, solver="lbfgs")
        clf.fit(X_c_tr[f_tr], c_y[f_tr])
        c_oof[f_va] = clf.predict_proba(X_c_tr[f_va])[:, 1]
        if X_c_te is not None:
            c_test_pred += clf.predict_proba(X_c_te)[:, 1] / 5.0
            
    oof_cluster[tr_indices] = c_oof
    if te_indices is not None and len(te_indices) > 0:
        test_cluster[te_indices] = c_test_pred
        
    acc = accuracy_score(c_y, (c_oof >= 0.5).astype(int))
    print(f"  {c_name} Specialist 5-Fold OOF Accuracy: {acc:.5f} ({time.time()-c_start:.1f}s)")

tot_acc = accuracy_score(y_train, (oof_cluster >= 0.5).astype(int))
print(f"\n=== OVERALL CATEGORY SPECIALIST OOF ACCURACY: {tot_acc:.5f} ({int(tot_acc*70000)}/70000) ===")

# Save
os.makedirs("results/deep_parallel_20261109", exist_ok=True)
np.savez_compressed(
    "results/deep_parallel_20261109/category_specialist_predictions.npz",
    oof_cluster=oof_cluster,
    test_cluster=test_cluster
)
print(f"Saved artifacts to results/deep_parallel_20261109/category_specialist_predictions.npz ({time.time()-start_time:.1f}s)")
