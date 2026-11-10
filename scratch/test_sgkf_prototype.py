"""Prototype for outer StratifiedGroupKFold seller CV feature pipeline.
"""
import time, json, gc
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression

print("Testing feature pipeline setup...", flush=True)
t0 = time.time()
with open("data/ap_2026_q_2/raw/train_data.jsonlines", "r", encoding="utf-8") as f:
    sample_rows = [json.loads(f.readline()) for _ in range(5000)]

titles = [r.get("title") or "" for r in sample_rows]
y = np.array([1 if r.get("condition") == "used" else 0 for r in sample_rows])
sellers = np.array([str(r.get("seller_id")) for r in sample_rows])

print(f"Loaded 5000 rows in {time.time()-t0:.2f}s. Unique sellers: {len(np.unique(sellers))}", flush=True)

vec = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), min_df=2)
X_tfidf = vec.fit_transform(titles)
print(f"TFIDF shape: {X_tfidf.shape}", flush=True)

svd = TruncatedSVD(n_components=50, random_state=42)
X_svd = svd.fit_transform(X_tfidf)
print(f"SVD shape: {X_svd.shape} in {time.time()-t0:.2f}s", flush=True)

# Test StratifiedGroupKFold
sgkf = StratifiedGroupKFold(n_splits=5)
for fold, (tr, va) in enumerate(sgkf.split(titles, y, groups=sellers)):
    tr_sellers = set(sellers[tr])
    va_sellers = set(sellers[va])
    assert len(tr_sellers & va_sellers) == 0
    print(f"Fold {fold}: tr={len(tr)}, va={len(va)}, tr_sellers={len(tr_sellers)}, va_sellers={len(va_sellers)}, overlap={len(tr_sellers & va_sellers)}")
    break

print("All prototype tests passed successfully!")
