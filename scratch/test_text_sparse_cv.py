"""Test sparse text classifier on Fold 1 of outer StratifiedGroupKFold seller CV.
"""
import json, time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

print("Loading train data...", flush=True)
t0 = time.time()
with open("data/ap_2026_q_2/raw/train_data.jsonlines", "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

y = np.array([1 if r.get("condition") == "used" else 0 for r in rows], dtype=int)
sellers = np.array([str(r.get("seller_id") or "UNKNOWN") for r in rows])
corpus = []
for r in rows:
    title = str(r.get("title") or "").strip()
    warr = str(r.get("warranty") or "").strip()
    attrs = [str(a.get("value_name") or a.get("name") or "") for a in (r.get("attributes") or []) if isinstance(a, dict)]
    corpus.append(f"{title} WARRANTY {warr} ATTR {' '.join(attrs)}")

print(f"Loaded {len(rows)} rows in {time.time()-t0:.2f}s.", flush=True)

# Split Fold 1
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261110)
for fold_i, (tr_idx, val_idx) in enumerate(sgkf.split(corpus, y, groups=sellers)):
    print(f"\nEvaluating Fold {fold_i + 1} with Sparse Text Classifier (Word 1-2 + Char 3-5)...", flush=True)
    t_feat = time.time()

    # Word TF-IDF
    vec_word = TfidfVectorizer(ngram_range=(1, 2), max_features=75000, sublinear_tf=True, min_df=2)
    X_tr_w = vec_word.fit_transform([corpus[i] for i in tr_idx])
    X_val_w = vec_word.transform([corpus[i] for i in val_idx])

    # Char TF-IDF
    vec_char = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=75000, sublinear_tf=True, min_df=3)
    X_tr_c = vec_char.fit_transform([corpus[i] for i in tr_idx])
    X_val_c = vec_char.transform([corpus[i] for i in val_idx])

    X_tr = sparse.hstack([X_tr_w, X_tr_c], format="csr")
    X_val = sparse.hstack([X_val_w, X_val_c], format="csr")
    print(f"Feature extraction completed in {time.time()-t_feat:.2f}s. Sparse matrix: {X_tr.shape}", flush=True)

    # Train Logistic Regression
    t_train = time.time()
    clf = LogisticRegression(C=2.0, max_iter=200, solver="saga", n_jobs=4, random_state=42)
    clf.fit(X_tr, y[tr_idx])
    p_val = clf.predict_proba(X_val)[:, 1]
    print(f"Trained in {time.time()-t_train:.2f}s.", flush=True)

    acc = accuracy_score(y[val_idx], (p_val >= 0.5).astype(int))
    ll = log_loss(y[val_idx], p_val)
    auc = roc_auc_score(y[val_idx], p_val)
    print(f"Fold 1 Sparse Text: Acc = {acc:.5f}, LogLoss = {ll:.5f}, ROC-AUC = {auc:.5f}")

    # Load Tri-Titan OOF on Fold 1 to test blend
    df_trees = pd.read_csv("results/seller_group_validation_20261110/oof_predictions_all_models.csv")
    p_trees_val = df_trees.iloc[val_idx]["tri_titan_arm_a"].values
    acc_trees = accuracy_score(y[val_idx], (p_trees_val >= 0.5).astype(int))
    print(f"Fold 1 Tri-Titan Trees: Acc = {acc_trees:.5f}")

    for w in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        p_blend = w * p_val + (1 - w) * p_trees_val
        acc_b = accuracy_score(y[val_idx], (p_blend >= 0.5).astype(int))
        ll_b = log_loss(y[val_idx], p_blend)
        print(f"  Blend (w_text={w:.1f}): Acc = {acc_b:.5f}, LogLoss = {ll_b:.5f} (Δ vs Trees = {acc_b - acc_trees:+.5f})")
    break
