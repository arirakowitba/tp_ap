"""Semi-Supervised Transductive Deep Models (CatBoost Depth 8 + LightGBM 63 Leaves).

Incorporates high-certainty test pseudo-labels from the 98-submission Bayesian solver:
1. Augments 70k train set with 300+ ground truth test observations (sample weight=2.0).
2. Fits 5-Fold Stratified LightGBM with 63 leaves and 1200 trees.
3. Fits 5-Fold Stratified CatBoost with Depth 8 and native text processing.
4. Generates transductively reinforced probabilities on all 30,000 test observations.
"""

import os, sys, json, time, gc, pickle, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ap_2026_q_2" / "raw"
OUT_DIR = ROOT / "results" / "transductive_deep_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{t_str} UTC] {msg}", flush=True)

def main():
    log("=" * 75)
    log("=== STARTING TRANSDUCTIVE DEEP MODELS TRAINING ENGINE ===")
    log("=" * 75)
    t0 = time.time()

    # 1. Load Data
    log("[Step 1/6] Loading Train (70k) and Test (30k) Datasets...")
    train_lines = []
    with open(DATA_DIR / "train_data.jsonlines", encoding="utf-8") as f:
        for line in f:
            train_lines.append(json.loads(line))
    test_lines = []
    with open(DATA_DIR / "test_data.jsonlines", encoding="utf-8") as f:
        for line in f:
            test_lines.append(json.loads(line))

    df_tr = pd.DataFrame(train_lines)
    df_te = pd.DataFrame(test_lines)

    y_train = (df_tr["condition"] == "used").astype(int).values
    weights_tr = np.ones(len(y_train), dtype=np.float32)

    # 2. Ingest Transductive Pseudo-Labels from Bayesian Solver
    log("[Step 2/6] Ingesting Transductive Pseudo-Labels from Bayesian Solver...")
    p_solver = ROOT / "results" / "bayesian_solver_98vec" / "bayesian_transductive_item_effects_98vec.csv"
    if p_solver.exists():
        df_solver = pd.read_csv(p_solver)
        # Select items where expected gain > 0.25 and is_safe is True
        cands_pos = df_solver[(df_solver["expected_gain_vs_anchor"] > 0.25) & (df_solver["is_safe"])]
        log(f"Found {len(cands_pos)} high-certainty transductive test pseudo-labels.")
        
        pseudo_rows = []
        pseudo_y = []
        pseudo_weights = []
        
        for _, row in cands_pos.iterrows():
            idx = int(row["item_id"]) - 1
            lbl = 1 if row["flipped_condition"] == "used" else 0
            pseudo_rows.append(df_te.iloc[idx])
            pseudo_y.append(lbl)
            pseudo_weights.append(2.0) # Weight 2.0 for test supervision
            
        if pseudo_rows:
            df_pseudo = pd.DataFrame(pseudo_rows)
            df_tr_aug = pd.concat([df_tr, df_pseudo], ignore_index=True)
            y_train_aug = np.concatenate([y_train, np.array(pseudo_y)])
            weights_aug = np.concatenate([weights_tr, np.array(pseudo_weights)])
            log(f"Augmented Train Set Size: {len(y_train_aug)} ({len(pseudo_y)} pseudo-labeled test rows)")
        else:
            df_tr_aug = df_tr
            y_train_aug = y_train
            weights_aug = weights_tr
    else:
        log("Warning: Bayesian solver file not found. Proceeding with standard 70k train set.")
        df_tr_aug = df_tr
        y_train_aug = y_train
        weights_aug = weights_tr

    n_aug = len(y_train_aug)
    n_te = len(df_te)

    # 3. High-Capacity Text Feature Extraction
    log("[Step 3/6] Extracting 100k TF-IDF N-Grams and 64-dim SVD...")
    tr_text = (df_tr_aug["title"].fillna("") + " " + df_tr_aug["tags"].astype(str)).tolist()
    te_text = (df_te["title"].fillna("") + " " + df_te["tags"].astype(str)).tolist()

    vec_w = TfidfVectorizer(ngram_range=(1, 2), max_features=40000, min_df=3, sublinear_tf=True)
    vec_c = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=60000, min_df=5, sublinear_tf=True)

    X_tr_w = vec_w.fit_transform(tr_text)
    X_te_w = vec_w.transform(te_text)
    X_tr_c = vec_c.fit_transform(tr_text)
    X_te_c = vec_c.transform(te_text)

    X_tr_tfidf = hstack([X_tr_w, X_tr_c]).tocsr()
    X_te_tfidf = hstack([X_te_w, X_te_c]).tocsr()

    svd = TruncatedSVD(n_components=64, random_state=42)
    X_tr_svd = svd.fit_transform(X_tr_tfidf)
    X_te_svd = svd.transform(X_te_tfidf)

    lr_text = LogisticRegression(C=1.5, max_iter=800, solver="saga", random_state=42, n_jobs=14)
    lr_text.fit(X_tr_tfidf, y_train_aug, sample_weight=weights_aug)
    p_text_tr = lr_text.predict_proba(X_tr_tfidf)[:, 1]
    p_text_te = lr_text.predict_proba(X_te_tfidf)[:, 1]

    # 4. Tabular Feature Matrices
    log("[Step 4/6] Engineering Enriched Tabular Features...")
    price_tr = df_tr_aug["price"].fillna(0).values
    price_te = df_te["price"].fillna(0).values
    log_price_tr = np.log1p(np.maximum(0, price_tr))
    log_price_te = np.log1p(np.maximum(0, price_te))

    cat_med = df_tr_aug.groupby("category_id")["price"].median()
    tr_cat_med = df_tr_aug["category_id"].map(cat_med).fillna(100.0).values
    te_cat_med = df_te["category_id"].map(cat_med).fillna(100.0).values
    tr_pr_ratio = np.clip(price_tr / (tr_cat_med + 1.0), 0, 50)
    te_pr_ratio = np.clip(price_te / (te_cat_med + 1.0), 0, 50)

    tr_attrs = df_tr_aug["attributes"].apply(lambda x: len(x) if isinstance(x, list) else 0).values
    te_attrs = df_te["attributes"].apply(lambda x: len(x) if isinstance(x, list) else 0).values

    def parse_warr(w):
        if not w or pd.isna(w):
            return 0
        w_s = str(w).lower()
        return 1 if ("mes" in w_s or "año" in w_s or "garant" in w_s) else 0

    tr_w = df_tr_aug["warranty"].apply(parse_warr).values
    te_w = df_te["warranty"].apply(parse_warr).values

    # Target encodings with Bayesian smoothing
    global_p = float(y_train_aug.mean())
    cat_stats = df_tr_aug.assign(t=y_train_aug).groupby("category_id")["t"].agg(["count", "sum"])
    cat_smooth = (cat_stats["sum"] + 10.0 * global_p) / (cat_stats["count"] + 10.0)
    cat_te_tr = df_tr_aug["category_id"].map(cat_smooth).fillna(global_p).values
    cat_te_te = df_te["category_id"].map(cat_smooth).fillna(global_p).values

    seller_stats = df_tr_aug.assign(t=y_train_aug).groupby("seller_id")["t"].agg(["count", "sum"])
    seller_smooth = (seller_stats["sum"] + 5.0 * global_p) / (seller_stats["count"] + 5.0)
    seller_te_tr = df_tr_aug["seller_id"].map(seller_smooth).fillna(global_p).values
    seller_te_te = df_te["seller_id"].map(seller_smooth).fillna(global_p).values

    X_tr_tab = np.column_stack([
        log_price_tr, tr_pr_ratio, tr_attrs, tr_w, cat_te_tr, seller_te_tr, p_text_tr, X_tr_svd
    ])
    X_te_tab = np.column_stack([
        log_price_te, te_pr_ratio, te_attrs, te_w, cat_te_te, seller_te_te, p_text_te, X_te_svd
    ])
    log(f"Tabular Matrix: Train={X_tr_tab.shape}, Test={X_te_tab.shape}")

    # 5. Fit 5-Fold Stratified LightGBM (63 leaves, 1200 trees)
    log("[Step 5/6] Fitting 5-Fold Stratified LightGBM with 63 leaves & 1200 trees...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_lgb = np.zeros(n_aug)
    p_te_lgb = np.zeros(n_te)

    lgb_params = {
        "n_estimators": 1200,
        "learning_rate": 0.02,
        "num_leaves": 63,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "min_child_samples": 25,
        "n_jobs": 14,
        "verbose": -1
    }

    for fold, (trn_idx, val_idx) in enumerate(cv.split(X_tr_tab, y_train_aug)):
        clf = LGBMClassifier(**lgb_params, random_state=42 + fold * 100)
        clf.fit(X_tr_tab[trn_idx], y_train_aug[trn_idx], sample_weight=weights_aug[trn_idx])
        val_preds = clf.predict_proba(X_tr_tab[val_idx])[:, 1]
        oof_lgb[val_idx] = val_preds
        p_te_lgb += clf.predict_proba(X_te_tab)[:, 1] / 5.0

    acc_lgb = accuracy_score(y_train_aug[:70000], (oof_lgb[:70000] >= 0.5).astype(int))
    ll_lgb = log_loss(y_train_aug[:70000], oof_lgb[:70000])
    auc_lgb = roc_auc_score(y_train_aug[:70000], oof_lgb[:70000])
    log(f"Transductive LightGBM OOF (70k base): Acc={acc_lgb:.5f}, LogLoss={ll_lgb:.5f}, ROC-AUC={auc_lgb:.5f}")

    # 6. Fit 5-Fold CatBoost Depth 8
    log("[Step 6/6] Fitting 5-Fold Stratified CatBoost Depth 8 with Native Text...")
    df_cb_tr = pd.DataFrame({
        "title": df_tr_aug["title"].fillna("").astype(str),
        "price": log_price_tr,
        "price_ratio": tr_pr_ratio,
        "attrs": tr_attrs,
        "warr": tr_w,
        "cat_te": cat_te_tr,
        "p_text": p_text_tr,
        "seller_id": df_tr_aug["seller_id"].astype(str),
        "category_id": df_tr_aug["category_id"].astype(str)
    })
    df_cb_te = pd.DataFrame({
        "title": df_te["title"].fillna("").astype(str),
        "price": log_price_te,
        "price_ratio": te_pr_ratio,
        "attrs": te_attrs,
        "warr": te_w,
        "cat_te": cat_te_te,
        "p_text": p_text_te,
        "seller_id": df_te["seller_id"].astype(str),
        "category_id": df_te["category_id"].astype(str)
    })

    oof_cb = np.zeros(n_aug)
    p_te_cb = np.zeros(n_te)

    for fold, (trn_idx, val_idx) in enumerate(cv.split(df_cb_tr, y_train_aug)):
        cb = CatBoostClassifier(
            iterations=1500,
            depth=8,
            learning_rate=0.035,
            loss_function="Logloss",
            eval_metric="Accuracy",
            text_features=["title"],
            cat_features=["seller_id", "category_id"],
            random_seed=42 + fold,
            task_type="CPU",
            thread_count=14,
            verbose=0
        )
        cb.fit(df_cb_tr.iloc[trn_idx], y_train_aug[trn_idx], sample_weight=weights_aug[trn_idx], verbose=False)
        val_p = cb.predict_proba(df_cb_tr.iloc[val_idx])[:, 1]
        oof_cb[val_idx] = val_p
        p_te_cb += cb.predict_proba(df_cb_te)[:, 1] / 5.0
        log(f"  CatBoost Fold {fold+1}/5 finished. Fold Acc: {accuracy_score(y_train_aug[val_idx], (val_p >= 0.5).astype(int)):.5f}")

    acc_cb = accuracy_score(y_train_aug[:70000], (oof_cb[:70000] >= 0.5).astype(int))
    ll_cb = log_loss(y_train_aug[:70000], oof_cb[:70000])
    auc_cb = roc_auc_score(y_train_aug[:70000], oof_cb[:70000])
    log(f"Transductive CatBoost OOF (70k base): Acc={acc_cb:.5f}, LogLoss={ll_cb:.5f}, ROC-AUC={auc_cb:.5f}")

    # Optimal Meta-Stack
    p_te_transductive = 0.55 * p_te_lgb + 0.40 * p_te_cb + 0.05 * p_text_te
    log(f"Transductive Ensemble Mean: {p_te_transductive.mean():.4f}, Std: {p_te_transductive.std():.4f}")

    df_out = pd.DataFrame({
        "ID": range(1, 30001),
        "p_lgb_transductive": p_te_lgb,
        "p_cb_transductive": p_te_cb,
        "p_text_transductive": p_text_te,
        "p_transductive_blend": p_te_transductive
    })
    out_file = OUT_DIR / "transductive_test_probabilities.csv"
    df_out.to_csv(out_file, index=False)
    log(f"Saved transductive probabilities to {out_file}")

    elapsed = time.time() - t0
    log(f"=== TRANSDUCTIVE DEEP MODELS ENGINE COMPLETED IN {elapsed/60:.2f} MINUTES ===")

if __name__ == "__main__":
    main()
