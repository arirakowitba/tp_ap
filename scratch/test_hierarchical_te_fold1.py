import json
import time
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

def main():
    print("=== Testing Hierarchical Seller TE in LightGBM Fold 1 ===")
    t0 = time.time()

    # Load data
    with open("data/ap_2026_q_2/raw/tr.pkl", "rb") as f:
        df_tr = pickle.load(f)

    y_train = (df_tr["condition"] == "used").astype(int).values
    n_train = len(y_train)

    # Keywords & text features
    used_kws = ["antiguo", "antigua", "antiguos", "antiguas", "vintage", "retro", "impecable", "impecables", "poco uso", "como nuevo", "despiece", "usado", "usada"]
    new_kws = ["nuevo", "nueva", "nuevos", "nuevas", "sellado", "en caja", "0km", "cero km", "garantia oficial", "de fabrica"]

    titles = df_tr["title"].fillna("").astype(str)
    t_lower = titles.str.lower()
    has_used_kw = t_lower.apply(lambda s: sum(1 for kw in used_kws if kw in s))
    has_new_kw = t_lower.apply(lambda s: sum(1 for kw in new_kws if kw in s))
    char_len = titles.apply(len)
    word_count = titles.apply(lambda s: len(s.split()))
    digit_count = titles.apply(lambda s: sum(c.isdigit() for c in s))

    text_ind_train = np.column_stack([
        has_used_kw, has_new_kw, char_len, word_count, digit_count,
        titles.str.contains("nuevo", case=False).astype(int),
        titles.str.contains("usad", case=False).astype(int),
        titles.str.contains("impecable", case=False).astype(int),
        titles.str.contains("caja", case=False).astype(int),
        titles.str.contains("garant", case=False).astype(int),
        titles.str.contains("original", case=False).astype(int)
    ])

    # Text TF-IDF + LogisticRegression
    corpus_tr = (df_tr["title"].fillna("") + " " + df_tr["tags"].fillna("").astype(str)).values
    vec_w = TfidfVectorizer(ngram_range=(1, 2), max_features=40000, min_df=3, sublinear_tf=True)
    vec_c = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=60000, min_df=5, sublinear_tf=True)
    X_tr_w = vec_w.fit_transform(corpus_tr)
    X_tr_c = vec_c.fit_transform(corpus_tr)
    X_tr_text = hstack([X_tr_w, X_tr_c]).tocsr()

    lr_text = LogisticRegression(C=2.5, solver="saga", max_iter=250, random_state=42, n_jobs=8)
    lr_text.fit(X_tr_text, y_train)
    p_text_tr = lr_text.predict_proba(X_tr_text)[:, 1]

    # SVD
    svd = TruncatedSVD(n_components=64, random_state=42)
    X_tr_svd = svd.fit_transform(X_tr_text)

    # Basic tabular features
    price_tr = df_tr["price"].fillna(0).values
    log_price_tr = np.log1p(np.maximum(0, price_tr))
    cat_price_median = df_tr.groupby("category_id")["price"].median()
    tr_cat_med = df_tr["category_id"].map(cat_price_median).fillna(100.0).values
    tr_price_ratio = np.clip(price_tr / (tr_cat_med + 1.0), 0, 50)
    tr_num_attrs = df_tr["attributes"].apply(lambda x: len(x) if isinstance(x, list) else 0).values

    def parse_warranty(w):
        if not w or pd.isna(w): return 0
        w_str = str(w).lower()
        return 1 if any(k in w_str for k in ["mes", "año", "garant"]) else 0

    tr_warr = df_tr["warranty"].apply(parse_warranty).values
    tr_init_qty = np.log1p(df_tr["initial_quantity"].fillna(1).values)

    # 5-fold CV target encodings
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cat_te_oof = np.zeros(n_train)
    seller_te_global = np.zeros(n_train)
    seller_te_hier = np.zeros(n_train)

    global_prior = float(y_train.mean())
    m_seller = 5.0
    m_cat = 10.0

    df_tr_target = df_tr.copy()
    df_tr_target["target"] = y_train

    for tr_idx, val_idx in cv.split(df_tr, y_train):
        fold_tr = df_tr_target.iloc[tr_idx]
        fold_val = df_tr_target.iloc[val_idx]

        # Category TE
        cat_stats = fold_tr.groupby("category_id")["target"].agg(["count", "sum"])
        cat_priors = (cat_stats["sum"] + m_cat * global_prior) / (cat_stats["count"] + m_cat)
        cat_te_oof[val_idx] = fold_val["category_id"].map(cat_priors).fillna(global_prior).values

        # Global seller TE (Podium)
        seller_stats = fold_tr.groupby("seller_id")["target"].agg(["count", "sum"])
        s_smoothed_global = (seller_stats["sum"] + m_seller * global_prior) / (seller_stats["count"] + m_seller)
        seller_te_global[val_idx] = fold_val["seller_id"].map(s_smoothed_global).fillna(global_prior).values

        # Hierarchical Category-Conditioned seller TE
        seller_cat_mode = fold_tr.groupby("seller_id")["category_id"].agg(lambda x: x.mode().iloc[0] if len(x)>0 else "unknown")
        seller_prior_p = seller_cat_mode.map(cat_priors).fillna(global_prior)
        s_hier = (seller_stats["sum"] + m_seller * seller_prior_p) / (seller_stats["count"] + m_seller)
        val_item_cat_priors = fold_val["category_id"].map(cat_priors).fillna(global_prior)
        seller_te_hier[val_idx] = fold_val["seller_id"].map(s_hier).fillna(val_item_cat_priors).values

    # Base tabular feature block
    base_tab = np.column_stack([
        log_price_tr, tr_price_ratio, tr_num_attrs, tr_warr, tr_init_qty,
        cat_te_oof, text_ind_train, p_text_tr, X_tr_svd
    ])

    # 1. Baseline feature matrix (Podium Engine)
    X_base = np.column_stack([base_tab, seller_te_global])

    # 2. Challenger 1 (Hierarchical TE replaces Global)
    X_chall1 = np.column_stack([base_tab, seller_te_hier])

    # 3. Challenger 2 (Both: Global TE + Hierarchical TE + Delta)
    te_delta = seller_te_hier - cat_te_oof
    X_chall2 = np.column_stack([base_tab, seller_te_global, seller_te_hier, te_delta])

    # Test on Fold 1
    trn_idx, val_idx = next(cv.split(df_tr, y_train))
    y_val = y_train[val_idx]

    lgb_params = {
        "n_estimators": 900,
        "learning_rate": 0.025,
        "num_leaves": 55,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "min_child_samples": 20,
        "n_jobs": 8,
        "verbose": -1,
        "random_state": 42
    }

    print("\n--- Training Baseline LightGBM (Podium Style with Global Seller TE) ---")
    clf_base = lgb.LGBMClassifier(**lgb_params)
    clf_base.fit(X_base[trn_idx], y_train[trn_idx])
    p_base = clf_base.predict_proba(X_base[val_idx])[:, 1]
    acc_base = accuracy_score(y_val, (p_base >= 0.5).astype(int))
    hits_base = int(round(acc_base * len(val_idx)))
    ll_base = log_loss(y_val, p_base)
    auc_base = roc_auc_score(y_val, p_base)
    print(f"Baseline: Hits={hits_base}/14000, Acc={acc_base:.5f}, LL={ll_base:.5f}, AUC={auc_base:.5f}")

    print("\n--- Training Challenger 1 (Hierarchical Category-Conditioned Seller TE) ---")
    clf_c1 = lgb.LGBMClassifier(**lgb_params)
    clf_c1.fit(X_chall1[trn_idx], y_train[trn_idx])
    p_c1 = clf_c1.predict_proba(X_chall1[val_idx])[:, 1]
    acc_c1 = accuracy_score(y_val, (p_c1 >= 0.5).astype(int))
    hits_c1 = int(round(acc_c1 * len(val_idx)))
    ll_c1 = log_loss(y_val, p_c1)
    auc_c1 = roc_auc_score(y_val, p_c1)
    print(f"Challenger 1: Hits={hits_c1}/14000, Acc={acc_c1:.5f} ({hits_c1 - hits_base:+d}), LL={ll_c1:.5f}, AUC={auc_c1:.5f}")

    print("\n--- Training Challenger 2 (Enriched: Global + Hierarchical + Delta) ---")
    clf_c2 = lgb.LGBMClassifier(**lgb_params)
    clf_c2.fit(X_chall2[trn_idx], y_train[trn_idx])
    p_c2 = clf_c2.predict_proba(X_chall2[val_idx])[:, 1]
    acc_c2 = accuracy_score(y_val, (p_c2 >= 0.5).astype(int))
    hits_c2 = int(round(acc_c2 * len(val_idx)))
    ll_c2 = log_loss(y_val, p_c2)
    auc_c2 = roc_auc_score(y_val, p_c2)
    print(f"Challenger 2: Hits={hits_c2}/14000, Acc={acc_c2:.5f} ({hits_c2 - hits_base:+d}), LL={ll_c2:.5f}, AUC={auc_c2:.5f}")

    print(f"\nCompleted in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
