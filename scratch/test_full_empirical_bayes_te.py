import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

def main():
    print("=== Testing Fully Empirical Bayes Category-Conditioned Smoothing ===")
    with open("data/ap_2026_q_2/raw/tr.pkl", "rb") as f:
        df_tr = pickle.load(f)

    y_train = (df_tr["condition"] == "used").astype(int).values
    df_tr["target"] = y_train
    n_train = len(df_tr)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    seller_te_global = np.zeros(n_train)
    seller_te_hier_full = np.zeros(n_train)

    global_prior = float(y_train.mean())
    m_seller = 5.0
    m_cat = 10.0

    for tr_idx, val_idx in cv.split(df_tr, y_train):
        fold_tr = df_tr.iloc[tr_idx]
        fold_val = df_tr.iloc[val_idx]

        # 1. Category priors
        cat_stats = fold_tr.groupby("category_id")["target"].agg(["count", "sum"])
        cat_priors = (cat_stats["sum"] + m_cat * global_prior) / (cat_stats["count"] + m_cat)

        # 2. Global smoothing (Podium Engine formula)
        seller_stats = fold_tr.groupby("seller_id")["target"].agg(["count", "sum"])
        s_smoothed_global = (seller_stats["sum"] + m_seller * global_prior) / (seller_stats["count"] + m_seller)
        seller_te_global[val_idx] = fold_val["seller_id"].map(s_smoothed_global).fillna(global_prior).values

        # 3. Fully Empirical Bayes category-conditioned smoothing
        # Map each item's seller in fold_tr to its category prior
        fold_tr_cats = fold_tr["category_id"].map(cat_priors).fillna(global_prior)
        
        # Calculate seller's primary category or mean category prior
        seller_cat_prior = fold_tr.groupby("seller_id")["category_id"].agg(lambda x: x.mode().iloc[0] if len(x)>0 else "unknown")
        seller_prior_p = seller_cat_prior.map(cat_priors).fillna(global_prior)

        # Category-conditioned seller posterior: (sum + m * prior_p) / (count + m)
        s_hier = (seller_stats["sum"] + m_seller * seller_prior_p) / (seller_stats["count"] + m_seller)

        # For validation: if seller unseen, use the item's category prior!
        val_item_cat_priors = fold_val["category_id"].map(cat_priors).fillna(global_prior)
        seller_te_hier_full[val_idx] = fold_val["seller_id"].map(s_hier).fillna(val_item_cat_priors).values

    print(f"Overall Global Seller TE LogLoss:       {log_loss(y_train, seller_te_global):.5f}")
    print(f"Overall Hierarchical Seller TE LogLoss: {log_loss(y_train, seller_te_hier_full):.5f}")
    print(f"Overall Global Seller TE ROC-AUC:       {roc_auc_score(y_train, seller_te_global):.5f}")
    print(f"Overall Hierarchical Seller TE ROC-AUC: {roc_auc_score(y_train, seller_te_hier_full):.5f}")

    # Evaluate on seen sellers only
    seen_masks = np.zeros(n_train, dtype=bool)
    for tr_idx, val_idx in cv.split(df_tr, y_train):
        train_sellers = set(df_tr.iloc[tr_idx]["seller_id"])
        val_sellers = df_tr.iloc[val_idx]["seller_id"]
        seen_masks[val_idx] = val_sellers.isin(train_sellers).values

    print(f"\nSeen Sellers ({seen_masks.sum()} items, {seen_masks.sum()/n_train*100:.1f}%):")
    print(f"  LogLoss: Global={log_loss(y_train[seen_masks], seller_te_global[seen_masks]):.5f} -> Hier={log_loss(y_train[seen_masks], seller_te_hier_full[seen_masks]):.5f}")
    print(f"  ROC-AUC: Global={roc_auc_score(y_train[seen_masks], seller_te_global[seen_masks]):.5f} -> Hier={roc_auc_score(y_train[seen_masks], seller_te_hier_full[seen_masks]):.5f}")

if __name__ == "__main__":
    main()
