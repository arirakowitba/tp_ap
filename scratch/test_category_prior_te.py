import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

def main():
    print("=== Testing Hierarchical Category-Conditioned Seller Prior ===")
    with open("data/ap_2026_q_2/raw/tr.pkl", "rb") as f:
        df_tr = pickle.load(f)

    y_train = (df_tr["condition"] == "used").astype(int).values
    df_tr["target"] = y_train
    n_train = len(df_tr)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    seller_te_global = np.zeros(n_train)
    seller_te_hier = np.zeros(n_train)

    global_prior = float(y_train.mean())
    m_seller = 5.0
    m_cat = 10.0

    for tr_idx, val_idx in cv.split(df_tr, y_train):
        fold_tr = df_tr.iloc[tr_idx]
        fold_val = df_tr.iloc[val_idx]

        # 1. Category stats in fold
        cat_stats = fold_tr.groupby("category_id")["target"].agg(["count", "sum"])
        cat_smoothed = (cat_stats["sum"] + m_cat * global_prior) / (cat_stats["count"] + m_cat)

        # 2. Global seller stats
        seller_stats = fold_tr.groupby("seller_id")["target"].agg(["count", "sum"])
        s_smoothed_global = (seller_stats["sum"] + m_seller * global_prior) / (seller_stats["count"] + m_seller)
        seller_te_global[val_idx] = fold_val["seller_id"].map(s_smoothed_global).fillna(global_prior).values

        # 3. Hierarchical Category-Conditioned seller stats
        # For each seller in fold_tr, what is their primary category? Or category-specific seller TE:
        # If unseen seller in val, fillna with cat_smoothed of their category!
        val_cat_priors = fold_val["category_id"].map(cat_smoothed).fillna(global_prior)
        
        # Merge seller stats with category prior
        seller_te_hier[val_idx] = fold_val["seller_id"].map(s_smoothed_global).fillna(val_cat_priors).values

    print(f"Global Seller TE OOF LogLoss:       {log_loss(y_train, seller_te_global):.5f}")
    print(f"Hierarchical Seller TE OOF LogLoss: {log_loss(y_train, seller_te_hier):.5f}")
    
    # Evaluate specifically on NEW / UNSEEN sellers in each fold
    # Unseen sellers are those whose seller_id was not in fold_tr
    unseen_masks = np.zeros(n_train, dtype=bool)
    for tr_idx, val_idx in cv.split(df_tr, y_train):
        train_sellers = set(df_tr.iloc[tr_idx]["seller_id"])
        val_sellers = df_tr.iloc[val_idx]["seller_id"]
        unseen_masks[val_idx] = ~val_sellers.isin(train_sellers).values

    n_unseen = unseen_masks.sum()
    print(f"\nUnseen sellers across folds: {n_unseen} items ({n_unseen/n_train*100:.1f}%)")
    ll_unseen_glob = log_loss(y_train[unseen_masks], seller_te_global[unseen_masks])
    ll_unseen_hier = log_loss(y_train[unseen_masks], seller_te_hier[unseen_masks])
    auc_unseen_glob = roc_auc_score(y_train[unseen_masks], seller_te_global[unseen_masks])
    auc_unseen_hier = roc_auc_score(y_train[unseen_masks], seller_te_hier[unseen_masks])

    print(f"Unseen Sellers LogLoss: Global={ll_unseen_glob:.5f} -> Hierarchical={ll_unseen_hier:.5f} (Delta: {ll_unseen_hier - ll_unseen_glob:.5f})")
    print(f"Unseen Sellers ROC-AUC: Global={auc_unseen_glob:.5f} -> Hierarchical={auc_unseen_hier:.5f} (Delta: {auc_unseen_hier - auc_unseen_glob:+.5f})")

if __name__ == "__main__":
    main()
