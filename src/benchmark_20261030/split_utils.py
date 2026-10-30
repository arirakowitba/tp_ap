"""Split Utilities and Partition Integrity Verification.
"""
import numpy as np

def split_seller_groups_r151(seller, seed):
    """Canonical development split from round151/153 (deep_catboost_nbsvm_ablation_20261029.py)."""
    rng = np.random.default_rng(seed)
    unique = np.unique(seller)
    rng.shuffle(unique)
    n_new = len(unique) // 10
    new_tune = set(unique[:n_new])
    new_confirm = set(unique[n_new: 2 * n_new])
    seen_sellers = ~(np.isin(seller, list(new_tune | new_confirm)))
    seen_row_indices = np.where(seen_sellers)[0]
    rng.shuffle(seen_row_indices)
    n_seen_tune = len(seen_row_indices) // 10
    seen_tune = seen_row_indices[:n_seen_tune]
    seen_confirm = seen_row_indices[n_seen_tune: 2 * n_seen_tune]
    tune_rows = np.flatnonzero(np.isin(seller, list(new_tune)))
    confirm_rows = np.flatnonzero(np.isin(seller, list(new_confirm)))
    tune = np.r_[tune_rows, seen_tune]
    confirm = np.r_[confirm_rows, seen_confirm]
    fit = np.setdiff1d(np.arange(len(seller)), np.r_[tune, confirm])
    return fit, tune, confirm

def split_seller_groups_r157(seller, seed):
    """Canonical split from R157 (strong_ensemble_audit_and_refit_20261029.py)."""
    rng = np.random.default_rng(seed)
    unique = np.unique(seller)
    rng.shuffle(unique)
    new_tune = set(unique[:len(unique)//10])
    new_confirm = set(unique[len(unique)//10:len(unique)//5])
    seen = ~(np.isin(seller, list(new_tune | new_confirm)))
    ids = np.where(seen)[0]
    rng.shuffle(ids)
    seen_tune, seen_confirm = ids[:len(ids)//10], ids[len(ids)//10:len(ids)//5]
    tune = np.flatnonzero(np.isin(seller, list(new_tune)))
    confirm = np.flatnonzero(np.isin(seller, list(new_confirm)))
    tune = np.r_[tune, seen_tune]
    confirm = np.r_[confirm, seen_confirm]
    fit = np.setdiff1d(np.arange(len(seller)), np.r_[tune, confirm])
    return fit, tune, confirm
