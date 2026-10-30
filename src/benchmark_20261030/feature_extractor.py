"""Feature Extraction for R151 (Pipeline A) and R157 (Pipeline B).
Delegates to the canonical extract_deep_features from R151 to guarantee exact reproduction.
"""
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
import scipy.sparse as sp

from src.revision_submissions.deep_catboost_nbsvm_ablation_20261029 import extract_deep_features

def extract_r151_features(rows):
    """Extracts exact deep feature set from R151 (deep_catboost_nbsvm_ablation_20261029.py)."""
    F_df, cat_df, titles, warranties, combos = extract_deep_features(rows)
    return F_df, cat_df

def compute_leak_free_te_r151(cat_df, fit, eval_indices_list, y_fit, sellers_fit, alpha=5.0):
    """Computes exact R151 target encoding with alpha=5."""
    prior = float(y_fit.mean())
    te_cols = ["seller_id", "category_id", "city"]
    gkf = GroupKFold(n_splits=4)
    
    fit_te = {c: np.full(len(fit), prior, dtype=np.float32) for c in te_cols}
    eval_tes = [{c: None for c in te_cols} for _ in eval_indices_list]
    
    for tr, va in gkf.split(fit, y_fit, sellers_fit):
        for c in te_cols:
            k_tr = cat_df[c].iloc[fit[tr]]
            k_va = cat_df[c].iloc[fit[va]]
            stats = pd.DataFrame({"k": k_tr, "y": y_fit[tr]}).groupby("k").y.agg(["sum", "count"])
            smoothed = (stats["sum"] + alpha * prior) / (stats["count"] + alpha)
            fit_te[c][va] = k_va.map(smoothed).fillna(prior).values.astype(np.float32)
            
    for c in te_cols:
        k_fit = cat_df[c].iloc[fit]
        stats = pd.DataFrame({"k": k_fit, "y": y_fit}).groupby("k").y.agg(["sum", "count"])
        smoothed = (stats["sum"] + alpha * prior) / (stats["count"] + alpha)
        for idx, eval_indices in enumerate(eval_indices_list):
            eval_tes[idx][c] = cat_df[c].iloc[eval_indices].map(smoothed).fillna(prior).values.astype(np.float32)
            
    return fit_te, eval_tes

def compute_txt_sparse_p(combos, titles, fit, eval_indices_list, y_fit, sellers_fit, seed=2026):
    """Computes leak-free txt_sparse_p via 4-fold GroupKFold on fit."""
    vec_combo = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=120000,
                                sublinear_tf=True, strip_accents="unicode", lowercase=True)
    vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                               max_features=150000, sublinear_tf=True)
    
    X_w_fit = vec_combo.fit_transform([combos[i] for i in fit])
    X_c_fit = vec_char.fit_transform([titles[i] for i in fit])
    X_sp_fit = sp.hstack([X_w_fit, X_c_fit], format="csr")
    
    oof_sp = np.zeros(len(fit), dtype=np.float32)
    gkf = GroupKFold(n_splits=4)
    for tr, va in gkf.split(fit, y_fit, sellers_fit):
        lr_sp = LogisticRegression(C=3.0, max_iter=250, solver="liblinear", random_state=seed)
        lr_sp.fit(X_sp_fit[tr], y_fit[tr])
        oof_sp[va] = lr_sp.predict_proba(X_sp_fit[va])[:, 1]
        
    lr_full = LogisticRegression(C=3.0, max_iter=300, solver="liblinear", random_state=seed).fit(X_sp_fit, y_fit)
    eval_probs = []
    for eval_indices in eval_indices_list:
        X_w_ev = vec_combo.transform([combos[i] for i in eval_indices])
        X_c_ev = vec_char.transform([titles[i] for i in eval_indices])
        X_sp_ev = sp.hstack([X_w_ev, X_c_ev], format="csr")
        eval_probs.append(lr_full.predict_proba(X_sp_ev)[:, 1].astype(np.float32))
        
    return oof_sp, eval_probs
