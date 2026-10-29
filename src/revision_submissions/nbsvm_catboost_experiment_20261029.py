"""Honest Group-OOF NBSVM + Sparse Dual Text Signal into CatBoost.

Tests whether combining clean, leak-free GroupKFold OOF predictions from
both NB-SVM (log-count ratio scaled binary n-grams on title+warranty)
and SparseLinear (TF-IDF + structured metadata) provides a superior,
statistically significant boost to CatBoost on external seller-group confirm.
"""
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, log_loss
from catboost import CatBoostClassifier, Pool

ROOT = Path(__file__).resolve().parents[2]
TRAIN_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "train_data.jsonlines"
RESULTS_DIR = ROOT / "results" / "revision_submissions"


def load_train_data():
    t0 = time.time()
    rows = []
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    y_used = np.array([1 if r["condition"] == "used" else 0 for r in rows], dtype=int)
    ids = np.array([str(r["id"]) for r in rows])
    sellers = np.array([str(r["seller_id"]) for r in rows])
    titles = [r.get("title") or "" for r in rows]
    warranties = [r.get("warranty") or "" for r in rows]
    return rows, ids, sellers, y_used, titles, warranties


def split_seller_groups(seller, seed):
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


def cluster_bootstrap_ci(y_true, base_pred, test_pred, clusters, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(clusters)
    n_clust = len(unique_clusters)
    clust_to_idx = defaultdict(list)
    for i, c in enumerate(clusters):
        clust_to_idx[c].append(i)
    base_correct = (base_pred == y_true)
    test_correct = (test_pred == y_true)
    diff = test_correct.astype(int) - base_correct.astype(int)
    clust_sums = np.array([np.sum(diff[clust_to_idx[c]]) for c in unique_clusters])
    boot_nets = np.zeros(n_boot)
    for b in range(n_boot):
        sample_idx = rng.integers(0, n_clust, size=n_clust)
        boot_nets[b] = np.sum(clust_sums[sample_idx])
    return float(np.percentile(boot_nets, 2.5)), float(np.percentile(boot_nets, 97.5)), float(np.mean(boot_nets))


def extract_metadata(rows, indices, cat_medians=None):
    data = []
    for idx in indices:
        r = rows[idx]
        title = r.get("title") or ""
        warranty = r.get("warranty") or ""
        sh = r.get("shipping") or {}
        adr = r.get("seller_address") or {}
        state = (adr.get("state") or {}).get("id") or ""
        pics = r.get("pictures") or []
        attrs = r.get("attributes") or []
        variations = r.get("variations") or []
        tags = r.get("tags") or []
        
        price = float(r.get("price") or 0)
        log_price = np.log1p(price)
        cat_id = str(r.get("category_id") or "")
        cat_med = cat_medians.get(cat_id, cat_medians.get("__overall__", log_price)) if cat_medians else log_price
        rel_price_cat = log_price - cat_med
        
        init_q = float(r.get("initial_quantity") or 0)
        avail_q = float(r.get("available_quantity") or 0)
        sold_q = float(r.get("sold_quantity") or 0)
        
        feat = {
            "log_price": log_price,
            "rel_price_cat": rel_price_cat,
            "log_init_q": np.log1p(init_q),
            "log_avail_q": np.log1p(avail_q),
            "log_sold_q": np.log1p(sold_q),
            "sold_ratio": sold_q / (init_q + 1.0),
            "init_is_one": int(init_q == 1),
            "sold_is_zero": int(sold_q == 0),
            "title_len": len(title),
            "title_words": len(title.split()),
            "title_upper_frac": sum(ch.isupper() for ch in title) / max(len(title), 1),
            "title_digits": sum(ch.isdigit() for ch in title),
            "n_pics": len(pics),
            "n_attrs": len(attrs),
            "n_variations": len(variations),
            "has_warranty": int(bool(warranty)),
            "warranty_len": len(warranty),
            "free_shipping": int(bool(sh.get("free_shipping"))),
            "local_pickup": int(bool(sh.get("local_pick_up"))),
            "accepts_mp": int(bool(r.get("accepts_mercadopago"))),
            "has_video": int(bool(r.get("video_id"))),
            "auto_relist": int(bool(r.get("automatic_relist"))),
            "tag_dragged_bids": int("dragged_bids_and_visits" in tags),
            "category_id": cat_id,
            "listing_type_id": str(r.get("listing_type_id") or ""),
            "buying_mode": str(r.get("buying_mode") or ""),
            "shipping_mode": str(sh.get("mode") or ""),
            "state": state,
        }
        data.append(feat)
    return data


def compute_category_medians(rows, fit_indices):
    cat_prices = defaultdict(list)
    all_prices = []
    for idx in fit_indices:
        r = rows[idx]
        price = float(r.get("price") or 0)
        lp = np.log1p(price)
        cat_id = str(r.get("category_id") or "")
        cat_prices[cat_id].append(lp)
        all_prices.append(lp)
    overall_med = float(np.median(all_prices)) if all_prices else 0.0
    medians = {"__overall__": overall_med}
    for cat_id, lps in cat_prices.items():
        medians[cat_id] = float(np.median(lps))
    return medians


def main():
    rows, ids, sellers, y_used, titles, warranties = load_train_data()
    seeds = [2026, 314159]
    all_results = []
    
    for seed in seeds:
        print(f"\n=======================================================", flush=True)
        print(f"RUNNING NBSVM + SPARSE DUAL TEXT ON SEED {seed}", flush=True)
        print(f"=======================================================", flush=True)
        
        fit, tune, confirm = split_seller_groups(sellers, seed)
        cat_medians = compute_category_medians(rows, fit)
        
        meta_fit = extract_metadata(rows, fit, cat_medians)
        meta_tune = extract_metadata(rows, tune, cat_medians)
        meta_confirm = extract_metadata(rows, confirm, cat_medians)
        
        # 1. Sparse TF-IDF Model (Model A text)
        print("Fitting Sparse TF-IDF on fit...", flush=True)
        vec_w = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=75000,
                                sublinear_tf=True, strip_accents="unicode")
        vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                                max_features=120000, sublinear_tf=True)
        vec_wr = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=25000,
                                 sublinear_tf=True, strip_accents="unicode")
        
        w_fit = vec_w.fit_transform([titles[i] for i in fit])
        c_fit = vec_c.fit_transform([titles[i] for i in fit])
        wr_fit = vec_wr.fit_transform([warranties[i] for i in fit])
        
        dict_vec = DictVectorizer(sparse=True)
        m_fit = dict_vec.fit_transform(meta_fit)
        X_sp_fit = sp.hstack([w_fit, c_fit, wr_fit, m_fit], format="csr")
        
        w_tune = vec_w.transform([titles[i] for i in tune])
        c_tune = vec_c.transform([titles[i] for i in tune])
        wr_tune = vec_wr.transform([warranties[i] for i in tune])
        m_tune = dict_vec.transform(meta_tune)
        X_sp_tune = sp.hstack([w_tune, c_tune, wr_tune, m_tune], format="csr")
        
        w_confirm = vec_w.transform([titles[i] for i in confirm])
        c_confirm = vec_c.transform([titles[i] for i in confirm])
        wr_confirm = vec_wr.transform([warranties[i] for i in confirm])
        m_confirm = dict_vec.transform(meta_confirm)
        X_sp_confirm = sp.hstack([w_confirm, c_confirm, wr_confirm, m_confirm], format="csr")
        
        # 2. NB-SVM Vectorizer (Model B text: log-count ratio scaled binary ngrams)
        print("Fitting NB-SVM text representations on fit...", flush=True)
        txt_combined_fit = [f"{titles[i]} __w__ {warranties[i]}" for i in fit]
        txt_combined_tune = [f"{titles[i]} __w__ {warranties[i]}" for i in tune]
        txt_combined_confirm = [f"{titles[i]} __w__ {warranties[i]}" for i in confirm]
        
        cv_w = CountVectorizer(ngram_range=(1, 3), min_df=3, binary=True, strip_accents="unicode", max_features=100000)
        cv_c = CountVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, binary=True, strip_accents="unicode", max_features=150000)
        
        X_nb_w_fit = cv_w.fit_transform(txt_combined_fit)
        X_nb_c_fit = cv_c.fit_transform([titles[i] for i in fit])
        X_nb_fit = sp.hstack([X_nb_w_fit, X_nb_c_fit]).tocsr().astype(np.float32)
        
        X_nb_w_tune = cv_w.transform(txt_combined_tune)
        X_nb_c_tune = cv_c.transform([titles[i] for i in tune])
        X_nb_tune = sp.hstack([X_nb_w_tune, X_nb_c_tune]).tocsr().astype(np.float32)
        
        X_nb_w_confirm = cv_w.transform(txt_combined_confirm)
        X_nb_c_confirm = cv_c.transform([titles[i] for i in confirm])
        X_nb_confirm = sp.hstack([X_nb_w_confirm, X_nb_c_confirm]).tocsr().astype(np.float32)
        
        # 3. Honest 4-Fold GroupKFold inside fit for BOTH text models
        print("Computing clean GroupKFold OOF for SparseLinear and NB-SVM on fit...", flush=True)
        oof_p_sparse = np.zeros(len(fit), dtype=np.float32)
        oof_p_nbsvm = np.zeros(len(fit), dtype=np.float32)
        
        gkf = GroupKFold(n_splits=4)
        for fold, (tr, va) in enumerate(gkf.split(X_sp_fit, y_used[fit], sellers[fit])):
            # Sparse LR
            lr_sp = LogisticRegression(C=2.5, max_iter=200, solver="liblinear", random_state=seed + fold)
            lr_sp.fit(X_sp_fit[tr], y_used[fit][tr])
            oof_p_sparse[va] = lr_sp.predict_proba(X_sp_fit[va])[:, 1]
            
            # NB-SVM
            p_cnt = 1.0 + X_nb_fit[tr][y_used[fit][tr] == 1].sum(0)
            q_cnt = 1.0 + X_nb_fit[tr][y_used[fit][tr] == 0].sum(0)
            r_fold = np.asarray(np.log((p_cnt / p_cnt.sum()) / (q_cnt / q_cnt.sum()))).ravel()
            R_fold = sp.diags(r_fold)
            
            lr_nb = LogisticRegression(C=0.5, max_iter=200, solver="liblinear", random_state=seed + fold)
            lr_nb.fit(X_nb_fit[tr] @ R_fold, y_used[fit][tr])
            oof_p_nbsvm[va] = lr_nb.predict_proba(X_nb_fit[va] @ R_fold)[:, 1]
            
        print(f"Fit OOF Sparse Acc: {accuracy_score(y_used[fit], oof_p_sparse > 0.5):.4f}", flush=True)
        print(f"Fit OOF NBSVM Acc:  {accuracy_score(y_used[fit], oof_p_nbsvm > 0.5):.4f}", flush=True)
        
        # Fit full text models on entire fit
        lr_sp_full = LogisticRegression(C=2.5, max_iter=250, solver="liblinear", random_state=seed).fit(X_sp_fit, y_used[fit])
        p_sp_tune = lr_sp_full.predict_proba(X_sp_tune)[:, 1]
        p_sp_confirm = lr_sp_full.predict_proba(X_sp_confirm)[:, 1]
        
        p_cnt_full = 1.0 + X_nb_fit[y_used[fit] == 1].sum(0)
        q_cnt_full = 1.0 + X_nb_fit[y_used[fit] == 0].sum(0)
        r_full = np.asarray(np.log((p_cnt_full / p_cnt_full.sum()) / (q_cnt_full / q_cnt_full.sum()))).ravel()
        R_full = sp.diags(r_full)
        
        lr_nb_full = LogisticRegression(C=0.5, max_iter=250, solver="liblinear", random_state=seed).fit(X_nb_fit @ R_full, y_used[fit])
        p_nb_tune = lr_nb_full.predict_proba(X_nb_tune @ R_full)[:, 1]
        p_nb_confirm = lr_nb_full.predict_proba(X_nb_confirm @ R_full)[:, 1]
        
        # 4. Prepare CatBoost feature matrices
        CAT_COLS = ["category_id", "listing_type_id", "buying_mode", "shipping_mode", "state"]
        NUM_COLS = [k for k in meta_fit[0].keys() if k not in CAT_COLS]
        
        # Baseline CatBoost: only sparse_p
        df_base_fit = pd.DataFrame([{col: m[col] for col in NUM_COLS + CAT_COLS} for m in meta_fit])
        df_base_fit["sparse_p"] = oof_p_sparse
        for c in CAT_COLS: df_base_fit[c] = df_base_fit[c].astype(str)
        
        df_base_tune = pd.DataFrame([{col: m[col] for col in NUM_COLS + CAT_COLS} for m in meta_tune])
        df_base_tune["sparse_p"] = p_sp_tune
        for c in CAT_COLS: df_base_tune[c] = df_base_tune[c].astype(str)
        
        df_base_confirm = pd.DataFrame([{col: m[col] for col in NUM_COLS + CAT_COLS} for m in meta_confirm])
        df_base_confirm["sparse_p"] = p_sp_confirm
        for c in CAT_COLS: df_base_confirm[c] = df_base_confirm[c].astype(str)
        
        # Dual CatBoost: sparse_p AND nbsvm_p
        df_dual_fit = df_base_fit.copy()
        df_dual_fit["nbsvm_p"] = oof_p_nbsvm
        
        df_dual_tune = df_base_tune.copy()
        df_dual_tune["nbsvm_p"] = p_nb_tune
        
        df_dual_confirm = df_base_confirm.copy()
        df_dual_confirm["nbsvm_p"] = p_nb_confirm
        
        # 5. Train Baseline CatBoost
        print("Training Baseline CatBoost (sparse_p only)...", flush=True)
        cb_base = CatBoostClassifier(iterations=800, depth=5, learning_rate=0.08, l2_leaf_reg=5.0,
                                     random_seed=seed, thread_count=12, verbose=0)
        t0 = time.time()
        cb_base.fit(Pool(df_base_fit, y_used[fit], cat_features=CAT_COLS))
        print(f"Baseline CatBoost fitted in {time.time()-t0:.2f}s", flush=True)
        p_base_confirm = cb_base.predict_proba(Pool(df_base_confirm, cat_features=CAT_COLS))[:, 1]
        
        # 6. Train Dual CatBoost
        print("Training Dual-Text CatBoost (sparse_p + nbsvm_p)...", flush=True)
        cb_dual = CatBoostClassifier(iterations=800, depth=5, learning_rate=0.08, l2_leaf_reg=5.0,
                                     random_seed=seed, thread_count=12, verbose=0)
        t0 = time.time()
        cb_dual.fit(Pool(df_dual_fit, y_used[fit], cat_features=CAT_COLS))
        print(f"Dual CatBoost fitted in {time.time()-t0:.2f}s", flush=True)
        p_dual_confirm = cb_dual.predict_proba(Pool(df_dual_confirm, cat_features=CAT_COLS))[:, 1]
        
        # 7. Evaluate Baseline vs Dual CatBoost on Confirm
        y_c = y_used[confirm]
        pred_base_c = (p_base_confirm >= 0.5).astype(int)
        pred_dual_c = (p_dual_confirm >= 0.5).astype(int)
        
        acc_base = accuracy_score(y_c, pred_base_c)
        acc_dual = accuracy_score(y_c, pred_dual_c)
        ll_base = log_loss(y_c, p_base_confirm)
        ll_dual = log_loss(y_c, p_dual_confirm)
        
        fixes = int(np.sum((pred_base_c != y_c) & (pred_dual_c == y_c)))
        breaks = int(np.sum((pred_base_c == y_c) & (pred_dual_c != y_c)))
        net = fixes - breaks
        
        ci_low, ci_high, _ = cluster_bootstrap_ci(y_c, pred_base_c, pred_dual_c, sellers[confirm])
        
        pairs_prev = pd.read_csv(RESULTS_DIR / f"paired_predictions_seed_{seed}.csv")
        seen_mask = pairs_prev["is_seen_seller"].values.astype(bool)
        new_mask = ~seen_mask
        
        acc_base_seen = accuracy_score(y_c[seen_mask], pred_base_c[seen_mask])
        acc_dual_seen = accuracy_score(y_c[seen_mask], pred_dual_c[seen_mask])
        acc_base_new = accuracy_score(y_c[new_mask], pred_base_c[new_mask])
        acc_dual_new = accuracy_score(y_c[new_mask], pred_dual_c[new_mask])
        
        print(f"\nCONFIRM RESULTS (Seed {seed}):", flush=True)
        print(f"  Baseline CatBoost Acc: {acc_base:.5f}, LL: {ll_base:.5f}", flush=True)
        print(f"  Dual CatBoost Acc:     {acc_dual:.5f}, LL: {ll_dual:.5f}", flush=True)
        print(f"  Fixes/Breaks:          {fixes}/{breaks} -> NET: {net:+d}", flush=True)
        print(f"  Bootstrap 95% CI:      [{ci_low:+.1f}, {ci_high:+.1f}]", flush=True)
        print(f"  Seen sellers Acc:      {acc_base_seen:.5f} -> {acc_dual_seen:.5f} (n={seen_mask.sum()})", flush=True)
        print(f"  New sellers Acc:       {acc_base_new:.5f} -> {acc_dual_new:.5f} (n={new_mask.sum()})", flush=True)
        
        out_pairs = RESULTS_DIR / f"paired_dual_text_seed_{seed}.csv"
        with open(out_pairs, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "seller_id", "y_true", "is_seen", "p_base", "p_dual", "pred_base", "pred_dual"])
            for idx, i in enumerate(confirm):
                w.writerow([
                    ids[i], sellers[i], y_used[i], int(seen_mask[idx]),
                    round(float(p_base_confirm[idx]), 5),
                    round(float(p_dual_confirm[idx]), 5),
                    int(pred_base_c[idx]),
                    int(pred_dual_c[idx])
                ])
                
        all_results.append({
            "seed": seed,
            "confirm_n": len(confirm),
            "acc_base": float(acc_base),
            "acc_dual": float(acc_dual),
            "ll_base": float(ll_base),
            "ll_dual": float(ll_dual),
            "fixes": fixes,
            "breaks": breaks,
            "net": net,
            "ci_95": [ci_low, ci_high],
            "seen_acc_base": float(acc_base_seen),
            "seen_acc_dual": float(acc_dual_seen),
            "new_acc_base": float(acc_base_new),
            "new_acc_dual": float(acc_dual_new),
        })
        
    summary_file = RESULTS_DIR / "round144_dual_text_results.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {summary_file}", flush=True)


if __name__ == "__main__":
    main()
