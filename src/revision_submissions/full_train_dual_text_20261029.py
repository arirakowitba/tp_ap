"""Full 70k Training of Dual-Text CatBoost (Sparse TF-IDF + NB-SVM) and Comparison to Submission 48.

Trains the confirmed Dual-Text CatBoost architecture on the full 70k labeled train dataset
using 5-fold GroupKFold by seller for internal OOF text features.
Predicts probabilities and labels on the 30k test dataset (IDs 1 to 30000).
Evaluates exact disagreements (flips) against Submission 48, broken down by:
- Seen vs New sellers in train
- Direction (used -> new vs new -> used)
- Margin / confidence
Generates candidate submission CSVs, computes SHA256 hashes, and prepares comparison artifacts.
"""
import csv
import hashlib
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
TEST_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines"
S48_JSON = ROOT / "results" / "revision_submissions" / "posterior_flips_20261027" / "s48.json"
RESULTS_DIR = ROOT / "results" / "revision_submissions"
SUBMISSIONS_DIR = ROOT / "submissions" / "20261029"


def load_dataset(path, is_test=False):
    t0 = time.time()
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    ids = np.array([str(r["id"]) for r in rows])
    sellers = np.array([str(r["seller_id"]) for r in rows])
    titles = [r.get("title") or "" for r in rows]
    warranties = [r.get("warranty") or "" for r in rows]
    if not is_test:
        y_used = np.array([1 if r["condition"] == "used" else 0 for r in rows], dtype=int)
        print(f"Loaded {len(rows)} train rows in {time.time()-t0:.2f}s", flush=True)
        return rows, ids, sellers, y_used, titles, warranties
    else:
        print(f"Loaded {len(rows)} test rows in {time.time()-t0:.2f}s", flush=True)
        return rows, ids, sellers, titles, warranties


def compute_category_medians(rows):
    cat_prices = defaultdict(list)
    all_prices = []
    for r in rows:
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


def extract_metadata(rows, cat_medians):
    data = []
    for r in rows:
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
        cat_med = cat_medians.get(cat_id, cat_medians.get("__overall__", log_price))
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


def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    train_rows, train_ids, train_sellers, y_train, train_titles, train_warranties = load_dataset(TRAIN_JSON, is_test=False)
    test_rows, test_ids, test_sellers, test_titles, test_warranties = load_dataset(TEST_JSON, is_test=True)
    
    with open(S48_JSON, "r", encoding="utf-8") as f:
        s48_dict = json.load(f)
    s48_labels = np.array([s48_dict[str(i)] for i in range(1, 30001)])
    s48_y_used = (s48_labels == "used").astype(int)
    
    train_seller_set = set(train_sellers)
    test_is_seen = np.array([s in train_seller_set for s in test_sellers], dtype=bool)
    print(f"Test seen sellers count: {test_is_seen.sum()} ({test_is_seen.sum()/len(test_sellers)*100:.2f}%)", flush=True)
    
    # 2. Extract metadata
    print("Extracting metadata...", flush=True)
    cat_medians = compute_category_medians(train_rows)
    meta_train = extract_metadata(train_rows, cat_medians)
    meta_test = extract_metadata(test_rows, cat_medians)
    
    # 3. Vectorize Sparse TF-IDF text + metadata
    print("Fitting Sparse TF-IDF Vectorizers on 70k train...", flush=True)
    t0 = time.time()
    vec_w = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=75000,
                            sublinear_tf=True, strip_accents="unicode")
    vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                            max_features=120000, sublinear_tf=True)
    vec_wr = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=25000,
                             sublinear_tf=True, strip_accents="unicode")
    
    w_tr = vec_w.fit_transform(train_titles)
    c_tr = vec_c.fit_transform(train_titles)
    wr_tr = vec_wr.fit_transform(train_warranties)
    
    w_te = vec_w.transform(test_titles)
    c_te = vec_c.transform(test_titles)
    wr_te = vec_wr.transform(test_warranties)
    
    dict_vec = DictVectorizer(sparse=True)
    m_tr = dict_vec.fit_transform(meta_train)
    m_te = dict_vec.transform(meta_test)
    
    X_sp_train = sp.hstack([w_tr, c_tr, wr_tr, m_tr], format="csr")
    X_sp_test = sp.hstack([w_te, c_te, wr_te, m_te], format="csr")
    print(f"Sparse representation constructed in {time.time()-t0:.2f}s: shape {X_sp_train.shape}", flush=True)
    
    # 4. Vectorize NB-SVM text
    print("Fitting NB-SVM CountVectorizers on 70k train...", flush=True)
    t0 = time.time()
    txt_comb_train = [f"{train_titles[i]} __w__ {train_warranties[i]}" for i in range(len(train_titles))]
    txt_comb_test = [f"{test_titles[i]} __w__ {test_warranties[i]}" for i in range(len(test_titles))]
    
    cv_w = CountVectorizer(ngram_range=(1, 3), min_df=3, binary=True, strip_accents="unicode", max_features=100000)
    cv_c = CountVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, binary=True, strip_accents="unicode", max_features=150000)
    
    X_nb_w_tr = cv_w.fit_transform(txt_comb_train)
    X_nb_c_tr = cv_c.fit_transform(train_titles)
    X_nb_train = sp.hstack([X_nb_w_tr, X_nb_c_tr]).tocsr().astype(np.float32)
    
    X_nb_w_te = cv_w.transform(txt_comb_test)
    X_nb_c_te = cv_c.transform(test_titles)
    X_nb_test = sp.hstack([X_nb_w_te, X_nb_c_te]).tocsr().astype(np.float32)
    print(f"NB-SVM representation constructed in {time.time()-t0:.2f}s: shape {X_nb_train.shape}", flush=True)
    
    # 5. Clean 5-fold GroupKFold by seller for train OOF features
    print("Computing 5-fold GroupKFold OOF on 70k train for Sparse and NB-SVM...", flush=True)
    t0 = time.time()
    train_oof_sparse = np.zeros(len(train_rows), dtype=np.float32)
    train_oof_nbsvm = np.zeros(len(train_rows), dtype=np.float32)
    
    gkf = GroupKFold(n_splits=5)
    for fold, (tr, va) in enumerate(gkf.split(X_sp_train, y_train, train_sellers)):
        print(f"  Fold {fold+1}/5...", flush=True)
        # Sparse LR
        lr_sp = LogisticRegression(C=2.5, max_iter=200, solver="liblinear", random_state=2026 + fold)
        lr_sp.fit(X_sp_train[tr], y_train[tr])
        train_oof_sparse[va] = lr_sp.predict_proba(X_sp_train[va])[:, 1]
        
        # NB-SVM
        p_cnt = 1.0 + X_nb_train[tr][y_train[tr] == 1].sum(0)
        q_cnt = 1.0 + X_nb_train[tr][y_train[tr] == 0].sum(0)
        r_fold = np.asarray(np.log((p_cnt / p_cnt.sum()) / (q_cnt / q_cnt.sum()))).ravel()
        R_fold = sp.diags(r_fold)
        
        lr_nb = LogisticRegression(C=0.5, max_iter=200, solver="liblinear", random_state=2026 + fold)
        lr_nb.fit(X_nb_train[tr] @ R_fold, y_train[tr])
        train_oof_nbsvm[va] = lr_nb.predict_proba(X_nb_train[va] @ R_fold)[:, 1]
        
    print(f"OOF generation completed in {time.time()-t0:.2f}s", flush=True)
    print(f"  Train OOF Sparse Acc: {accuracy_score(y_train, train_oof_sparse > 0.5):.5f}, LL: {log_loss(y_train, train_oof_sparse):.5f}", flush=True)
    print(f"  Train OOF NBSVM Acc:  {accuracy_score(y_train, train_oof_nbsvm > 0.5):.5f}, LL: {log_loss(y_train, train_oof_nbsvm):.5f}", flush=True)
    
    # 6. Fit full text models on entire 70k train to predict test
    print("Fitting text models on full 70k to predict test set...", flush=True)
    t0 = time.time()
    lr_sp_full = LogisticRegression(C=2.5, max_iter=250, solver="liblinear", random_state=2026).fit(X_sp_train, y_train)
    test_sparse_p = lr_sp_full.predict_proba(X_sp_test)[:, 1]
    
    p_cnt_full = 1.0 + X_nb_train[y_train == 1].sum(0)
    q_cnt_full = 1.0 + X_nb_train[y_train == 0].sum(0)
    r_full = np.asarray(np.log((p_cnt_full / p_cnt_full.sum()) / (q_cnt_full / q_cnt_full.sum()))).ravel()
    R_full = sp.diags(r_full)
    
    lr_nb_full = LogisticRegression(C=0.5, max_iter=250, solver="liblinear", random_state=2026).fit(X_nb_train @ R_full, y_train)
    test_nbsvm_p = lr_nb_full.predict_proba(X_nb_test @ R_full)[:, 1]
    print(f"Full text models fitted and test predicted in {time.time()-t0:.2f}s", flush=True)
    
    # 7. Prepare DataFrames for CatBoost
    CAT_COLS = ["category_id", "listing_type_id", "buying_mode", "shipping_mode", "state"]
    NUM_COLS = [k for k in meta_train[0].keys() if k not in CAT_COLS]
    
    df_train = pd.DataFrame([{col: m[col] for col in NUM_COLS + CAT_COLS} for m in meta_train])
    df_train["sparse_p"] = train_oof_sparse
    df_train["nbsvm_p"] = train_oof_nbsvm
    for c in CAT_COLS: df_train[c] = df_train[c].astype(str)
    
    df_test = pd.DataFrame([{col: m[col] for col in NUM_COLS + CAT_COLS} for m in meta_test])
    df_test["sparse_p"] = test_sparse_p
    df_test["nbsvm_p"] = test_nbsvm_p
    for c in CAT_COLS: df_test[c] = df_test[c].astype(str)
    
    # 8. Train CatBoost on full 70k
    print("Training Dual-Text CatBoost on full 70k train...", flush=True)
    t0 = time.time()
    cb_full = CatBoostClassifier(iterations=800, depth=5, learning_rate=0.08, l2_leaf_reg=5.0,
                                 random_seed=2026, thread_count=12, verbose=100)
    cb_full.fit(Pool(df_train, y_train, cat_features=CAT_COLS))
    print(f"CatBoost full train completed in {time.time()-t0:.2f}s", flush=True)
    
    # 9. Predict test probabilities
    test_catboost_p = cb_full.predict_proba(Pool(df_test, cat_features=CAT_COLS))[:, 1]
    pred_catboost_y = (test_catboost_p >= 0.50).astype(int)
    
    # Save raw test probabilities
    np.save(RESULTS_DIR / "dual_catboost_test_probs.npy", test_catboost_p)
    print(f"Saved test probabilities to {RESULTS_DIR / 'dual_catboost_test_probs.npy'}", flush=True)
    
    # 10. Compare against Submission 48
    print("\n=======================================================", flush=True)
    print("ANALYSIS OF DISAGREEMENTS AGAINST SUBMISSION 48", flush=True)
    print("=======================================================", flush=True)
    
    flips_mask = (pred_catboost_y != s48_y_used)
    n_flips = int(flips_mask.sum())
    print(f"Total flips vs 48: {n_flips} / 30000 ({n_flips/300:.2f}%)", flush=True)
    
    # Breakdown by Seen vs New
    flips_seen = flips_mask & test_is_seen
    flips_new = flips_mask & (~test_is_seen)
    print(f"Flips on Seen Sellers: {flips_seen.sum()} / {test_is_seen.sum()} ({flips_seen.sum()/test_is_seen.sum()*100:.2f}%)", flush=True)
    print(f"Flips on New Sellers:  {flips_new.sum()} / {(~test_is_seen).sum()} ({flips_new.sum()/(~test_is_seen).sum()*100:.2f}%)", flush=True)
    
    # Breakdown by Direction
    # s48=used (1) -> cb=new (0): used -> new
    # s48=new (0) -> cb=used (1): new -> used
    flips_used_to_new = flips_mask & (s48_y_used == 1) & (pred_catboost_y == 0)
    flips_new_to_used = flips_mask & (s48_y_used == 0) & (pred_catboost_y == 1)
    
    print(f"\nDirection of flips:", flush=True)
    print(f"  used -> new: {flips_used_to_new.sum()} (Seen: {(flips_used_to_new & test_is_seen).sum()}, New: {(flips_used_to_new & ~test_is_seen).sum()})", flush=True)
    print(f"  new -> used: {flips_new_to_used.sum()} (Seen: {(flips_new_to_used & test_is_seen).sum()}, New: {(flips_new_to_used & ~test_is_seen).sum()})", flush=True)
    
    # Confidence Margin Analysis
    margins = np.abs(test_catboost_p - 0.50)
    print(f"\nMargin percentiles on flips (|p - 0.5|):", flush=True)
    for q in [10, 25, 50, 75, 90, 95]:
        print(f"  p{q}: {np.percentile(margins[flips_mask], q):.4f}", flush=True)
        
    # Write full flip audit CSV
    audit_file = RESULTS_DIR / "flips_dual_catboost_vs_s48.csv"
    with open(audit_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "test_id", "seller_id", "is_seen", "s48_label", "cb_label",
                    "cb_p_used", "margin", "direction", "title", "price", "category_id"])
        for i in range(len(test_rows)):
            if flips_mask[i]:
                direction = "used->new" if s48_y_used[i] == 1 else "new->used"
                w.writerow([
                    i + 1, test_ids[i], test_sellers[i], int(test_is_seen[i]),
                    s48_labels[i], "used" if pred_catboost_y[i] == 1 else "new",
                    round(float(test_catboost_p[i]), 5),
                    round(float(margins[i]), 5),
                    direction,
                    test_titles[i],
                    test_rows[i].get("price"),
                    test_rows[i].get("category_id")
                ])
    print(f"\nWrote flip audit CSV to {audit_file}", flush=True)
    
    # 11. Generate Candidate Submissions
    
    # Option 1: Pure Dual-Text CatBoost Full (all 30k)
    cand_pure_path = SUBMISSIONS_DIR / "submission_64_dual_catboost_pure.csv"
    with open(cand_pure_path, "w", newline="\n", encoding="utf-8") as f:
        f.write("ID,condition\n")
        for i in range(len(test_rows)):
            cond = "used" if pred_catboost_y[i] == 1 else "new"
            f.write(f"{i+1},{cond}\n")
    sha_pure = compute_sha256(cand_pure_path)
    print(f"\n[Candidate 1: Pure Dual CatBoost Full]")
    print(f"  Path: {cand_pure_path}")
    print(f"  Total flips vs 48: {n_flips}")
    print(f"  SHA256: {sha_pure}")
    
    # Option 2: Conservative Target - New Sellers Only with high margin (> 0.25, i.e., p < 0.25 or p > 0.75)
    # Why? Round 144 proved the Dual-Text gain is concentrated (+25 and +34 net) in NEW SELLERS,
    # while leaving seen sellers untouched protects against the seen-seller penalty observed in fin_stack.
    cand_cons_new_path = SUBMISSIONS_DIR / "submission_65_dual_catboost_new_sellers_top_margin.csv"
    cons_mask = (~test_is_seen) & flips_mask & (margins >= 0.25)
    n_cons = int(cons_mask.sum())
    
    with open(cand_cons_new_path, "w", newline="\n", encoding="utf-8") as f:
        f.write("ID,condition\n")
        for i in range(len(test_rows)):
            if cons_mask[i]:
                cond = "used" if pred_catboost_y[i] == 1 else "new"
            else:
                cond = s48_labels[i]
            f.write(f"{i+1},{cond}\n")
    sha_cons = compute_sha256(cand_cons_new_path)
    print(f"\n[Candidate 2: Conservative Dual CatBoost - New Sellers High Margin (|p-0.5| >= 0.25)]")
    print(f"  Path: {cand_cons_new_path}")
    print(f"  Total flips vs 48: {n_cons}")
    print(f"  Direction breakdown: used->new: {(cons_mask & flips_used_to_new).sum()}, new->used: {(cons_mask & flips_new_to_used).sum()}")
    print(f"  SHA256: {sha_cons}")
    
    # Save full run metadata
    run_meta = {
        "timestamp": "2026-10-29",
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "train_oof_sparse_acc": float(accuracy_score(y_train, train_oof_sparse > 0.5)),
        "train_oof_nbsvm_acc": float(accuracy_score(y_train, train_oof_nbsvm > 0.5)),
        "n_flips_pure": n_flips,
        "n_flips_seen": int(flips_seen.sum()),
        "n_flips_new": int(flips_new.sum()),
        "flips_used_to_new": int(flips_used_to_new.sum()),
        "flips_new_to_used": int(flips_new_to_used.sum()),
        "candidate_pure": {
            "path": str(cand_pure_path),
            "sha256": sha_pure,
            "flips": n_flips
        },
        "candidate_conservative_new": {
            "path": str(cand_cons_new_path),
            "sha256": sha_cons,
            "flips": n_cons,
            "flips_used_to_new": int((cons_mask & flips_used_to_new).sum()),
            "flips_new_to_used": int((cons_mask & flips_new_to_used).sum())
        }
    }
    with open(RESULTS_DIR / "round145_full_train_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)
    print(f"\nRun metadata saved to {RESULTS_DIR / 'round145_full_train_metadata.json'}", flush=True)


if __name__ == "__main__":
    main()
