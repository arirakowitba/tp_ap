"""Enriched Intrinsic Features & Fine-Grained Semantic Condition Validation.

Tests whether fine-grained Spanish condition cues (e.g. 'como nuevo' -> used,
'para repuesto' -> used, 'sellado de fabrica' -> new), multi-level relative price
z-scores, and listing trust indicators (official store, deals, warranty tiers)
provide genuine, leak-free gains over the baseline CatBoost on identical
external seller-group splits (seeds 2026 and 314159).
"""
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
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


def extract_enriched_features(rows, indices, fit_stats=None):
    """Extract baseline features + enriched fine-grained condition and trust cues."""
    USED_PATTERNS = {
        "cue_como_nuevo": r"(?i)\bcomo\s+nuev[oa]s?\b|\bigual\s+a\s+nuev[oa]s?\b|\bcasi\s+nuev[oa]s?\b",
        "cue_poco_uso": r"(?i)\bpoco\s+uso\b|\bcon\s+uso\b|\bsegunda\s+mano\b",
        "cue_impecable": r"(?i)\bimpecable\b|\bexcelente\s+estado\b|\bmuy\s+buen\s+estado\b|\bbuen\s+estado\b",
        "cue_repuesto": r"(?i)\brepuesto\b|\bpara\s+repuesto\b|\bdesarme\b|\brot[oa]\b|\ba\s+reparar\b|\bpara\s+piezas\b",
        "cue_detalles": r"(?i)\bcon\s+detalles?\b|\bdetalles?\s+de\s+uso\b|\bmarcas?\s+de\s+uso\b",
        "cue_no_funciona": r"(?i)\bno\s+funciona\b|\bfunciona\s+a\s+revisar\b",
        "cue_usado": r"(?i)\busad[oa]s?\b",
        "cue_vintage": r"(?i)\bvintage\b|\bantigu[oa]s?\b|\bde\s+colecci[oó]n\b|\bretro\b|\bde\s+[eé]poca\b",
        "cue_lote": r"(?i)\blote\b|\bcolecci[oó]n\b",
        "cue_reacondicionado": r"(?i)\breacondicionad[oa]\b|\brefurbished\b|\boutlet\b",
        "warr_sin": r"(?i)\bsin\s+garant[ií]a\b|\bno\s+tiene\s+garant[ií]a\b|\bno\s+posee\b",
        "warr_prueba": r"(?i)\bprueba\b|\bprobar\b|\bfunciona\s+bien\b|\bse\s+prueba\b",
        "warr_dias": r"(?i)\b\d+\s*d[ií]as?\b",
    }
    
    NEW_PATTERNS = {
        "cue_nuevo": r"(?i)\bnuev[oa]s?\b",
        "cue_sellado": r"(?i)\bsellad[oa]s?\b|\bcerrad[oa]\s+de\s+f[aá]brica\b|\ben\s+bl[ií]ster\b|\b0km\b|\b0\s*km\b",
        "cue_en_caja": r"(?i)\ben\s+caja\b|\bcaja\s+original\b",
        "cue_sin_uso": r"(?i)\bsin\s+uso\b|\bnunca\s+usad[oa]\b|\ba\s+estrenar\b",
        "cue_original": r"(?i)\boriginal\b|\b100%\s*original\b",
        "cue_factura": r"(?i)\bfactura\s+[ab]\b|\bhacemos\s+factura\b|\bemitimos\s+factura\b",
        "warr_fabrica": r"(?i)\bf[aá]brica\b|\boficial\b|\bde\s+por\s+vida\b|\bgarant[ií]a\s+escrita\b",
        "warr_meses": r"(?i)\bmes(es)?\b|\ba[ñn]os?\b",
    }
    
    data = []
    for idx in indices:
        r = rows[idx]
        title = r.get("title") or ""
        warranty = r.get("warranty") or ""
        sh = r.get("shipping") or {}
        adr = r.get("seller_address") or {}
        city = ((adr.get("city") or {}).get("name") or "").lower().strip()
        state = (adr.get("state") or {}).get("id") or ""
        pics = r.get("pictures") or []
        attrs = r.get("attributes") or []
        variations = r.get("variations") or []
        tags = r.get("tags") or []
        
        price = float(r.get("price") or 0)
        log_price = np.log1p(price)
        cat_id = str(r.get("category_id") or "")
        
        # Relative price stats
        if fit_stats:
            cat_mean = fit_stats["mean"].get(cat_id, fit_stats["overall_mean"])
            cat_std = fit_stats["std"].get(cat_id, fit_stats["overall_std"])
            cat_med = fit_stats["med"].get(cat_id, fit_stats["overall_med"])
            z_price_cat = (log_price - cat_mean) / (cat_std + 1e-4)
            rel_price_cat = log_price - cat_med
        else:
            z_price_cat = 0.0
            rel_price_cat = 0.0
            
        init_q = float(r.get("initial_quantity") or 0)
        avail_q = float(r.get("available_quantity") or 0)
        sold_q = float(r.get("sold_quantity") or 0)
        
        feat = {
            "log_price": log_price,
            "rel_price_cat": rel_price_cat,
            "z_price_cat": z_price_cat,
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
            "is_official_store": int(r.get("official_store_id") is not None),
            "has_deals": int(len(r.get("deal_ids") or []) > 0),
            "category_id": cat_id,
            "listing_type_id": str(r.get("listing_type_id") or ""),
            "buying_mode": str(r.get("buying_mode") or ""),
            "shipping_mode": str(sh.get("mode") or ""),
            "state": state,
        }
        
        # Enriched cues
        used_score = 0
        for name, pat in USED_PATTERNS.items():
            hit = int(bool(re.search(pat, title + " " + warranty)))
            feat[name] = hit
            used_score += hit
        feat["total_used_cues"] = used_score
        
        new_score = 0
        for name, pat in NEW_PATTERNS.items():
            hit = int(bool(re.search(pat, title + " " + warranty)))
            feat[name] = hit
            new_score += hit
        feat["total_new_cues"] = new_score
        feat["net_cue_score"] = new_score - used_score
        
        data.append(feat)
        
    return data


def compute_fit_price_stats(rows, fit_indices):
    cat_prices = defaultdict(list)
    all_prices = []
    for idx in fit_indices:
        r = rows[idx]
        price = float(r.get("price") or 0)
        lp = np.log1p(price)
        cat_id = str(r.get("category_id") or "")
        cat_prices[cat_id].append(lp)
        all_prices.append(lp)
        
    overall_mean = float(np.mean(all_prices)) if all_prices else 0.0
    overall_std = float(np.std(all_prices)) if all_prices else 1.0
    overall_med = float(np.median(all_prices)) if all_prices else 0.0
    
    means = {}
    stds = {}
    meds = {}
    for cat_id, lps in cat_prices.items():
        means[cat_id] = float(np.mean(lps))
        stds[cat_id] = float(np.std(lps)) if len(lps) > 1 else overall_std
        meds[cat_id] = float(np.median(lps))
        
    return {
        "mean": means, "std": stds, "med": meds,
        "overall_mean": overall_mean, "overall_std": overall_std, "overall_med": overall_med
    }


def main():
    rows, ids, sellers, y_used, titles, warranties = load_train_data()
    seeds = [2026, 314159]
    all_results = []
    
    for seed in seeds:
        print(f"\n=======================================================", flush=True)
        print(f"RUNNING ENRICHED CUES EXPERIMENT ON SEED {seed}", flush=True)
        print(f"=======================================================", flush=True)
        
        fit, tune, confirm = split_seller_groups(sellers, seed)
        fit_stats = compute_fit_price_stats(rows, fit)
        
        meta_fit = extract_enriched_features(rows, fit, fit_stats)
        meta_tune = extract_enriched_features(rows, tune, fit_stats)
        meta_confirm = extract_enriched_features(rows, confirm, fit_stats)
        
        # 1. Text Vectorizers on fit
        print("Fitting text vectorizers on fit...", flush=True)
        vec_title_word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=75000,
                                         sublinear_tf=True, strip_accents="unicode")
        vec_title_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                                         max_features=120000, sublinear_tf=True)
        vec_warr = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=25000,
                                   sublinear_tf=True, strip_accents="unicode")
        
        w_fit = vec_title_word.fit_transform([titles[i] for i in fit])
        c_fit = vec_title_char.fit_transform([titles[i] for i in fit])
        wr_fit = vec_warr.fit_transform([warranties[i] for i in fit])
        
        dict_vec = DictVectorizer(sparse=True)
        m_fit = dict_vec.fit_transform(meta_fit)
        X_sparse_fit = hstack([w_fit, c_fit, wr_fit, m_fit], format="csr")
        
        w_tune = vec_title_word.transform([titles[i] for i in tune])
        c_tune = vec_title_char.transform([titles[i] for i in tune])
        wr_tune = vec_warr.transform([warranties[i] for i in tune])
        m_tune = dict_vec.transform(meta_tune)
        X_sparse_tune = hstack([w_tune, c_tune, wr_tune, m_tune], format="csr")
        
        w_confirm = vec_title_word.transform([titles[i] for i in confirm])
        c_confirm = vec_title_char.transform([titles[i] for i in confirm])
        wr_confirm = vec_warr.transform([warranties[i] for i in confirm])
        m_confirm = dict_vec.transform(meta_confirm)
        X_sparse_confirm = hstack([w_confirm, c_confirm, wr_confirm, m_confirm], format="csr")
        
        # 2. Honest 4-fold GroupKFold Sparse OOF
        print("Fitting Sparse OOF on fit...", flush=True)
        oof_p_sparse = np.zeros(len(fit), dtype=np.float32)
        gkf = GroupKFold(n_splits=4)
        for fold, (tr, va) in enumerate(gkf.split(X_sparse_fit, y_used[fit], sellers[fit])):
            lr_fold = LogisticRegression(C=2.5, max_iter=200, solver="liblinear", random_state=seed + fold)
            lr_fold.fit(X_sparse_fit[tr], y_used[fit][tr])
            oof_p_sparse[va] = lr_fold.predict_proba(X_sparse_fit[va])[:, 1]
            
        lr_full = LogisticRegression(C=2.5, max_iter=250, solver="liblinear", random_state=seed)
        lr_full.fit(X_sparse_fit, y_used[fit])
        p_sparse_tune = lr_full.predict_proba(X_sparse_tune)[:, 1]
        p_sparse_confirm = lr_full.predict_proba(X_sparse_confirm)[:, 1]
        
        # 3. Features for CatBoost
        CAT_COLS = ["category_id", "listing_type_id", "buying_mode", "shipping_mode", "state"]
        NUM_COLS = [k for k in meta_fit[0].keys() if k not in CAT_COLS]
        
        df_cat_fit = pd.DataFrame([{col: m[col] for col in NUM_COLS + CAT_COLS} for m in meta_fit])
        df_cat_fit["sparse_p"] = oof_p_sparse
        for c in CAT_COLS: df_cat_fit[c] = df_cat_fit[c].astype(str)
        
        df_cat_tune = pd.DataFrame([{col: m[col] for col in NUM_COLS + CAT_COLS} for m in meta_tune])
        df_cat_tune["sparse_p"] = p_sparse_tune
        for c in CAT_COLS: df_cat_tune[c] = df_cat_tune[c].astype(str)
        
        df_cat_confirm = pd.DataFrame([{col: m[col] for col in NUM_COLS + CAT_COLS} for m in meta_confirm])
        df_cat_confirm["sparse_p"] = p_sparse_confirm
        for c in CAT_COLS: df_cat_confirm[c] = df_cat_confirm[c].astype(str)
        
        # 4. Train Enriched CatBoost
        print(f"Training Enriched CatBoost ({len(NUM_COLS)+len(CAT_COLS)+1} features)...", flush=True)
        cat_pool_fit = Pool(df_cat_fit, y_used[fit], cat_features=CAT_COLS)
        cat_pool_tune = Pool(df_cat_tune, cat_features=CAT_COLS)
        cat_pool_confirm = Pool(df_cat_confirm, cat_features=CAT_COLS)
        
        cb_enriched = CatBoostClassifier(
            iterations=1000,
            depth=5,
            learning_rate=0.07,
            l2_leaf_reg=5.0,
            random_seed=seed,
            thread_count=12,
            verbose=0
        )
        t0 = time.time()
        cb_enriched.fit(cat_pool_fit)
        print(f"Enriched CatBoost fitted in {time.time()-t0:.2f}s", flush=True)
        
        p_enr_tune = cb_enriched.predict_proba(cat_pool_tune)[:, 1]
        p_enr_confirm = cb_enriched.predict_proba(cat_pool_confirm)[:, 1]
        
        # Load baseline predictions from previous round
        pairs_prev = pd.read_csv(RESULTS_DIR / f"paired_predictions_seed_{seed}.csv")
        p_base_confirm = pairs_prev["p_cat_used"].values
        pred_base_confirm = pairs_prev["pred_cat_used"].values
        
        # Check alignment
        assert np.array_equal(ids[confirm], pairs_prev["id"].astype(str).values)
        
        # Evaluate Enriched CatBoost vs Baseline CatBoost on Confirm
        y_c = y_used[confirm]
        pred_enr_confirm = (p_enr_confirm >= 0.5).astype(int)
        
        acc_base = accuracy_score(y_c, pred_base_confirm)
        acc_enr = accuracy_score(y_c, pred_enr_confirm)
        ll_base = log_loss(y_c, p_base_confirm)
        ll_enr = log_loss(y_c, p_enr_confirm)
        
        fixes = int(np.sum((pred_base_confirm != y_c) & (pred_enr_confirm == y_c)))
        breaks = int(np.sum((pred_base_confirm == y_c) & (pred_enr_confirm != y_c)))
        net = fixes - breaks
        
        ci_low, ci_high, _ = cluster_bootstrap_ci(y_c, pred_base_confirm, pred_enr_confirm, sellers[confirm])
        
        seen_mask = pairs_prev["is_seen_seller"].values.astype(bool)
        new_mask = ~seen_mask
        
        acc_base_seen = accuracy_score(y_c[seen_mask], pred_base_confirm[seen_mask])
        acc_enr_seen = accuracy_score(y_c[seen_mask], pred_enr_confirm[seen_mask])
        acc_base_new = accuracy_score(y_c[new_mask], pred_base_confirm[new_mask])
        acc_enr_new = accuracy_score(y_c[new_mask], pred_enr_confirm[new_mask])
        
        print(f"\nCONFIRM COMPARISON (Seed {seed}):", flush=True)
        print(f"  Baseline CatBoost Acc: {acc_base:.5f}, LL: {ll_base:.5f}", flush=True)
        print(f"  Enriched CatBoost Acc: {acc_enr:.5f}, LL: {ll_enr:.5f}", flush=True)
        print(f"  Fixes/Breaks vs Base:  {fixes}/{breaks} -> NET: {net:+d}", flush=True)
        print(f"  Bootstrap 95% CI:      [{ci_low:+.1f}, {ci_high:+.1f}]", flush=True)
        print(f"  Seen sellers Acc:      {acc_base_seen:.5f} -> {acc_enr_seen:.5f} (n={seen_mask.sum()})", flush=True)
        print(f"  New sellers Acc:       {acc_base_new:.5f} -> {acc_enr_new:.5f} (n={new_mask.sum()})", flush=True)
        
        # Save enriched paired predictions
        out_pairs = RESULTS_DIR / f"paired_enriched_seed_{seed}.csv"
        with open(out_pairs, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "seller_id", "y_true", "is_seen", "p_base_cb", "p_enr_cb", "pred_base_cb", "pred_enr_cb"])
            for idx, i in enumerate(confirm):
                w.writerow([
                    ids[i], sellers[i], y_used[i], int(seen_mask[idx]),
                    round(float(p_base_confirm[idx]), 5),
                    round(float(p_enr_confirm[idx]), 5),
                    int(pred_base_confirm[idx]),
                    int(pred_enr_confirm[idx])
                ])
                
        all_results.append({
            "seed": seed,
            "confirm_n": len(confirm),
            "acc_base": float(acc_base),
            "acc_enr": float(acc_enr),
            "ll_base": float(ll_base),
            "ll_enr": float(ll_enr),
            "fixes": fixes,
            "breaks": breaks,
            "net": net,
            "ci_95": [ci_low, ci_high],
            "seen_acc_base": float(acc_base_seen),
            "seen_acc_enr": float(acc_enr_seen),
            "new_acc_base": float(acc_base_new),
            "new_acc_enr": float(acc_enr_new),
        })
        
    summary_file = RESULTS_DIR / "round143_enriched_features_results.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {summary_file}", flush=True)


if __name__ == "__main__":
    main()
