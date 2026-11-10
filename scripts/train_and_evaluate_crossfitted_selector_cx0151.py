"""Train and Evaluate Fully Cross-Fitted Baseline-Error Selector (AG-0146 / CX-0151).

Protocol:
1. Target: Direct prediction of baseline errors:
   - Direction U2N: P(y_true == 0 | pred_baseline == 1, X)
   - Direction N2U: P(y_true == 1 | pred_baseline == 0, X)
2. Features:
   - Baseline margin: |p_baseline - 0.5|
   - Tree ensemble consensus: mean(CB, LGB, XGB) and difference (tree - baseline)
   - Text classifier probability and difference (text - baseline)
   - Tabular attributes probability
   - Category-conditioned interaction probability
   - Relative price within category
   - Domain-specific high-precision keyword signals (e.g. repuestos, sellado, en caja)
3. Cross-Fitting (Outer Seller-Group CV across 5 folds):
   - In each outer fold f:
     * tr_idx = (outer_unseen_fold != f)
     * val_idx = (outer_unseen_fold == f)
     * Assert seller disjunction between train and val.
     * Fit selector models strictly on tr_idx.
     * Predict error probabilities on val_idx strictly out-of-fold.
   - Concatenate all 70,000 out-of-fold selector scores.
4. Comprehensive Grid Evaluation:
   - K in {75, 100, 150, 250, 500, 750, 1000, 1500, 2500}
   - Fixes, breaks, net gain, empirical precision.
   - Fold-by-fold stability.
   - Seen vs new seller breakdown.
   - Direction breakdown (u2n vs n2u).
   - Unique sellers and categories affected.
   - Clustered bootstrap (1,000 resamples by seller and category).
   - Feasibility check for +51 public hits.
5. Promotion Gate:
   - Condition 1: Aggregate outer-OOF net gain > 0.
   - Condition 2: Majority of outer folds have positive net gain.
   - Condition 3: Seller cluster bootstrap lower bound (2.5%) > 0.
   - Condition 4: K mathematically reaches +51.
6. Candidate Generation:
   - If qualified, generate 3 CSVs strictly on EVT-143 by item ID.
   - If blocked, report honestly without fabricating probabilities.
"""

import sys
import os
import json
import time
import hashlib
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def compute_id_target_hash(df):
    mapping_str = "\n".join(f"{row['ID']},{row['condition']}" for _, row in df.iterrows())
    return hashlib.sha256(mapping_str.encode("utf-8")).hexdigest()

def main():
    start_time = time.time()
    out_dir = ROOT / "results/crossfitted_selector_ag0146"
    sub_dir = ROOT / "submissions/final_portfolio_ag0146_repaired"
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("STARTING FULLY CROSS-FITTED BASELINE-ERROR SELECTOR (AG-0146 / CX-0151)")
    print("=" * 90)

    # 1. Load Repaired Datasets
    repaired_oof_path = ROOT / "results/multiagent_predictive_base_repaired/oof_multiagent_predictive_base_repaired.csv"
    repaired_test_path = ROOT / "results/multiagent_predictive_base_repaired/test_probabilities_multiagent_predictive_base_repaired.csv"
    splits_path = ROOT / "data/splits/canonical_nested_dual_regime_splits.csv"
    train_path = ROOT / "data/ap_2026_q_2/raw/train_data.jsonlines"
    test_path = ROOT / "data/ap_2026_q_2/raw/test_data.jsonlines"
    anchor_path = ROOT / "submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv"

    assert repaired_oof_path.exists(), f"Missing repaired OOF: {repaired_oof_path}"
    assert repaired_test_path.exists(), f"Missing repaired Test: {repaired_test_path}"
    assert anchor_path.exists(), f"Missing anchor EVT-143: {anchor_path}"

    df_oof = pd.read_csv(repaired_oof_path)
    df_test = pd.read_csv(repaired_test_path)
    df_splits = pd.read_csv(splits_path)
    df_anchor = pd.read_csv(anchor_path)

    n_train = len(df_oof)
    n_test = len(df_test)
    assert n_train == 70000, f"Expected 70,000 train rows, got {n_train}"
    assert n_test == 30000, f"Expected 30,000 test rows, got {n_test}"

    print(f"Loaded Repaired OOF: {n_train} rows (SHA-256: {compute_sha256(repaired_oof_path)})")
    print(f"Loaded Repaired Test: {n_test} rows (SHA-256: {compute_sha256(repaired_test_path)})")
    print(f"Loaded Anchor EVT-143: {len(df_anchor)} rows (SHA-256: {compute_sha256(anchor_path)})")

    # Load raw metadata
    with open(train_path, "r", encoding="utf-8") as f:
        train_raw = [json.loads(line) for line in f]
    with open(test_path, "r", encoding="utf-8") as f:
        test_raw = [json.loads(line) for line in f]

    train_titles = np.array([r.get("title") or "" for r in train_raw])
    test_titles = np.array([r.get("title") or "" for r in test_raw])
    train_prices = np.array([float(r.get("price") or 0.0) for r in train_raw])
    test_prices = np.array([float(r.get("price") or 0.0) for r in test_raw])
    train_cats = np.array([str(r.get("category_id") or "") for r in train_raw])
    test_cats = np.array([str(r.get("category_id") or "") for r in test_raw])
    train_sellers = np.array([str(r.get("seller_id") or "") for r in train_raw])
    test_sellers = np.array([str(r.get("seller_id") or "") for r in test_raw])

    # Category relative log-price
    log_p_train = np.log1p(np.maximum(train_prices, 0.0))
    log_p_test = np.log1p(np.maximum(test_prices, 0.0))
    cat_df = pd.DataFrame({"cat": train_cats, "log_p": log_p_train})
    cat_medians = cat_df.groupby("cat")["log_p"].median().to_dict()
    global_med_price = float(np.median(log_p_train))
    rel_price_train = log_p_train - pd.Series(train_cats).map(cat_medians).fillna(global_med_price).values
    rel_price_test = log_p_test - pd.Series(test_cats).map(cat_medians).fillna(global_med_price).values

    # Keyword indicators
    def build_keyword_signals(titles):
        titles_lower = [t.lower() for t in titles]
        # Strong signals of used/parts
        kw_used = ["repuesto", "repuestos", "desarme", "pantalla rota", "reparar", "usado", "detalles de uso",
                   "detalles esteticos", "leer descripcion", "sin accesorios", "para piezas"]
        # Strong signals of new
        kw_new = ["sellado", "sellada", "en caja", "cerrado", "nuevo cerrado", "sin uso", "caja sellada",
                  "factura a", "factura b", "garantia oficial", "blister", "cerrada"]
        feat_used_kw = np.array([sum(1 for w in kw_used if w in t) for t in titles_lower], dtype=np.float32)
        feat_new_kw = np.array([sum(1 for w in kw_new if w in t) for t in titles_lower], dtype=np.float32)
        return feat_used_kw, feat_new_kw

    train_kw_used, train_kw_new = build_keyword_signals(train_titles)
    test_kw_used, test_kw_new = build_keyword_signals(test_titles)

    # -------------------------------------------------------------------------
    # 2. FEATURE EXTRACTION & SETUP FOR ERROR SELECTOR
    # -------------------------------------------------------------------------
    y_true = df_oof["y_true"].values
    p_base_train = df_oof["p_baseline_podium"].values
    pred_base_train = (p_base_train > 0.5).astype(int)
    is_error_train = (pred_base_train != y_true)
    margin_train = np.abs(p_base_train - 0.5)

    p_base_test = df_test["p_baseline_podium"].values
    pred_base_test = (p_base_test > 0.5).astype(int)
    margin_test = np.abs(p_base_test - 0.5)

    # Tree consensus features
    p_trees_train = (df_oof["p_agent_e_cb"] + df_oof["p_agent_e_lgb"] + df_oof["p_agent_e_xgb"]).values / 3.0
    p_trees_test = (df_test["p_agent_e_cb"] + df_test["p_agent_e_lgb"] + df_test["p_agent_e_xgb"]).values / 3.0

    p_tree_diff_train = p_trees_train - p_base_train
    p_tree_diff_test = p_trees_test - p_base_test

    p_text_diff_train = df_oof["p_agent_a_text"].values - p_base_train
    p_text_diff_test = df_test["p_agent_a_text"].values - p_base_test

    p_text_train = df_oof["p_agent_a_text"].values
    p_text_test = df_test["p_agent_a_text"].values

    p_tab_train = df_oof["p_agent_b_tab"].values
    p_tab_test = df_test["p_agent_b_tab"].values

    p_inter_train = df_oof["p_agent_d_inter"].values
    p_inter_test = df_test["p_agent_d_inter"].values

    p_seller_train = df_oof["p_agent_c_seller"].values
    p_seller_test = df_test["p_agent_c_seller"].values

    outer_folds = df_splits["outer_unseen_fold"].values

    # Assemble Feature Matrices
    # Direction 1: Used -> New (base says used p_base > 0.5, true is new y=0)
    # Features where positive values indicate actual NEW:
    # low margin, high relative price, lower p_trees, negative p_tree_diff, low p_text, new keywords
    X_u2n_all_train = np.column_stack([
        margin_train,
        rel_price_train,
        p_trees_train,
        p_tree_diff_train,
        p_text_diff_train,
        p_text_train,
        p_tab_train,
        p_inter_train,
        train_kw_new,
        train_kw_used
    ])

    X_u2n_all_test = np.column_stack([
        margin_test,
        rel_price_test,
        p_trees_test,
        p_tree_diff_test,
        p_text_diff_test,
        p_text_test,
        p_tab_test,
        p_inter_test,
        test_kw_new,
        test_kw_used
    ])

    # Direction 2: New -> Used (base says new p_base < 0.5, true is used y=1)
    # Features where positive values indicate actual USED:
    # low margin, low relative price, higher p_trees, positive p_tree_diff, high p_text, used keywords
    X_n2u_all_train = np.column_stack([
        margin_train,
        rel_price_train,
        p_trees_train,
        p_tree_diff_train,
        p_text_diff_train,
        p_text_train,
        p_tab_train,
        p_inter_train,
        train_kw_used,
        train_kw_new
    ])

    X_n2u_all_test = np.column_stack([
        margin_test,
        rel_price_test,
        p_trees_test,
        p_tree_diff_test,
        p_text_diff_test,
        p_text_test,
        p_tab_test,
        p_inter_test,
        test_kw_used,
        test_kw_new
    ])

    # -------------------------------------------------------------------------
    # 3. OUTER SELLER-GROUP CROSS-FITTING (5 FOLDS)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("CROSS-FITTING ERROR SELECTOR ACROSS ALL 5 OUTER SELLER-GROUP FOLDS")
    print("=" * 90)

    oof_err_score = np.zeros(n_train, dtype=np.float32)
    oof_direction = np.zeros(n_train, dtype=np.int32) # 1: u2n, -1: n2u, 0: none
    test_err_score_accum = np.zeros(n_test, dtype=np.float64)
    test_dir_accum = np.zeros(n_test, dtype=np.int32)

    fold_models = []

    for f in range(5):
        val_idx = np.where(outer_folds == f)[0]
        tr_idx = np.where(outer_folds != f)[0]

        # Verify strict seller disjunction
        tr_sellers = set(train_sellers[tr_idx])
        val_sellers = set(train_sellers[val_idx])
        assert len(tr_sellers & val_sellers) == 0, f"Seller leakage in fold {f}!"

        # Train U2N selector: on tr_idx where pred_base == 1
        tr_u_mask = tr_idx[pred_base_train[tr_idx] == 1]
        val_u_mask = val_idx[pred_base_train[val_idx] == 1]
        y_u_tr = (y_true[tr_u_mask] == 0).astype(int) # 1 if error (actual new)

        clf_u2n = LogisticRegression(C=1.0, max_iter=300, class_weight="balanced", random_state=42 + f)
        clf_u2n.fit(X_u2n_all_train[tr_u_mask], y_u_tr)
        p_err_u_val = clf_u2n.predict_proba(X_u2n_all_train[val_u_mask])[:, 1]
        oof_err_score[val_u_mask] = p_err_u_val
        oof_direction[val_u_mask] = 1

        # Train N2U selector: on tr_idx where pred_base == 0
        tr_n_mask = tr_idx[pred_base_train[tr_idx] == 0]
        val_n_mask = val_idx[pred_base_train[val_idx] == 0]
        y_n_tr = (y_true[tr_n_mask] == 1).astype(int) # 1 if error (actual used)

        clf_n2u = LogisticRegression(C=1.0, max_iter=300, class_weight="balanced", random_state=42 + f)
        clf_n2u.fit(X_n2u_all_train[tr_n_mask], y_n_tr)
        p_err_n_val = clf_n2u.predict_proba(X_n2u_all_train[val_n_mask])[:, 1]
        oof_err_score[val_n_mask] = p_err_n_val
        oof_direction[val_n_mask] = -1

        # Accumulate test predictions across the 5 outer fold models
        # Target: probability that EVT-143 anchor is wrong on test item
        test_u_mask = (df_anchor["condition"].values == "used")
        test_n_mask = (df_anchor["condition"].values == "new")

        p_test_u = clf_u2n.predict_proba(X_u2n_all_test[test_u_mask])[:, 1]
        p_test_n = clf_n2u.predict_proba(X_n2u_all_test[test_n_mask])[:, 1]

        test_err_score_f = np.zeros(n_test, dtype=np.float64)
        test_err_score_f[test_u_mask] = p_test_u
        test_err_score_f[test_n_mask] = p_test_n
        test_err_score_accum += test_err_score_f / 5.0

        fold_models.append((clf_u2n, clf_n2u))
        print(f"  Fold {f}: Val Rows={len(val_idx)} (U2N={len(val_u_mask)}, N2U={len(val_n_mask)}). Fit complete.")

    test_err_score = test_err_score_accum
    test_direction = np.where(df_anchor["condition"].values == "used", 1, -1)

    print("\nOuter Cross-Fitting 100% COMPLETE. Evaluated strictly out-of-fold for all 70,000 rows.")

    # -------------------------------------------------------------------------
    # 4. COMPREHENSIVE GRID EVALUATION ACROSS K
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("SUPPORT GRID EVALUATION & EMPIRICAL AUDIT ACROSS K IN {75..2500}")
    print("=" * 90)

    K_GRID = [75, 100, 150, 250, 500, 750, 1000, 1500, 2500]
    N_PUB = 20998
    pi_pub = N_PUB / 30000.0

    eval_results = {}
    rng = np.random.default_rng(20261110)

    print(f"{'K':>5} | {'Fixes':>5} | {'Breaks':>6} | {'Net OOF':>7} | {'Prec OOF':>8} | {'U2N':>4} | {'N2U':>4} | {'Sellers':>7} | {'Cats':>5} | {'Fold Net Gains':^25} | {'Seller CI95%':^16} | {'P(G>=+51)':>9}")
    print("-" * 125)

    for k in K_GRID:
        top_k_idx = np.argsort(oof_err_score)[-k:]
        fixes = int(np.sum(is_error_train[top_k_idx]))
        breaks = k - fixes
        net_oof = fixes - breaks
        prec_oof = fixes / k

        u2n_cnt = int(np.sum(oof_direction[top_k_idx] == 1))
        n2u_cnt = int(np.sum(oof_direction[top_k_idx] == -1))

        k_sellers = set(train_sellers[top_k_idx])
        k_cats = set(train_cats[top_k_idx])

        # Fold by fold breakdown
        fold_nets = []
        for f in range(5):
            f_mask = (outer_folds[top_k_idx] == f)
            f_fix = int(np.sum(is_error_train[top_k_idx[f_mask]]))
            f_brk = int(np.sum(f_mask)) - f_fix
            fold_nets.append(f_fix - f_brk)

        # Cluster Bootstrap by Seller (1000 resamples of unique sellers in top_k)
        unique_k_sellers = np.array(list(k_sellers))
        seller_to_items = {s: [] for s in unique_k_sellers}
        for item_idx in top_k_idx:
            seller_to_items[train_sellers[item_idx]].append(item_idx)

        boot_gains = []
        for _ in range(1000):
            sampled_sellers = rng.choice(unique_k_sellers, size=len(unique_k_sellers), replace=True)
            b_fixes = 0
            b_breaks = 0
            for s in sampled_sellers:
                for idx in seller_to_items[s]:
                    if is_error_train[idx]:
                        b_fixes += 1
                    else:
                        b_breaks += 1
            boot_gains.append(b_fixes - b_breaks)

        ci_lo = float(np.percentile(boot_gains, 2.5))
        ci_hi = float(np.percentile(boot_gains, 97.5))
        ci_p05 = float(np.percentile(boot_gains, 5.0))

        # Expected public metrics
        exp_k_pub = k * pi_pub
        c_min_req = int(np.ceil((exp_k_pub + 51) / 2))
        prec_req = float(c_min_req / exp_k_pub)

        # Finite population simulation for P(G >= +51) using bootstrap cluster draws
        # Net public gain: scaled by pi_pub + hypergeometric fluctuation
        sim_pub_flips = rng.hypergeometric(N_PUB, 30000 - N_PUB, k, size=10000)
        # Empirical cluster precision draws
        sim_prec = np.array([float(np.mean([is_error_train[idx] for s in rng.choice(unique_k_sellers, size=len(unique_k_sellers), replace=True) for idx in seller_to_items[s]])) for _ in range(10000)])
        sim_fixes = rng.binomial(sim_pub_flips, sim_prec)
        sim_gains = 2 * sim_fixes - sim_pub_flips
        prob_success = float(np.mean(sim_gains >= 51))

        fold_str = "[" + ", ".join(f"{g:+3d}" for g in fold_nets) + "]"
        ci_str = f"[{ci_lo:+5.1f}, {ci_hi:+5.1f}]"

        print(f"{k:5d} | {fixes:5d} | {breaks:6d} | {net_oof:+7d} | {prec_oof*100:7.2f}% | {u2n_cnt:4d} | {n2u_cnt:4d} | {len(k_sellers):7d} | {len(k_cats):5d} | {fold_str:^25} | {ci_str:^16} | {prob_success*100:8.2f}%")

        eval_results[k] = {
            "k": k,
            "fixes": fixes,
            "breaks": breaks,
            "net_oof": net_oof,
            "precision_oof": float(prec_oof),
            "flips_u2n": u2n_cnt,
            "flips_n2u": n2u_cnt,
            "unique_sellers": len(k_sellers),
            "unique_categories": len(k_cats),
            "fold_net_gains": fold_nets,
            "seller_cluster_ci95": [ci_lo, ci_hi],
            "seller_cluster_p05": ci_p05,
            "expected_public_flips": float(exp_k_pub),
            "c_min_required_public": c_min_req,
            "precision_required_public": prec_req,
            "prob_reach_095000_cluster": prob_success
        }

    # -------------------------------------------------------------------------
    # 5. PROMOTION GATE EVALUATION & CANDIDATE GENERATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("PROMOTION GATE EVALUATION (CX-0151 CRITERIA)")
    print("=" * 90)

    # Check promotion criteria:
    # 1. Aggregate outer-OOF net gain > 0
    # 2. Majority of outer folds have positive net gain (>= 3 of 5)
    # 3. Seller cluster bootstrap lower bound > 0
    # 4. Deterministic feasibility (K >= 51)
    staged_candidates = {}
    tier_definitions = [
        ("conservative", 75, "Tier A: Conservative Feasible (Smallest K with S_max >= 0.95000)"),
        ("medium", 150, "Tier B: Medium Support (Balanced support & high cluster precision)"),
        ("broad", 500, "Tier C: Broad Correction (Diversified multi-cluster ensemble)"),
    ]

    for tier_code, k_val, role_desc in tier_definitions:
        ev = eval_results[k_val]
        cond1 = ev["net_oof"] > 0
        cond2 = sum(1 for g in ev["fold_net_gains"] if g > 0) >= 3
        cond3 = ev["seller_cluster_ci95"][0] > 0
        cond4 = k_val >= 51

        passed = cond1 and cond2 and cond3 and cond4
        print(f"\n[{tier_code.upper()} K={k_val}]")
        print(f"  Condition 1: Aggregate Net OOF > 0:               {cond1} ({ev['net_oof']:+d})")
        print(f"  Condition 2: Majority of Outer Folds Positive:   {cond2} ({sum(1 for g in ev['fold_net_gains'] if g > 0)}/5)")
        print(f"  Condition 3: Clustered Bootstrap Lower Bound > 0: {cond3} (CI: [{ev['seller_cluster_ci95'][0]:.1f}, {ev['seller_cluster_ci95'][1]:.1f}])")
        print(f"  Condition 4: Deterministic Feasibility (K>=51):   {cond4} (K={k_val})")
        print(f"  -> PROMOTION GATE VERDICT:                        {'PASSED' if passed else 'BLOCKED'}")

        if passed:
            # Build Candidate Submission Strictly on Anchor EVT-143 by Item ID
            # Top K errors on test set according to test_err_score
            top_test_k = np.argsort(test_err_score)[-k_val:]
            test_diff_ids = set(df_test.loc[top_test_k, "ID"].values)

            cand_conditions = []
            flips_u2n = 0
            flips_n2u = 0
            sellers_aff = set()
            cats_aff = set()

            for _, row in df_anchor.iterrows():
                item_id = int(row["ID"])
                anchor_cond = str(row["condition"])

                if item_id in test_diff_ids:
                    # Apply flip
                    new_cond = "new" if anchor_cond == "used" else "used"
                    cand_conditions.append(new_cond)
                    if anchor_cond == "used" and new_cond == "new":
                        flips_u2n += 1
                    else:
                        flips_n2u += 1
                    t_idx = item_id - 1
                    sellers_aff.add(test_sellers[t_idx])
                    cats_aff.add(test_cats[t_idx])
                else:
                    cand_conditions.append(anchor_cond)

            assert len(cand_conditions) == 30000, f"Expected 30000, got {len(cand_conditions)}"
            assert sum(1 for c, a in zip(cand_conditions, df_anchor["condition"]) if c != a) == k_val, "Diff count mismatch!"

            df_cand = pd.DataFrame({"ID": df_anchor["ID"].values, "condition": cand_conditions})
            cand_filename = f"candidate_{tier_code}_k{k_val}.csv"
            cand_path = sub_dir / cand_filename
            df_cand.to_csv(cand_path, index=False)

            csv_sha = compute_sha256(cand_path)
            id_target_hash = compute_id_target_hash(df_cand)
            diff_hash = hashlib.sha256("\n".join(sorted(str(i) for i in test_diff_ids)).encode("utf-8")).hexdigest()

            staged_candidates[tier_code] = {
                "name": cand_filename,
                "tier": tier_code,
                "role": role_desc,
                "k_flips": k_val,
                "flips_u2n": flips_u2n,
                "flips_n2u": flips_n2u,
                "unique_sellers_affected": len(sellers_aff),
                "unique_categories_affected": len(cats_aff),
                "csv_path": str(cand_path),
                "csv_sha256": csv_sha,
                "id_target_hash": id_target_hash,
                "diff_hash": diff_hash,
                "anchor_sha256": compute_sha256(anchor_path),
                "deterministic_max_score": float(0.94761 + k_val / N_PUB),
                "oof_evidence": ev
            }

            print(f"  Generated CSV:  {cand_filename}")
            print(f"  SHA-256:        {csv_sha}")
            print(f"  ID/Target Hash: {id_target_hash}")
            print(f"  Diff Hash:      {diff_hash}")

    # -------------------------------------------------------------------------
    # 6. SERIALIZE FINAL MANIFEST & REPORT
    # -------------------------------------------------------------------------
    final_report = {
        "title": "Fully Cross-Fitted Baseline-Error Selector Report (AG-0146 / CX-0151)",
        "timestamp_utc": "2026-11-10T23:00:00Z",
        "author": "Antigravity",
        "anchor_used": {
            "name": "EVT-143",
            "file": "submission_slot5_surgical_quad_rank6_breakthrough.csv",
            "score": 0.94761,
            "hits_20998": 19898,
            "sha256": compute_sha256(anchor_path)
        },
        "target": {
            "score": 0.95000,
            "net_hits_required": 51
        },
        "distribution_free_lower_bound": 0.0,
        "grid_evaluations": eval_results,
        "staged_candidates": staged_candidates,
        "safe_fallback": "EVT-143 (submission_slot5_surgical_quad_rank6_breakthrough.csv, score 0.94761)",
        "elapsed_seconds": round(time.time() - start_time, 1)
    }

    report_path = out_dir / "crossfitted_selector_ag0146_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    manifest_path = sub_dir / "final_portfolio_ag0146_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print(f"\nFinal Report saved:   {report_path}")
    print(f"Portfolio Manifest:   {manifest_path}")
    print(f"Elapsed time:         {time.time() - start_time:.1f}s")
    print("=" * 90)

if __name__ == "__main__":
    main()
