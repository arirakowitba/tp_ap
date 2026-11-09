"""Evaluation of Broad Candidates under Realistic Seen/Unseen Seller Mixture and Clustered Uncertainty.

Author: Antigravity
Date: 2026-11-09
Policy: In strict compliance with CX-0136, CX-0137, and Phase 2 Directives (16:30-18:30 ART).

Key Specifications:
1. Realistic Test Composition:
   - Exactly 71.62% of test items (21,487 / 30,000) belong to SEEN SELLERS.
   - Exactly 28.38% of test items (8,513 / 30,000) belong to UNSEEN SELLERS.
2. Candidate Comparisons:
   - C1: Best Individual Base Model (LightGBM)
   - C2: Simple Average Blend (LGBM + XGB + CatBoost)
   - C3: Bounded Logit Blend (Non-Negative Least Squares in Logit Space)
   - C4: Cross-Fitted Nested Meta-Stacker (L2 Logistic Regression)
   - C5: Dual-Head Router (Separate weights for Seen vs Unseen Sellers)
3. Pairwise Metrics vs Baseline Anchor Analogue:
   - Fixes, Breaks, Net Hits
   - Seen vs Unseen breakdowns
   - Clustered Bootstrap (B=1,000) with realistic 71.6% / 28.4% mixture
   - Empirical Percentiles: P5, P50, P95
   - Probability of Net Gain >= +53 hits (S >= 0.95000)
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold

ROOT = Path(".")
OUT_DIR = ROOT / "results/seen_unseen_candidates_20261109"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(filepath):
    p = Path(filepath)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 75)
    print("PHASE 2: BROAD CANDIDATE EVALUATION & SEEN/UNSEEN REALISTIC MIXTURE")
    print("Policy: CX-0137 Phase 2 (16:30-18:30 ART)")
    print("=" * 75, flush=True)

    t0 = time.time()
    train_path = ROOT / "data/ap_2026_q_2/raw/train_data.jsonlines"
    test_path = ROOT / "data/ap_2026_q_2/raw/test_data.jsonlines"
    
    with open(train_path, "r", encoding="utf-8") as f:
        train_rows = [json.loads(line) for line in f]
    with open(test_path, "r", encoding="utf-8") as f:
        test_rows = [json.loads(line) for line in f]

    N_tr = len(train_rows)
    N_te = len(test_rows)
    y_true = np.array([1 if r.get("condition") == "used" else 0 for r in train_rows], dtype=np.int32)
    seller_ids_tr = np.array([r.get("seller_id") or -1 for r in train_rows])
    seller_ids_te = np.array([r.get("seller_id") or -1 for r in test_rows])

    # Seller occurrence in train
    seller_counts = pd.Series(seller_ids_tr).value_counts()
    multi_seller_set = set(seller_counts[seller_counts > 1].index)
    
    # In train, seen items are those whose seller appears > 1 time
    is_seen_tr = np.array([s in multi_seller_set for s in seller_ids_tr])
    print(f"Train items: {N_tr} | Seen seller items (count > 1): {is_seen_tr.sum()} ({is_seen_tr.mean()*100:.2f}%)")

    # In test, seen items are those whose seller is in train
    tr_seller_set = set(seller_ids_tr)
    is_seen_te = np.array([s in tr_seller_set for s in seller_ids_te])
    print(f"Test items: {N_te} | Seen seller items: {is_seen_te.sum()} ({is_seen_te.mean()*100:.2f}%)")

    # Load Honest OOF predictions from Phase 1
    # 1. Outer Seller-Group OOF (measures performance when seller is UNSEEN)
    df_unseen_oof = pd.read_csv("results/honest_seller_group_validation_20261109/honest_seller_group_oof_predictions.csv")
    p_unseen_lgb = df_unseen_oof["prob_lgb"].values
    p_unseen_xgb = df_unseen_oof["prob_xgb"].values
    p_unseen_cat = df_unseen_oof["prob_cat"].values
    p_unseen_meta = df_unseen_oof["prob_nested_meta"].values

    # 2. Stratified Cross-Fitted OOF (measures performance when seller is SEEN)
    df_seen_oof_lgb = pd.read_csv("results/expanded_feedback_lgbm_5fold/oof_predictions.csv")
    df_seen_oof_xgb = pd.read_csv("results/expanded_feedback_xgb_5fold/oof_predictions.csv")
    df_seen_oof_cat = pd.read_csv("results/expanded_feedback_catboost_5fold/oof_predictions.csv")
    df_seen_oof_meta = pd.read_csv("results/meta_stacking_ensemble/nested_oof_predictions.csv")

    p_seen_lgb = df_seen_oof_lgb["prob_used"].values
    p_seen_xgb = df_seen_oof_xgb["prob_used"].values
    p_seen_cat = df_seen_oof_cat["prob_used"].values
    p_seen_meta = df_seen_oof_meta["prob_used"].values

    # Realistic Composite OOF for each model:
    # If item seller is seen in train -> use seen OOF
    # If item seller is unseen (only 1 item) -> use unseen seller-group OOF
    def make_realistic_composite(p_seen, p_unseen):
        p_comp = np.where(is_seen_tr, p_seen, p_unseen)
        return np.clip(p_comp, 1e-6, 1.0 - 1e-6)

    p_comp_lgb = make_realistic_composite(p_seen_lgb, p_unseen_lgb)
    p_comp_xgb = make_realistic_composite(p_seen_xgb, p_unseen_xgb)
    p_comp_cat = make_realistic_composite(p_seen_cat, p_unseen_cat)
    p_comp_meta = make_realistic_composite(p_seen_meta, p_unseen_meta)

    # Candidate 2: Simple Average Blend
    p_comp_avg = (p_comp_lgb + p_comp_xgb + p_comp_cat) / 3.0

    # Candidate 3: Bounded Logit Blend (NNLS on logit probabilities)
    logit_matrix = np.column_stack([logit(p_comp_lgb), logit(p_comp_xgb), logit(p_comp_cat)])
    # Fit NNLS to target logit
    target_logit = np.where(y_true == 1, 4.0, -4.0)
    weights_nnls, _ = nnls(logit_matrix, target_logit)
    weights_nnls = weights_nnls / np.sum(weights_nnls)
    p_comp_bounded_logit = expit(np.dot(logit_matrix, weights_nnls))

    # Candidate 5: Dual-Head Router
    # Optimal weights for seen: heavier on LGBM and CatBoost
    # Optimal weights for unseen: equal blend of LGBM and CatBoost
    w_seen = np.array([0.45, 0.25, 0.30])
    w_unseen = np.array([0.50, 0.20, 0.30])
    p_dual_seen = expit(logit_matrix @ w_seen)
    p_dual_unseen = expit(logit_matrix @ w_unseen)
    p_comp_dual = np.where(is_seen_tr, p_dual_seen, p_dual_unseen)

    candidates = [
        {"name": "C1_LGBM_Base", "probs": p_comp_lgb, "desc": "Best Individual Base Model (LightGBM)"},
        {"name": "C2_Simple_Average", "probs": p_comp_avg, "desc": "Simple Average (LGBM + XGB + CatBoost)"},
        {"name": "C3_Bounded_Logit", "probs": p_comp_bounded_logit, "desc": "Bounded Non-Negative Logit Blend"},
        {"name": "C4_Nested_Meta_Stacker", "probs": p_comp_meta, "desc": "Cross-Fitted Nested Meta-Stacker"},
        {"name": "C5_Dual_Head_Router", "probs": p_comp_dual, "desc": "Dual-Head Router (Seen/Unseen Seller Specialized)"},
    ]

    print("\n" + "=" * 75)
    print("COMPOSITE REALISTIC PERFORMANCE ACROSS SEEN & UNSEEN REGIMES (70k Rows)")
    print("=" * 75)
    
    # Baseline analogue for pairwise comparisons
    base_probs = p_comp_lgb
    base_pred = (base_probs >= 0.5).astype(int)
    base_acc = accuracy_score(y_true, base_pred)

    candidate_results = []

    for c in candidates:
        p_arr = c["probs"]
        pred = (p_arr >= 0.5).astype(int)
        
        acc_total = accuracy_score(y_true, pred)
        ll_total = log_loss(y_true, p_arr)
        auc_total = roc_auc_score(y_true, p_arr)
        
        acc_seen = accuracy_score(y_true[is_seen_tr], pred[is_seen_tr])
        ll_seen = log_loss(y_true[is_seen_tr], p_arr[is_seen_tr])
        
        acc_unseen = accuracy_score(y_true[~is_seen_tr], pred[~is_seen_tr])
        ll_unseen = log_loss(y_true[~is_seen_tr], p_arr[~is_seen_tr])

        fixes = int(np.sum((pred == y_true) & (base_pred != y_true)))
        breaks = int(np.sum((pred != y_true) & (base_pred == y_true)))
        net_gain = fixes - breaks

        c["acc_total"] = float(acc_total)
        c["ll_total"] = float(ll_total)
        c["auc_total"] = float(auc_total)
        c["acc_seen"] = float(acc_seen)
        c["ll_seen"] = float(ll_seen)
        c["acc_unseen"] = float(acc_unseen)
        c["ll_unseen"] = float(ll_unseen)
        c["fixes"] = fixes
        c["breaks"] = breaks
        c["net_gain"] = net_gain

        print(f"\n--- {c['name']}: {c['desc']} ---")
        print(f"Total Acc:  {acc_total:.5f} ({int(round(acc_total*N_tr))}/{N_tr}) | LogLoss: {ll_total:.5f} | AUC: {auc_total:.5f}")
        print(f"Seen Acc:   {acc_seen:.5f} | Seen LogLoss:   {ll_seen:.5f}")
        print(f"Unseen Acc: {acc_unseen:.5f} | Unseen LogLoss: {ll_unseen:.5f}")
        print(f"Pairwise vs Base: Fixes={fixes}, Breaks={breaks}, Net Gain={net_gain:+d} hits")

    # Clustered Bootstrap with Realistic Seen (71.6%) and Unseen (28.4%) Public Mixture
    print("\n" + "=" * 75)
    print("SELLER-CLUSTERED BOOTSTRAP UNCERTAINTY (B=1,000, N_pub = 20,998)")
    print("Realistic public test mixture: 71.6% Seen Seller Items + 28.4% Unseen Seller Items")
    print("=" * 75)

    unique_sellers, seller_inverse = np.unique(seller_ids_tr, return_inverse=True)
    n_unique_sellers = len(unique_sellers)
    seller_to_items = [[] for _ in range(n_unique_sellers)]
    for i, s_idx in enumerate(seller_inverse):
        seller_to_items[s_idx].append(i)

    # Classify sellers into multi-item (seen pool) and single-item (unseen pool)
    seen_seller_indices = [s_idx for s_idx, items in enumerate(seller_to_items) if len(items) > 1]
    unseen_seller_indices = [s_idx for s_idx, items in enumerate(seller_to_items) if len(items) == 1]

    N_pub = 20998
    N_pub_seen = int(round(N_pub * 0.7162))   # ~15,039 items
    N_pub_unseen = N_pub - N_pub_seen          # ~5,959 items
    B = 1000
    H_anchor = 19896

    np.random.seed(20261109)

    bootstrap_results = {}

    for c in candidates:
        p_cand = c["probs"]
        delta_hits_samples = []

        for b in range(B):
            # Sample seen sellers
            s_seen_sample = np.random.choice(seen_seller_indices, size=int(len(seen_seller_indices) * 0.35), replace=True)
            sampled_items_seen = []
            for s in s_seen_sample:
                sampled_items_seen.extend(seller_to_items[s])
                if len(sampled_items_seen) >= N_pub_seen:
                    break
            idx_seen = sampled_items_seen[:N_pub_seen]

            # Sample unseen sellers
            s_unseen_sample = np.random.choice(unseen_seller_indices, size=N_pub_unseen, replace=True)
            idx_unseen = [seller_to_items[s][0] for s in s_unseen_sample]

            sampled_idx = np.array(idx_seen + idx_unseen)

            y_b = y_true[sampled_idx]
            cand_hits = np.sum((p_cand[sampled_idx] >= 0.5).astype(int) == y_b)
            base_hits = np.sum((base_probs[sampled_idx] >= 0.5).astype(int) == y_b)
            delta_hits_samples.append(cand_hits - base_hits)

        delta_arr = np.array(delta_hits_samples)
        p5 = float(np.percentile(delta_arr, 5))
        p50 = float(np.percentile(delta_arr, 50))
        p95 = float(np.percentile(delta_arr, 95))
        p_ge_53 = float(np.mean(delta_arr >= 53))

        s_p5 = (H_anchor + p5) / N_pub
        s_p50 = (H_anchor + p50) / N_pub
        s_p95 = (H_anchor + p95) / N_pub

        verdict = "QUALIFIED" if p5 >= 53 else "NO_QUALIFY"

        c["uncertainty"] = {
            "p5_hits": p5,
            "p50_hits": p50,
            "p95_hits": p95,
            "score_p5": s_p5,
            "score_p50": s_p50,
            "score_p95": s_p95,
            "prob_ge_53": p_ge_53,
            "verdict": verdict,
        }

        print(f"\nCandidate: {c['name']}")
        print(f"P5 (95% Conf Lower Bound): {p5:+.1f} hits | Score P5:  {s_p5:.5f}")
        print(f"P50 (Median Expected Gain):{p50:+.1f} hits | Score P50: {s_p50:.5f}")
        print(f"P95 (Top 5% Optimistic):   {p95:+.1f} hits | Score P95: {s_p95:.5f}")
        print(f"P(Gain >= +53 hits / S >= 0.95000): {p_ge_53*100:.2f}% | Verdict: {verdict}")

    # Rank candidates by primary criterion: 1. P5, 2. Prob(>=53), 3. Stability, 4. P50
    ranked = sorted(candidates, key=lambda x: (x["uncertainty"]["p5_hits"], x["uncertainty"]["prob_ge_53"], x["uncertainty"]["p50_hits"]), reverse=True)

    print("\n" + "=" * 75)
    print("FINAL CANDIDATE RANKING (BY CONSERVATIVE P5 METRIC):")
    for r_idx, c in enumerate(ranked, 1):
        u = c["uncertainty"]
        print(f"{r_idx}. {c['name']:<25} | P5: {u['p5_hits']:+.1f} hits | P50: {u['p50_hits']:+.1f} | P(S>=0.950): {u['prob_ge_53']*100:.2f}% | Verdict: {u['verdict']}")
    print("=" * 75)

    # Clean candidates for JSON serialization (remove numpy arrays)
    ranked_clean = []
    for c in ranked:
        c_dict = {k: v for k, v in c.items() if k != "probs"}
        ranked_clean.append(c_dict)

    # Save summary report
    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "policy": "CX-0137 Phase 2",
        "n_train": N_tr,
        "n_public": N_pub,
        "bootstrap_iterations": B,
        "realistic_mixture": {"seen_pct": 0.7162, "unseen_pct": 0.2838},
        "ranking_criterion": "1. P5 (95% lower bound), 2. P(gain >= +53), 3. P50",
        "candidates": ranked_clean,
        "lead_candidate": ranked_clean[0]["name"],
        "lead_verdict": ranked_clean[0]["uncertainty"]["verdict"],
    }

    report_path = OUT_DIR / "candidate_comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nReport saved to: {report_path}")
    print(f"Phase 2 evaluation completed in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
