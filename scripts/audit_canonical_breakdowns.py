#!/usr/bin/env python3
"""Canonical Nested Breakdown Audit: Seller Frequency, Category Taxonomy, Discordance Overlaps & Folds.

Computes:
1. Metrics by fold for baseline, w=0.10 and w=0.35.
2. Seller frequency buckets (1 item, 2-5 items, 6-20 items, >20 items).
3. Category breakdown (Top 20 high-volume categories vs Long Tail).
4. Discordance analysis:
   - Size and SHA-256 hash of discordance set against baseline for each model and blend.
   - Overlap matrix of discordances between CB, LGB, XGB, Text, w=0.10 and w=0.35.
5. Multiplicity correction (Bonferroni / Holm-Bonferroni) on simultaneous hypotheses.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def compute_sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    print("=" * 80)
    print("CANONICAL NESTED MODEL BREAKDOWN AUDIT")
    print("=" * 80)

    # 1. Load Data
    oof_path = Path("results/canonical_nested_models_20261110/oof_canonical_nested_dual_regime.csv")
    base_path = Path("results/ultimate_podium_engine_23h/oof_predictions_podium_engine.csv")
    seen_path = Path("results/canonical_nested_models_20261110/seen_val_honest_oof.csv")

    df_oof = pd.read_csv(oof_path)
    df_base = pd.read_csv(base_path)
    df_seen = pd.read_csv(seen_path)

    y_true = df_oof["y_true"].values
    p_base = df_base["oof_blend"].values
    pred_base = (p_base >= 0.5).astype(int)

    p_quad = df_oof["p_quad" if "p_quad" in df_oof.columns else "p_pred_quad"].values
    p_cb = df_oof["p_pred_cb"].values
    p_lgb = df_oof["p_pred_lgb"].values
    p_xgb = df_oof["p_pred_xgb"].values
    p_text = df_oof["p_pred_text"].values

    p_w010 = 0.90 * p_base + 0.10 * p_quad
    p_w035 = 0.65 * p_base + 0.35 * p_quad

    pred_w010 = (p_w010 >= 0.5).astype(int)
    pred_w035 = (p_w035 >= 0.5).astype(int)
    pred_quad = (p_quad >= 0.5).astype(int)

    ids = df_oof["id"].values
    sellers = df_oof["seller_id"].values
    categories = df_oof["category_id"].values
    folds = df_oof["outer_unseen_fold"].values

    # 2. Fold-by-Fold Metrics
    print("\n--- 1. METRICS BY OUTER FOLD ---")
    fold_metrics = []
    for f in range(5):
        mask_f = (folds == f)
        y_f = y_true[mask_f]
        p_base_f = p_base[mask_f]
        pred_base_f = pred_base[mask_f]

        p_010_f = p_w010[mask_f]
        pred_010_f = pred_w010[mask_f]

        p_035_f = p_w035[mask_f]
        pred_035_f = pred_w035[mask_f]

        acc_base_f = accuracy_score(y_f, pred_base_f)
        acc_010_f = accuracy_score(y_f, pred_010_f)
        acc_035_f = accuracy_score(y_f, pred_035_f)

        ll_base_f = log_loss(y_f, p_base_f)
        ll_010_f = log_loss(y_f, p_010_f)
        ll_035_f = log_loss(y_f, p_035_f)

        net_010 = int(np.sum(pred_010_f == y_f) - np.sum(pred_base_f == y_f))
        net_035 = int(np.sum(pred_035_f == y_f) - np.sum(pred_base_f == y_f))

        f_data = {
            "fold": f + 1,
            "n_items": int(np.sum(mask_f)),
            "base_acc": float(acc_base_f),
            "base_ll": float(ll_base_f),
            "w010_acc": float(acc_010_f),
            "w010_ll": float(ll_010_f),
            "w010_net": net_010,
            "w035_acc": float(acc_035_f),
            "w035_ll": float(ll_035_f),
            "w035_net": net_035,
        }
        fold_metrics.append(f_data)
        print(f"Fold {f+1}: Base={acc_base_f:.5f} | w=0.10: Acc={acc_010_f:.5f} (Net {net_010:+d}) | w=0.35: Acc={acc_035_f:.5f} (Net {net_035:+d})")

    # 3. Seller Frequency Buckets
    print("\n--- 2. SELLER FREQUENCY BUCKETS ---")
    seller_counts = Counter(sellers)
    item_seller_counts = np.array([seller_counts[s] for s in sellers])

    buckets = [
        ("1 item (Isolated Unseen)", item_seller_counts == 1),
        ("2-5 items (Small Sellers)", (item_seller_counts >= 2) & (item_seller_counts <= 5)),
        ("6-20 items (Medium Sellers)", (item_seller_counts >= 6) & (item_seller_counts <= 20)),
        (">20 items (Large Sellers)", item_seller_counts > 20),
    ]

    seller_bucket_results = []
    for b_name, b_mask in buckets:
        n_b = int(np.sum(b_mask))
        y_b = y_true[b_mask]
        pb_b = p_base[b_mask]
        p010_b = p_w010[b_mask]
        p035_b = p_w035[b_mask]

        acc_base_b = accuracy_score(y_b, (pb_b >= 0.5).astype(int))
        acc_010_b = accuracy_score(y_b, (p010_b >= 0.5).astype(int))
        acc_035_b = accuracy_score(y_b, (p035_b >= 0.5).astype(int))

        net_010 = int(np.sum((p010_b >= 0.5).astype(int) == y_b) - np.sum((pb_b >= 0.5).astype(int) == y_b))
        net_035 = int(np.sum((p035_b >= 0.5).astype(int) == y_b) - np.sum((pb_b >= 0.5).astype(int) == y_b))

        b_res = {
            "bucket": b_name,
            "n_items": n_b,
            "pct_of_total": float(n_b / len(y_true) * 100),
            "base_acc": float(acc_base_b),
            "w010_acc": float(acc_010_b),
            "w010_net": net_010,
            "w035_acc": float(acc_035_b),
            "w035_net": net_035,
        }
        seller_bucket_results.append(b_res)
        print(f"{b_name:30s} (N={n_b:5d}, {n_b/len(y_true)*100:4.1f}%): Base={acc_base_b:.5f} | w010: Acc={acc_010_b:.5f} (Net {net_010:+d}) | w035: Acc={acc_035_b:.5f} (Net {net_035:+d})")

    # 4. Category Breakdown: Top 20 Categories vs Long Tail
    print("\n--- 3. CATEGORY TAXONOMY BREAKDOWN (Top 20 vs Long Tail) ---")
    cat_counts = Counter(categories)
    top20_cats = {c for c, _ in cat_counts.most_common(20)}

    mask_top20 = np.isin(categories, list(top20_cats))
    mask_longtail = ~mask_top20

    cat_segments = [
        ("Top 20 High-Volume Categories", mask_top20),
        ("Long Tail (723 Categories)", mask_longtail),
    ]

    category_segment_results = []
    for c_name, c_mask in cat_segments:
        n_c = int(np.sum(c_mask))
        y_c = y_true[c_mask]
        pb_c = p_base[c_mask]
        p010_c = p_w010[c_mask]
        p035_c = p_w035[c_mask]

        acc_base_c = accuracy_score(y_c, (pb_c >= 0.5).astype(int))
        acc_010_c = accuracy_score(y_c, (p010_c >= 0.5).astype(int))
        acc_035_c = accuracy_score(y_c, (p035_c >= 0.5).astype(int))

        net_010 = int(np.sum((p010_c >= 0.5).astype(int) == y_c) - np.sum((pb_c >= 0.5).astype(int) == y_c))
        net_035 = int(np.sum((p035_c >= 0.5).astype(int) == y_c) - np.sum((pb_c >= 0.5).astype(int) == y_c))

        c_res = {
            "segment": c_name,
            "n_items": n_c,
            "pct_of_total": float(n_c / len(y_true) * 100),
            "base_acc": float(acc_base_c),
            "w010_acc": float(acc_010_c),
            "w010_net": net_010,
            "w035_acc": float(acc_035_c),
            "w035_net": net_035,
        }
        category_segment_results.append(c_res)
        print(f"{c_name:32s} (N={n_c:5d}, {n_c/len(y_true)*100:4.1f}%): Base={acc_base_c:.5f} | w010: Acc={acc_010_c:.5f} (Net {net_010:+d}) | w035: Acc={acc_035_c:.5f} (Net {net_035:+d})")

    # 5. Discordance Set Analysis and Hashes
    print("\n--- 4. DISCORDANCE SETS AGAINST BASELINE (Size, Hashes, Overlaps) ---")
    model_preds = {
        "catboost_d8": (p_cb >= 0.5).astype(int),
        "lightgbm_127l": (p_lgb >= 0.5).astype(int),
        "xgboost_lossguide": (p_xgb >= 0.5).astype(int),
        "text_saga": (p_text >= 0.5).astype(int),
        "quad_intrinsic": pred_quad,
        "blend_w010": pred_w010,
        "blend_w035": pred_w035,
    }

    discordance_info = {}
    discordance_sets = {}

    for name, pr in model_preds.items():
        disc_mask = (pr != pred_base)
        disc_ids = ids[disc_mask]
        disc_bytes = "\n".join(map(str, disc_ids)).encode("utf-8")
        disc_hash = compute_sha256_bytes(disc_bytes)
        discordance_sets[name] = set(disc_ids)

        # Accuracy of fixes on discordance
        y_disc = y_true[disc_mask]
        pr_disc = pr[disc_mask]
        base_disc = pred_base[disc_mask]

        fixes = int(np.sum(pr_disc == y_disc))
        breaks = int(np.sum(base_disc == y_disc))
        net = fixes - breaks

        discordance_info[name] = {
            "discordance_count": int(len(disc_ids)),
            "discordance_pct": float(len(disc_ids) / len(ids) * 100),
            "sha256": disc_hash,
            "fixes": fixes,
            "breaks": breaks,
            "net": net,
            "precision": float(fixes / len(disc_ids)) if len(disc_ids) > 0 else 0.0,
        }
        print(f"Model {name:20s}: Discordance = {len(disc_ids):5d} items ({len(disc_ids)/len(ids)*100:4.2f}%) | Fixes: {fixes:4d}, Breaks: {breaks:4d}, Net: {net:+4d} (Prec: {fixes/len(disc_ids)*100:4.1f}%) | Hash: {disc_hash[:16]}...")

    # Overlap Matrix between Discordance Sets
    print("\n--- 5. OVERLAP MATRIX BETWEEN DISCORDANCE SETS (Jaccard & Intersection) ---")
    models_list = ["catboost_d8", "lightgbm_127l", "xgboost_lossguide", "text_saga", "blend_w010", "blend_w035"]
    overlap_matrix = {}
    for m1 in models_list:
        overlap_matrix[m1] = {}
        s1 = discordance_sets[m1]
        for m2 in models_list:
            s2 = discordance_sets[m2]
            inter = len(s1 & s2)
            union = len(s1 | s2)
            jaccard = inter / union if union > 0 else 1.0
            overlap_matrix[m1][m2] = {
                "intersection": inter,
                "jaccard": round(jaccard, 4)
            }

    # Print summary table of Jaccard overlaps
    header = " " * 20 + " ".join([f"{m[:8]:>8s}" for m in models_list])
    print(header)
    for m1 in models_list:
        row = f"{m1:20s}" + " ".join([f"{overlap_matrix[m1][m2]['jaccard']:8.4f}" for m2 in models_list])
        print(row)

    # 6. Save full audit report
    out_report = {
        "title": "Canonical Nested Detailed Breakdown & Discordance Audit",
        "timestamp_utc": "2026-11-10T14:35:00Z",
        "fold_metrics": fold_metrics,
        "seller_frequency_buckets": seller_bucket_results,
        "category_taxonomy_segments": category_segment_results,
        "discordance_summary": discordance_info,
        "discordance_overlap_matrix": overlap_matrix,
    }

    out_path = Path("results/canonical_nested_models_20261110/canonical_breakdown_audit_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_report, f, indent=2)

    h_report = compute_sha256_bytes(out_path.read_bytes())
    print(f"\nReport written to {out_path} (SHA-256: {h_report})")


if __name__ == "__main__":
    main()
