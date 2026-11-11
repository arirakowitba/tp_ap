import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter
import re

def main():
    print("=== Deep Analysis of 4,017 Dual Blend Errors ===")
    oof_df = pd.read_csv("results/dual_blend_winner_20261111/oof_predictions_dual_blend_winner.csv")
    with open("data/ap_2026_q_2/raw/tr.pkl", "rb") as f:
        tr = pickle.load(f)

    y_true = oof_df["y_true"].values
    oof_dual = oof_df["oof_dual_blend_winner"].values
    pred_dual = oof_df["pred_dual_blend_winner"].values

    errors = (pred_dual != y_true)
    n_errors = int(errors.sum())
    print(f"Total errors: {n_errors} / 70000 ({n_errors/700:.2f}%)")

    # Direction of errors:
    # 0 = new, 1 = used
    # FP: true=0 (new), pred=1 (used) -> N2U error
    # FN: true=1 (used), pred=0 (new) -> U2N error
    fp_mask = (y_true == 0) & (pred_dual == 1)
    fn_mask = (y_true == 1) & (pred_dual == 0)
    n_fp = int(fp_mask.sum())
    n_fn = int(fn_mask.sum())
    print(f"False Positives (True=New, Pred=Used): {n_fp} ({n_fp/n_errors*100:.1f}%)")
    print(f"False Negatives (True=Used, Pred=New): {n_fn} ({n_fn/n_errors*100:.1f}%)")

    # Margin / Confidence breakdown:
    # Probability distances from 0.5
    margin = np.abs(oof_dual - 0.5)
    close_calls = (margin < 0.1) & errors
    high_conf = (margin > 0.3) & errors
    print(f"Close calls (prob in [0.4, 0.6]): {int(close_calls.sum())} ({close_calls.sum()/n_errors*100:.1f}%)")
    print(f"High confidence errors (prob < 0.2 or > 0.8): {int(high_conf.sum())} ({high_conf.sum()/n_errors*100:.1f}%)")

    # Seller counts
    seller_counts = tr["seller_id"].value_counts()
    tr_seller_n = tr["seller_id"].map(seller_counts).values
    error_seller_n = tr_seller_n[errors]

    singleton_errors = (error_seller_n == 1).sum()
    small_seller_errors = ((error_seller_n >= 2) & (error_seller_n <= 5)).sum()
    large_seller_errors = (error_seller_n > 5).sum()
    print(f"\nSeller volume distribution among errors:")
    print(f"  Singleton sellers (n=1 in train): {singleton_errors} ({singleton_errors/n_errors*100:.1f}%)")
    print(f"  Small sellers (2 <= n <= 5):     {small_seller_errors} ({small_seller_errors/n_errors*100:.1f}%)")
    print(f"  Large sellers (n > 5):            {large_seller_errors} ({large_seller_errors/n_errors*100:.1f}%)")

    # Category breakdown
    error_cats = tr.loc[errors, "category_id"].value_counts()
    print("\nTop 15 Error Categories:")
    for cat, cnt in error_cats.head(15).items():
        total_in_cat = (tr["category_id"] == cat).sum()
        acc_in_cat = 1.0 - (cnt / total_in_cat)
        print(f"  {cat}: {cnt} errors / {total_in_cat} total (Cat Acc: {acc_in_cat:.3f})")

    # Inspecting high error categories:
    # MLA1055, MLA1227, MLA1648, MLA31838
    print("\n=== Cluster Inspection ===")
    # Cluster 2: MLA31838 / MLA31763 (Collectibles / Coins)
    numis_cats = ["MLA31838", "MLA31763", "MLA1430"]
    numis_mask = tr["category_id"].isin(numis_cats) & errors
    print(f"\nCollectibles/Numismatics Errors ({numis_mask.sum()} items):")
    for idx in np.where(numis_mask)[0][:10]:
        t = tr.iloc[idx]["title"]
        yt = "used" if y_true[idx] == 1 else "new"
        p = oof_dual[idx]
        print(f"  [{yt.upper()}] p={p:.3f} | {t}")

    # Inspect Auto Spares in Official Stores
    auto_cats = [c for c in error_cats.head(20).index if "MLA" in c]
    # Check official stores
    off_store_errors = (~tr["official_store_id"].isna()) & errors
    print(f"\nOfficial Store Errors ({off_store_errors.sum()} items):")
    for idx in np.where(off_store_errors)[0][:10]:
        t = tr.iloc[idx]["title"]
        yt = "used" if y_true[idx] == 1 else "new"
        p = oof_dual[idx]
        cat = tr.iloc[idx]["category_id"]
        print(f"  [{yt.upper()}] p={p:.3f} | {cat} | {t}")

    # Inspect Close Calls ([0.45, 0.55])
    very_close = (np.abs(oof_dual - 0.5) < 0.05)
    print(f"\nVery Close Calls across ALL 70k: {very_close.sum()} items")
    vc_correct = ((pred_dual == y_true) & very_close).sum()
    vc_error = ((pred_dual != y_true) & very_close).sum()
    print(f"  Accuracy on very close calls: {vc_correct / very_close.sum():.4f} ({vc_correct} vs {vc_error})")

if __name__ == "__main__":
    main()
