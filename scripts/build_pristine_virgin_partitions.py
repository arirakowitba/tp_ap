"""Build pristine 3-way partitions (Fit, Tune, Virgin Confirm) with zero leakage.

Specifications:
- Total items: 70,000
- Fit Pool: 46,000 items
- Tune Pool: 12,000 items (dual regime: seen/unseen)
- Virgin Confirm: 12,000 items (dual regime: 71.62% seen, 28.38% unseen from strictly disjoint sellers)
- Strict zero-leakage, reproducible seed 20261109.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".")
DATA_DIR = ROOT / "data/ap_2026_q_2/raw"
OUT_DIR = ROOT / "results/pristine_audit_20261109"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80)
    print("BUILDING PRISTINE 3-WAY PARTITIONS (FIT, TUNE, VIRGIN CONFIRM)")
    print("=" * 80)

    df_train = pd.read_json(DATA_DIR / "train_data.jsonlines", lines=True)
    N = len(df_train)
    print(f"Total training items: {N}")

    seller_counts = df_train["seller_id"].value_counts()
    single_sellers = np.array(seller_counts[seller_counts == 1].index, copy=True)
    multi_sellers = np.array(seller_counts[seller_counts > 1].index, copy=True)

    print(f"Total sellers: {len(seller_counts)} | Single-item sellers: {len(single_sellers)} | Multi-item sellers: {len(multi_sellers)}")

    rng = np.random.RandomState(20261109)
    rng.shuffle(single_sellers)
    rng.shuffle(multi_sellers)

    # 1. Allocate strictly disjoint sellers for Unseen Confirm (approx 3,406 items = 28.38% of 12,000)
    target_unseen_confirm = int(12000 * 0.2838)  # 3,406 items
    unseen_confirm_sellers = single_sellers[:target_unseen_confirm]
    remaining_single_sellers = single_sellers[target_unseen_confirm:]

    # Items for Unseen Confirm
    unseen_confirm_mask = df_train["seller_id"].isin(unseen_confirm_sellers)
    unseen_confirm_idx = np.where(unseen_confirm_mask)[0]

    # 2. Allocate strictly disjoint sellers for Unseen Tune (approx 3,406 items = 28.38% of 12,000)
    unseen_tune_sellers = remaining_single_sellers[:target_unseen_confirm]
    remaining_single_sellers = remaining_single_sellers[target_unseen_confirm:]

    unseen_tune_mask = df_train["seller_id"].isin(unseen_tune_sellers)
    unseen_tune_idx = np.where(unseen_tune_mask)[0]

    print(f"Virgin Unseen Confirm items: {len(unseen_confirm_idx)} (Sellers: {len(unseen_confirm_sellers)})")
    print(f"Tune Unseen items: {len(unseen_tune_idx)} (Sellers: {len(unseen_tune_sellers)})")

    # 3. For seen items: partition multi-seller items across fit, tune, and confirm
    # Target seen confirm: 12,000 - 3,406 = 8,594 items
    # Target seen tune: 12,000 - 3,406 = 8,594 items
    # Target seen fit: Remaining items
    seen_mask = ~unseen_confirm_mask & ~unseen_tune_mask
    seen_df = df_train[seen_mask].copy()

    # Stratified split by seller and target
    seen_indices = np.where(seen_mask)[0]
    seen_sellers = df_train.iloc[seen_indices]["seller_id"].values
    seen_targets = (df_train.iloc[seen_indices]["condition"] == "used").astype(int).values

    # Assign items of multi-item sellers
    # To ensure repeat sellers exist in fit:
    # We take 8,594 items for confirm, 8,594 items for tune, rest for fit
    # Ensuring every seller in seen_confirm has at least one item in fit!
    fit_seen_list = []
    tune_seen_list = []
    confirm_seen_list = []

    seller_to_items = {}
    for idx, s in zip(seen_indices, seen_sellers):
        seller_to_items.setdefault(s, []).append(idx)

    for s, items in seller_to_items.items():
        if len(items) == 1:
            fit_seen_list.extend(items)
        else:
            # Multi-item seller: guarantee at least 1 in fit!
            rng.shuffle(items)
            fit_seen_list.append(items[0])
            for item in items[1:]:
                # distribute remainder to confirm and tune up to targets
                if len(confirm_seen_list) < 8594:
                    confirm_seen_list.append(item)
                elif len(tune_seen_list) < 8594:
                    tune_seen_list.append(item)
                else:
                    fit_seen_list.append(item)

    seen_confirm_idx = np.array(confirm_seen_list)
    seen_tune_idx = np.array(tune_seen_list)
    fit_idx = np.array(fit_seen_list)

    # Combine Seen + Unseen
    virgin_confirm_idx = np.concatenate([seen_confirm_idx, unseen_confirm_idx])
    tune_idx = np.concatenate([seen_tune_idx, unseen_tune_idx])

    # Verification of zero overlap
    assert len(set(fit_idx) & set(tune_idx)) == 0, "Leak between Fit and Tune!"
    assert len(set(fit_idx) & set(virgin_confirm_idx)) == 0, "Leak between Fit and Virgin Confirm!"
    assert len(set(tune_idx) & set(virgin_confirm_idx)) == 0, "Leak between Tune and Virgin Confirm!"

    # Verification of unseen seller disjointness
    fit_sellers = set(df_train.iloc[fit_idx]["seller_id"])
    unseen_confirm_sellers_actual = set(df_train.iloc[unseen_confirm_idx]["seller_id"])
    assert len(fit_sellers & unseen_confirm_sellers_actual) == 0, "Unseen confirm sellers leak into Fit!"

    print("\n--- PARTITION SUMMARY ---")
    print(f"Fit Pool Size:           {len(fit_idx):6d} items ({len(fit_idx)/N*100:5.2f}%)")
    print(f"Tune Pool Size:          {len(tune_idx):6d} items ({len(tune_idx)/N*100:5.2f}%) [Seen: {len(seen_tune_idx)}, Unseen: {len(unseen_tune_idx)}]")
    print(f"Virgin Confirm Size:     {len(virgin_confirm_idx):6d} items ({len(virgin_confirm_idx)/N*100:5.2f}%) [Seen: {len(seen_confirm_idx)}, Unseen: {len(unseen_confirm_idx)}]")
    print(f"Total:                   {len(fit_idx) + len(tune_idx) + len(virgin_confirm_idx):6d} items")
    print(f"Seen ratio in Confirm:   {len(seen_confirm_idx) / len(virgin_confirm_idx) * 100:.2f}% (Target: 71.62%)")
    print("ALL LEAKAGE CHECKS PASSED (0 item overlap, 0 unseen seller overlap).")

    # Save to JSON
    out_p = OUT_DIR / "pristine_partitions.json"
    with open(out_p, "w") as f:
        json.dump({
            "seed": 20261109,
            "fit_idx": fit_idx.tolist(),
            "tune_idx": tune_idx.tolist(),
            "virgin_confirm_idx": virgin_confirm_idx.tolist(),
            "seen_confirm_idx": seen_confirm_idx.tolist(),
            "unseen_confirm_idx": unseen_confirm_idx.tolist(),
            "seen_tune_idx": seen_tune_idx.tolist(),
            "unseen_tune_idx": unseen_tune_idx.tolist(),
        }, f)
    print(f"\nSaved pristine partitions to: {out_p}")


if __name__ == "__main__":
    main()
