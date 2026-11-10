#!/usr/bin/env python3
"""Canonical Nested Dual-Regime Splits Generator & Isolation Audit (Agent 1).

Implements Agent 1 mandates:
- Outer StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261110) by seller_id.
- Within each outer fold, partition outer_train into:
  * fit_set: training items used for feature and model fitting.
  * seen_val: holdout items from multi-item sellers present in fit_set (train_items ∩ seen_val_items = ∅).
  * outer_va_unseen: 100% unseen sellers (fit_items ∩ unseen_val_items = ∅, seller overlap = 0).
- Inner seller-group split on fit_set (85/15) solely for early stopping iteration freezing.
- Exhaustive isolation assertions verifying ZERO leakage between all sets.
- Full map DataFrame with id, seller_id, category_id, condition, outer_fold, role_in_fold_0..4.
- Produces versioned JSON manifest, Parquet, and CSV with SHA-256 hashes.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print("AGENT 1: CANONICAL NESTED DUAL-REGIME SPLITS GENERATOR & AUDIT")
    print("=" * 80)

    start_t = time.time()
    out_dir = Path("data/splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path("data/ap_2026_q_2/raw/train_data.jsonlines")
    assert train_path.exists(), f"Missing {train_path}!"
    train_hash = compute_sha256(train_path)
    print(f"Dataset train_data.jsonlines SHA-256: {train_hash}")

    print("Loading train_data.jsonlines...", flush=True)
    with open(train_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    n_items = len(rows)
    print(f"Total items: {n_items}", flush=True)

    ids = np.array([str(r.get("id")) for r in rows])
    sellers = np.array([str(r.get("seller_id") or "UNKNOWN") for r in rows])
    categories = np.array([str(r.get("category_id") or "UNKNOWN") for r in rows])
    targets = np.array([1 if r.get("condition") == "used" else 0 for r in rows], dtype=int)

    # Compute target hash
    target_hash = hashlib.sha256(targets.tobytes()).hexdigest()
    print(f"Target vector SHA-256: {target_hash} | Base used rate: {targets.mean():.4f}")

    unique_sellers = np.unique(sellers)
    seller_counts = Counter(sellers)
    n_unique_sellers = len(unique_sellers)
    multi_item_sellers_global = {s for s, c in seller_counts.items() if c >= 2}
    print(f"Unique sellers: {n_unique_sellers} | Multi-item sellers (>=2 items): {len(multi_item_sellers_global)}")

    # 1. Outer StratifiedGroupKFold on seller_id
    n_splits = 5
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=20261110)
    outer_unseen_fold = np.zeros(n_items, dtype=int)

    fold_roles = {f"role_in_fold_{f}": np.full(n_items, "unused", dtype=object) for f in range(n_splits)}
    fold_manifests = []

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(ids, targets, groups=sellers)):
        print(f"\n--- Constructing Fold {fold+1}/{n_splits} ---", flush=True)
        outer_unseen_fold[va_idx] = fold

        # Total train for this fold
        tr_sellers = sellers[tr_idx]
        tr_counts = Counter(tr_sellers)
        multi_sellers_in_tr = {s for s, c in tr_counts.items() if c >= 2}

        # Deterministic seen_val selection: pick 1 item per multi-item seller
        seen_picked = set()
        seen_val_indices = []
        fit_indices = []

        for idx in tr_idx:
            s = sellers[idx]
            if s in multi_sellers_in_tr and s not in seen_picked:
                seen_val_indices.append(idx)
                seen_picked.add(s)
            else:
                fit_indices.append(idx)

        seen_val_idx = np.array(seen_val_indices)
        fit_idx = np.array(fit_indices)

        # 2. Strict Isolation Audits
        # Check A: fit_idx and seen_val_idx do NOT share any items
        overlap_fit_seen = len(set(fit_idx) & set(seen_val_idx))
        assert overlap_fit_seen == 0, f"FATAL: Fold {fold+1} fit_idx and seen_val_idx share {overlap_fit_seen} items!"

        # Check B: fit_idx and va_idx do NOT share any items
        overlap_fit_va = len(set(fit_idx) & set(va_idx))
        assert overlap_fit_va == 0, f"FATAL: Fold {fold+1} fit_idx and va_idx share {overlap_fit_va} items!"

        # Check C: seen_val_idx and va_idx do NOT share any items
        overlap_seen_va = len(set(seen_val_idx) & set(va_idx))
        assert overlap_seen_va == 0, f"FATAL: Fold {fold+1} seen_val_idx and va_idx share {overlap_seen_va} items!"

        # Check D: Exactly ZERO seller overlap between outer-train (fit + seen_val) and outer-unseen-val (va_idx)
        seller_overlap_outer = len(set(sellers[tr_idx]) & set(sellers[va_idx]))
        assert seller_overlap_outer == 0, f"FATAL: Fold {fold+1} has seller overlap of {seller_overlap_outer}!"

        # Check E: 100% of sellers in seen_val_idx have items in fit_idx
        sellers_in_seen_val = set(sellers[seen_val_idx])
        sellers_in_fit = set(sellers[fit_idx])
        assert sellers_in_seen_val.issubset(sellers_in_fit), f"FATAL: Fold {fold+1} seen_val sellers not subset of fit sellers!"

        # 3. Inner seller-group split on fit_idx
        fit_sellers = sellers[fit_idx]
        fit_targets = targets[fit_idx]
        inner_sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42 + fold)
        in_tr_pos, in_va_pos = next(inner_sgkf.split(fit_idx, fit_targets, groups=fit_sellers))
        inner_tr_idx = fit_idx[in_tr_pos]
        inner_va_idx = fit_idx[in_va_pos]

        # Check F: Inner seller-group split has zero seller overlap
        inner_seller_overlap = len(set(sellers[inner_tr_idx]) & set(sellers[inner_va_idx]))
        assert inner_seller_overlap == 0, f"FATAL: Fold {fold+1} inner split seller overlap: {inner_seller_overlap}!"

        # Record roles
        fold_roles[f"role_in_fold_{fold}"][fit_idx] = "fit"
        fold_roles[f"role_in_fold_{fold}"][seen_val_idx] = "seen_val"
        fold_roles[f"role_in_fold_{fold}"][va_idx] = "unseen_val"

        print(f"Fold {fold+1}: TotalTrain={len(tr_idx)} -> Fit={len(fit_idx)} ({len(sellers_in_fit)} sellers), SeenVal={len(seen_val_idx)} ({len(sellers_in_seen_val)} sellers) | UnseenVal={len(va_idx)} ({len(set(sellers[va_idx]))} sellers)")
        print(f"Fold {fold+1} Inner Split: InnerTrain={len(inner_tr_idx)}, InnerVal={len(inner_va_idx)} | Zero Overlap Passed!")

        fold_manifests.append({
            "fold": fold + 1,
            "n_outer_train": len(tr_idx),
            "n_fit": len(fit_idx),
            "n_seen_val": len(seen_val_idx),
            "n_unseen_val": len(va_idx),
            "n_inner_train": len(inner_tr_idx),
            "n_inner_val": len(inner_va_idx),
            "fit_sellers_count": len(sellers_in_fit),
            "seen_val_sellers_count": len(sellers_in_seen_val),
            "unseen_val_sellers_count": len(set(sellers[va_idx])),
            "overlap_fit_seen_items": overlap_fit_seen,
            "overlap_fit_unseen_items": overlap_fit_va,
            "overlap_seen_unseen_items": overlap_seen_va,
            "seller_overlap_outer": seller_overlap_outer,
            "seller_overlap_inner": inner_seller_overlap,
        })

    # 4. Construct Master Splits DataFrame
    print("\nAssembling Master Splits DataFrame...", flush=True)
    df_splits = pd.DataFrame({
        "id": ids,
        "seller_id": sellers,
        "category_id": categories,
        "condition_used": targets,
        "outer_unseen_fold": outer_unseen_fold,
    })
    for k, v in fold_roles.items():
        df_splits[k] = v

    # 5. Save Artifacts
    csv_path = out_dir / "canonical_nested_dual_regime_splits.csv"
    parquet_path = out_dir / "canonical_nested_dual_regime_splits.parquet"
    manifest_path = out_dir / "canonical_nested_dual_regime_splits_manifest.json"

    df_splits.to_csv(csv_path, index=False)
    csv_hash = compute_sha256(csv_path)

    parquet_hash = None
    try:
        df_splits.to_parquet(parquet_path, index=False)
        parquet_hash = compute_sha256(parquet_path)
    except Exception as e:
        print(f"Parquet engine not available ({e}); saved canonical CSV.")

    manifest_data = {
        "title": "Canonical Nested Dual-Regime Splits Manifest (Agent 1)",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "random_state_outer": 20261110,
        "outer_splits": n_splits,
        "inner_splits": 6,
        "total_items": n_items,
        "train_data_jsonlines_sha256": train_hash,
        "target_vector_sha256": target_hash,
        "splits_csv_path": str(csv_path),
        "splits_csv_sha256": csv_hash,
        "splits_parquet_path": str(parquet_path) if parquet_hash else None,
        "splits_parquet_sha256": parquet_hash,
        "folds": fold_manifests,
        "isolation_audit": {
            "all_folds_zero_item_overlap": True,
            "all_folds_zero_outer_seller_overlap": True,
            "all_folds_zero_inner_seller_overlap": True,
            "all_seen_val_sellers_in_fit": True,
            "audit_verdict": "PASSED_STRICT_ISOLATION",
        }
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    manifest_hash = compute_sha256(manifest_path)
    print(f"\nSaved CSV: {csv_path} | SHA-256: {csv_hash}")
    if parquet_hash:
        print(f"Saved Parquet: {parquet_path} | SHA-256: {parquet_hash}")
    print(f"Saved Manifest: {manifest_path} | SHA-256: {manifest_hash}")
    print(f"AGENT 1 COMPLETED in {time.time() - start_t:.1f}s | STRICT ISOLATION PASSED!")


if __name__ == "__main__":
    main()
