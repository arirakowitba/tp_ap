"""Canonical 3-Way Test-Representative Split Generator for AP-2026-Q-2.

Partitions 70,000 items into:
1. `fit`: Primary training partition (~40,881 rows)
2. `tune`: Development / hyperparameter tuning partition (~15,044 rows: ~58% cold new sellers, ~42% bracketed seen sellers)
3. `confirm`: Held-out confirmation partition (~14,075 rows: ~56% cold new sellers, ~44% bracketed seen sellers)

Guarantees:
- Zero seller overlap between cold sellers (tune or confirm) and fit.
- Zero seller overlap between cold tune sellers and cold confirm sellers.
- 100% causal bracketing for seen sellers in tune and confirm:
  every seen evaluation item is strictly bracketed between items of the SAME seller in fit:
  fit_min_time(seller) < start_time < fit_max_time(seller).
- Immutable SHA-256 digests and provenance manifest.
"""
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "splits" / "t98_parsed_train.pkl"
OUT_CSV = ROOT / "data" / "splits" / "canonical_3way_splits.csv"
OUT_MANIFEST = ROOT / "data" / "splits" / "canonical_3way_manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_3way_split():
    with open(DATA_PATH, "rb") as f:
        df = pickle.load(f)

    assert len(df) == 70000 and df["id"].is_unique

    rng = np.random.default_rng(20261113)
    sellers = np.sort(df["seller_id"].unique())
    rng.shuffle(sellers)

    # 1. Cold sellers: 24% total (12% tune, 12% confirm)
    n_cold = int(len(sellers) * 0.24)
    cold_tune_sellers = set(sellers[: n_cold // 2])
    cold_confirm_sellers = set(sellers[n_cold // 2 : n_cold])

    cold_tune = df["seller_id"].isin(cold_tune_sellers).to_numpy()
    cold_confirm = df["seller_id"].isin(cold_confirm_sellers).to_numpy()
    cold = cold_tune | cold_confirm

    # 2. Warm sellers: multi-item sellers with >= 4 distinct timestamps, not cold
    unique_times = df.groupby("seller_id")["start_time"].nunique()
    eligible = df["seller_id"].map(unique_times).ge(4).to_numpy() & ~cold
    order = df.groupby("seller_id")["start_time"].rank(method="dense").to_numpy()
    upper = df["seller_id"].map(unique_times).to_numpy()

    # Intermediate items strictly bracketed by earliest and latest timestamps
    bracketed = eligible & (order > 1) & (order < upper)

    # Assign intermediate bracketed items: 26% tune, 26% confirm, remainder stays in fit
    u = rng.random(len(df))
    warm_tune = bracketed & (u < 0.26)
    warm_confirm = bracketed & (u >= 0.26) & (u < 0.52)

    tune_mask = cold_tune | warm_tune
    confirm_mask = cold_confirm | warm_confirm
    fit_mask = ~tune_mask & ~confirm_mask

    fit_idx = np.flatnonzero(fit_mask)
    tune_idx = np.flatnonzero(tune_mask)
    confirm_idx = np.flatnonzero(confirm_mask)

    # Assertions
    assert len(fit_idx) + len(tune_idx) + len(confirm_idx) == 70000
    assert not (set(df.loc[cold_tune, "seller_id"]) & set(df.loc[fit_mask, "seller_id"])), "Leak: tune cold in fit!"
    assert not (set(df.loc[cold_confirm, "seller_id"]) & set(df.loc[fit_mask, "seller_id"])), "Leak: confirm cold in fit!"
    assert not (set(df.loc[cold_tune, "seller_id"]) & set(df.loc[cold_confirm, "seller_id"])), "Leak: tune cold in confirm cold!"

    # Bracketing check against fit
    fit_times = df.iloc[fit_mask].groupby("seller_id")["start_time"].agg(["min", "max"])
    for name, mask in [("tune_seen", warm_tune), ("confirm_seen", warm_confirm)]:
        st = df["start_time"].iloc[mask].to_numpy()
        links = fit_times.reindex(df["seller_id"].iloc[mask])
        assert ((links["min"] < st) & (links["max"] > st)).all(), f"{name} not strictly bracketed by fit items!"

    # Split digest
    split_digest = hashlib.sha256(np.r_[fit_idx, tune_idx, confirm_idx].tobytes()).hexdigest()

    # Build DataFrame
    roles = np.full(len(df), "fit", dtype=object)
    roles[tune_idx] = "tune"
    roles[confirm_idx] = "confirm"

    regimes = np.full(len(df), "fit", dtype=object)
    regimes[cold_tune] = "tune_cold_new"
    regimes[warm_tune] = "tune_seen_bracketed"
    regimes[cold_confirm] = "confirm_cold_new"
    regimes[warm_confirm] = "confirm_seen_bracketed"

    split_df = pd.DataFrame({
        "row": df["row"],
        "id": df["id"],
        "seller_id": df["seller_id"],
        "category_id": df["category_id"],
        "start_time": df["start_time"],
        "y": df["y"],
        "role": roles,
        "regime": regimes,
        "is_cold_tune": cold_tune,
        "is_warm_tune": warm_tune,
        "is_cold_confirm": cold_confirm,
        "is_warm_confirm": warm_confirm,
    })

    split_df.to_csv(OUT_CSV, index=False)

    # Check overlap with previous frente3 split if it exists
    overlap_info = {}
    frente3_csv = ROOT / "data" / "splits" / "frente3_test_representative_split.csv"
    if frente3_csv.exists():
        f3_df = pd.read_csv(frente3_csv)
        f3_val_ids = set(f3_df[f3_df["role"] == "val"]["id"])
        tune_ids = set(split_df[split_df["role"] == "tune"]["id"])
        confirm_ids = set(split_df[split_df["role"] == "confirm"]["id"])
        overlap_info["overlap_frente3_val_tune"] = len(f3_val_ids & tune_ids)
        overlap_info["overlap_frente3_val_confirm"] = len(f3_val_ids & confirm_ids)
        overlap_info["overlap_frente3_val_fit"] = len(f3_val_ids & set(split_df[split_df["role"] == "fit"]["id"]))

    manifest = {
        "generator": "scripts/build_3way_canonical_split.py",
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "data_path": str(DATA_PATH.relative_to(ROOT)),
        "data_sha256": sha256_file(DATA_PATH),
        "split_sha256": split_digest,
        "csv_sha256": sha256_file(OUT_CSV),
        "fit_rows": int(len(fit_idx)),
        "tune_rows": int(len(tune_idx)),
        "tune_cold_rows": int(cold_tune.sum()),
        "tune_seen_bracketed_rows": int(warm_tune.sum()),
        "confirm_rows": int(len(confirm_idx)),
        "confirm_cold_rows": int(cold_confirm.sum()),
        "confirm_seen_bracketed_rows": int(warm_confirm.sum()),
        "overlap_with_previous": overlap_info,
        "bracketing_guarantee": "100% strictly bracketed: fit_min < start_time < fit_max",
        "cold_disjointness_guarantee": "0 seller overlap with fit, 0 overlap between tune and confirm",
        "public_probability_claims": "NO_IDENTIFIABLE",
    }

    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Canonical 3-way split built successfully!")
    print(f"Fit: {len(fit_idx)} rows")
    print(f"Tune: {len(tune_idx)} rows (Cold: {cold_tune.sum()}, Seen bracketed: {warm_tune.sum()})")
    print(f"Confirm: {len(confirm_idx)} rows (Cold: {cold_confirm.sum()}, Seen bracketed: {warm_confirm.sum()})")
    print(f"Split digest SHA-256: {split_digest}")
    print(f"Manifest written to: {OUT_MANIFEST}")


if __name__ == "__main__":
    build_3way_split()
