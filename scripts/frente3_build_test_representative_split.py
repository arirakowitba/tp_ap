"""Frente 3: Test-Representative Canonical Split Generator for AP-2026-Q-2.

Reproduces the frozen canonical interleaved split matching CX-0185 / late_ap_nov13_report.json:
- Fit rows: 44,054
- Validation rows: 25,946
  - Cold (new sellers, zero overlap with fit): 12,844 rows
  - Seen (strictly bracketed between fit listings of the same seller): 13,102 rows
- Provenance hash of np.r_[fit, val].tobytes(): 9c7a6a9a1305ec3f281f5ac8a07a1f9f9410bdaef6dc101aa56f51b6393e51a5
"""
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "splits" / "t98_parsed_train.pkl"
OUT_CSV = ROOT / "data" / "splits" / "frente3_test_representative_split.csv"
OUT_MANIFEST = ROOT / "data" / "splits" / "frente3_manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_split():
    with open(DATA_PATH, "rb") as f:
        df = pickle.load(f)

    assert len(df) == 70000 and df["id"].is_unique

    rng = np.random.default_rng(28111322)
    sellers = np.sort(df["seller_id"].unique())
    rng.shuffle(sellers)

    # 1. Cold sellers: 1/6 of sellers randomly selected
    cold_sellers = set(sellers[: len(sellers) // 6])
    cold = df["seller_id"].isin(cold_sellers).to_numpy()

    # 2. Warm sellers: multi-item sellers with >= 3 distinct start times, not cold
    unique_times = df.groupby("seller_id")["start_time"].nunique()
    eligible = df["seller_id"].map(unique_times).ge(3).to_numpy() & ~cold
    order = df.groupby("seller_id")["start_time"].rank(method="dense").to_numpy()
    upper = df["seller_id"].map(unique_times).to_numpy()
    warm = eligible & (order > 1) & (order < upper) & (rng.random(len(df)) < 0.50)

    held = cold | warm
    fit = np.flatnonzero(~held)
    val = np.flatnonzero(held)

    # Integrity assertions
    assert not (set(df["seller_id"].iloc[cold]) & set(df["seller_id"].iloc[fit])), "Cold sellers leak into fit!"
    fit_times = df.iloc[fit].groupby("seller_id")["start_time"].agg(["min", "max"])
    seen = warm[val]
    stamp = df["start_time"].iloc[val[seen]].to_numpy()
    links = fit_times.reindex(df["seller_id"].iloc[val[seen]])
    assert ((links["min"] < stamp) & (links["max"] > stamp)).all(), "Seen validation item is not strictly bracketed!"

    split_digest = hashlib.sha256(np.r_[fit, val].tobytes()).hexdigest()
    expected_digest = "9c7a6a9a1305ec3f281f5ac8a07a1f9f9410bdaef6dc101aa56f51b6393e51a5"
    assert split_digest == expected_digest, f"Split digest {split_digest} != expected {expected_digest}"

    # Build DataFrame
    roles = np.full(len(df), "fit", dtype=object)
    roles[val] = "val"

    regimes = np.full(len(df), "fit", dtype=object)
    regimes[val[~seen]] = "new_seller"
    regimes[val[seen]] = "seen_bracketed"

    split_df = pd.DataFrame({
        "row": df["row"],
        "id": df["id"],
        "seller_id": df["seller_id"],
        "category_id": df["category_id"],
        "start_time": df["start_time"],
        "y": df["y"],
        "role": roles,
        "regime": regimes,
        "is_cold": cold,
        "is_bracketed": warm,
    })

    split_df.to_csv(OUT_CSV, index=False)

    manifest = {
        "generator": "scripts/frente3_build_test_representative_split.py",
        "data_path": str(DATA_PATH.relative_to(ROOT)),
        "data_sha256": sha256_file(DATA_PATH),
        "split_sha256": split_digest,
        "csv_sha256": sha256_file(OUT_CSV),
        "fit_rows": int(len(fit)),
        "val_rows": int(len(val)),
        "val_seen_bracketed_rows": int(seen.sum()),
        "val_cold_new_rows": int((~seen).sum()),
        "status": "FROZEN_PRE_DECLARED",
    }

    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Frente 3 Split generated and verified successfully:")
    print(f"  Fit: {len(fit)} rows")
    print(f"  Val: {len(val)} rows (Seen bracketed: {seen.sum()}, Cold new: {(~seen).sum()})")
    print(f"  Split SHA-256: {split_digest}")
    print(f"  Manifest written to: {OUT_MANIFEST}")


if __name__ == "__main__":
    build_split()
