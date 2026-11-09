"""Tune High-Confidence Margin Curation on Corrected Dual-Regime Models.

Author: Antigravity
Date: 2026-11-09
Policy: Evaluate variance reduction through selective margin filtering to maximize P5 under prequential stress testing.
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import accuracy_score

ROOT = Path(".")
DATA_DIR = ROOT / "data/ap_2026_q_2/raw"
OUT_DIR = ROOT / "results/dual_regime_20261109"


def main():
    print("=" * 80)
    print("ANALYSIS OF HIGH-CONFIDENCE MARGIN CURATION FOR VARIANCE CONTRACTION")
    print("=" * 80, flush=True)

    # 1. Load dual-regime audit report and holdout split
    with open(OUT_DIR / "frozen_dual_regime_indices.json", "r") as f:
        idx_data = json.load(f)
    seen_ho_idx = np.array(idx_data["seen_holdout_idx"])
    unseen_ho_idx = np.array(idx_data["unseen_holdout_idx"])
    
    df_train = pd.read_json(DATA_DIR / "train_data.jsonlines", lines=True)
    y_train = (df_train["condition"] == "used").astype(np.int8).values
    
    df_seen_ho = df_train.iloc[seen_ho_idx].reset_index(drop=True)
    df_unseen_ho = df_train.iloc[unseen_ho_idx].reset_index(drop=True)
    y_seen_ho = y_train[seen_ho_idx]
    y_unseen_ho = y_train[unseen_ho_idx]
    y_holdout = np.concatenate([y_seen_ho, y_unseen_ho])
    sellers_holdout = np.concatenate([df_seen_ho["seller_id"].values, df_unseen_ho["seller_id"].values])
    
    # Run a fine grid of margins delta in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    # We will re-run the margin thresholding using the trained base and meta probabilities
    print(f"Loaded {len(y_holdout)} dual-regime holdout items.")
    print("Starting margin sensitivity analysis...")


if __name__ == "__main__":
    main()
