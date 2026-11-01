"""Export Paired Validation Extract for Independent Audit (CX-0045 Action 4).

Exports paired validation rows (tune and confirm only, with ground truth train labels only).
Never exposes test set labels or raw competition data.
"""
import gzip
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "results" / "cache_20261030"
OUT_FILE = ROOT / "results" / "intensive_20261030" / "paired_validation_extract_cx0045.csv.gz"

def main():
    sellers_raw = np.array([json.loads(line)["seller_id"] for line in open(ROOT / "data" / "ap_2026_q_2" / "raw" / "train_data.jsonlines")])
    ids_raw = np.array([json.loads(line)["id"] for line in open(ROOT / "data" / "ap_2026_q_2" / "raw" / "train_data.jsonlines")])

    records = []
    for seed in [2026, 314159]:
        with open(CACHE_DIR / f"split_features_A_hist_seed{seed}.pkl", "rb") as f:
            sp = pickle.load(f)

        th_proxy = float(sp["best_th_cb"])

        for role in ["tune", "confirm"]:
            idxs = sp[role]
            y_true = sp[f"y_{role}"]
            p_cb = sp[f"p_cb_{role if role == 'tune' else 'conf'}"]
            p_sp = sp[f"p_sp_{role if role == 'tune' else 'conf'}"]
            seen = sp[f"is_seen_{role}"]

            pred_proxy = (p_cb >= th_proxy).astype(int)

            # Candidate blend w=0.15 at th=0.48
            p_cand = 0.85 * p_cb + 0.15 * p_sp
            pred_cand = (p_cand >= 0.48).astype(int)

            is_fix = (pred_cand == y_true) & (pred_proxy != y_true)
            is_break = (pred_cand != y_true) & (pred_proxy == y_true)

            for i in range(len(idxs)):
                records.append({
                    "split_seed": seed,
                    "split_role": role,
                    "train_row_idx": int(idxs[i]),
                    "item_id": str(ids_raw[idxs[i]]),
                    "seller_id": str(sellers_raw[idxs[i]]),
                    "is_seen_seller": bool(seen[i]),
                    "y_true": int(y_true[i]),
                    "p_proxy": float(p_cb[i]),
                    "pred_proxy": int(pred_proxy[i]),
                    "p_cand": float(p_cand[i]),
                    "pred_cand": int(pred_cand[i]),
                    "is_fix": bool(is_fix[i]),
                    "is_break": bool(is_break[i]),
                })

    df = pd.DataFrame(records)
    df.to_csv(OUT_FILE, index=False, compression="gzip")

    h = hashlib.sha256()
    with open(OUT_FILE, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    sha = h.hexdigest()

    print(f"Exported {len(df):,} paired validation rows to: {OUT_FILE}")
    print(f"SHA-256: {sha}")

if __name__ == "__main__":
    main()
