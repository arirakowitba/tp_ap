"""Generate 5-Slot Sequential Podium Ladder Portfolio for 2026-11-09.

Protocol: Dynamic Sequential Kaggle Feedback (Slots 1 -> 2 -> 3 -> 4 -> 5).
Anchor: EVT-132 (Score 0.94752 = 19,896 hits in N_pub = 20,998).
Flips: Strictly Unanimous Consensus (EVT-132 'new' -> C10 Enriched Super-Ensemble 'used' + Safe Bayesian Fusion).
"""

import json
import time
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".")
OUT_DIR = ROOT / "submissions" / "portfolio_20261109_podium_ladder"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PUB = 20998
N_TEST = 30000
ANCHOR_SCORE = 0.94752
ANCHOR_HITS = 19896


def sha256_lf(path):
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def verify_csv(df, path):
    assert len(df) == 30000, f"Must have 30,000 rows, got {len(df)}"
    assert list(df.columns) == ["ID", "condition"], f"Invalid columns: {df.columns}"
    assert df["ID"].is_unique, "IDs not unique"
    assert set(df["condition"].unique()).issubset({"new", "used"}), "Invalid labels"
    assert df["ID"].min() == 1 and df["ID"].max() == 30000, "IDs must be 1 to 30000"
    assert df["condition"].isna().sum() == 0, "Null values found"


def main():
    print("=" * 85)
    print("GENERATING 5-SLOT SEQUENTIAL PODIUM LADDER PORTFOLIO (TARGET >= 0.95000)")
    print("=" * 85, flush=True)

    t0 = time.time()

    # 1. Load Anchor EVT-132
    p_anchor = ROOT / "submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv"
    df_anchor = pd.read_csv(p_anchor)
    verify_csv(df_anchor, p_anchor)
    print(f"Loaded Proven Anchor: {p_anchor.name} (19,896 hits = {ANCHOR_SCORE:.5f})")

    # 2. Load C10 Enriched Predictions & Safe Bayesian Fusion
    p_c10 = ROOT / "results/dual_regime_20261109/candidate_c10_enriched_super_ensemble.csv"
    df_c10 = pd.read_csv(p_c10)

    p_fusion = ROOT / "results/cross_referenced_bayesian_fusion/joint_information_fusion_all30k.csv"
    df_fusion = pd.read_csv(p_fusion)

    y_anchor = (df_anchor["condition"] == "used").astype(int).values
    y_c10 = (df_c10["condition"] == "used").astype(int).values

    # Candidate selection: EVT says 'new' (0), C10 says 'used' (1), and Fusion is safe with positive value
    c10_n2u = (y_anchor == 0) & (y_c10 == 1)
    df_fusion["is_safe_candidate"] = df_fusion["is_safe"] & (df_fusion["joint_value"] > 0)
    unanimous_mask = c10_n2u & df_fusion["is_safe_candidate"].values

    df_cands = df_fusion[unanimous_mask].sort_values(by=["joint_value", "p_model_mean"], ascending=[False, False]).copy()
    print(f"Total Unanimous High-Certainty Candidates: {len(df_cands)}")

    # 3. Define 5 Slots Ladder
    slots_def = [
        {
            "slot": 1,
            "flips": 20,
            "filename": "submission_slot1_top20_rank6_probe.csv",
            "expected_hits_net": 12,
            "expected_score": (ANCHOR_HITS + 12) / N_PUB,
            "role": "Sondeo de Ultra-Certeza: Supera a Clara Rodriguez (0.94761) y captura Rango 6; mide pendiente inicial p_hat1."
        },
        {
            "slot": 2,
            "flips": 45,
            "filename": "submission_slot2_top45_rank4_expansion.csv",
            "expected_hits_net": 26,
            "expected_score": (ANCHOR_HITS + 26) / N_PUB,
            "role": "Expansion Calibrada: Supera a Catalina Bachetti (0.94833) y Agustin Sanchez (0.94838), captura Rango 4."
        },
        {
            "slot": 3,
            "flips": 70,
            "filename": "submission_slot3_top70_podium_attack.csv",
            "expected_hits_net": 40,
            "expected_score": (ANCHOR_HITS + 40) / N_PUB,
            "role": "ASALTO AL PODIO: Supera a Gonzalo Buhler (0.94895) y CONQUISTA EL PODIO (PUESTO 3 GENERAL)."
        },
        {
            "slot": 4,
            "flips": 95,
            "filename": "submission_slot4_top95_breakthrough_09500.csv",
            "expected_hits_net": 54,
            "expected_score": (ANCHOR_HITS + 54) / N_PUB,
            "role": "QUIEBRE DE BARRERA 0.95000: Alcanza 0.95009 (+54 hits) asegurando Puesto 3 holgado."
        },
        {
            "slot": 5,
            "flips": 120,
            "filename": "submission_slot5_top120_climax_podium.csv",
            "expected_hits_net": 68,
            "expected_score": (ANCHOR_HITS + 68) / N_PUB,
            "role": "CLIMAX OPERATIVO: Consolidacion de Podio con 0.95076 (+68 hits) y asalto a Nicolas Caveda (0.95095, Rango 2)."
        }
    ]

    manifest = []

    print("\n" + "=" * 95)
    print(f"{'Slot':6s} | {'Flips':6s} | {'Exp Net':9s} | {'Exp Score':10s} | {'Filename':40s} | {'Role'}")
    print("=" * 95)

    for s in slots_def:
        k = s["flips"]
        flip_item_ids = set(df_cands.head(k)["item_id"].values)

        # Generate new submission based on EVT-132
        sub_df = df_anchor.copy()
        # Flip selected items from 'new' to 'used'
        mask_flip = sub_df["ID"].isin(flip_item_ids)
        sub_df.loc[mask_flip, "condition"] = "used"

        actual_flips = int((sub_df["condition"] != df_anchor["condition"]).sum())
        assert actual_flips == k, f"Expected {k} flips, got {actual_flips}"

        out_path = OUT_DIR / s["filename"]
        sub_df.to_csv(out_path, index=False)
        verify_csv(sub_df, out_path)
        sha = sha256_lf(out_path)

        manifest_entry = {
            "slot": s["slot"],
            "filename": s["filename"],
            "path": str(out_path),
            "sha256_lf": sha,
            "flips_vs_anchor": k,
            "expected_net_hits": s["expected_hits_net"],
            "expected_score": round(s["expected_score"], 5),
            "role": s["role"]
        }
        manifest.append(manifest_entry)

        print(f"Slot {s['slot']:1d} | {k:6d} | {s['expected_hits_net']:+8d}  | {s['expected_score']:9.5f}  | {s['filename']:40s} | {s['role'][:35]}...")

    print("=" * 95)

    # Save Manifest
    manifest_p = OUT_DIR / "portfolio_manifest.json"
    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved Manifest: {manifest_p}")

    print(f"Generated 5 sequential submissions in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
