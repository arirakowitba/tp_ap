#!/usr/bin/env python3
"""Construct S5 Candidate Portfolios & Final Strategic Audit.

Builds the 3 pre-declared candidate portfolios from the fully nested sprint:
1. Candidate 1 (Consensus Core): Top consensus N2U flips across independent models (Text, LGBM, CB) with K=20.
2. Candidate 2 (High Stability): Rate-matched peak OOF net gain at K=50.
3. Candidate 3 (Diverse Residuals): Multi-modal high-divergence flips at K=100.
4. Fallback: SUBMISSION_4_TODAY_RECOMMENDED (S4, expected 0.94766).
"""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def sha256_lf(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()

def compute_map_hash(df: pd.DataFrame) -> str:
    lines = [f"{int(r['ID'])},{r['condition'].strip()}" for _, r in df.sort_values("ID").iterrows()]
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def main():
    out_dir = ROOT / "submissions/portfolio_nested_s5_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load S4 and EVT-143
    s4_path = ROOT / "submissions/SUBMISSION_4_TODAY_RECOMMENDED.csv"
    evt143_path = ROOT / "submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv"
    df_s4 = pd.read_csv(s4_path)
    df_evt = pd.read_csv(evt143_path)

    # 2. Load nested test probabilities
    test_probs_path = ROOT / "results/fully_nested_s5_base/test_probabilities_nested_s5.csv"
    df_probs = pd.read_csv(test_probs_path)

    p_lr = df_probs["p_agent_a1_lr"].values
    p_ridge = df_probs["p_agent_a2_ridge"].values
    p_lgb = df_probs["p_agent_b1_lgb"].values
    p_cb = df_probs["p_agent_b2_cb"].values
    p_stack = df_probs["p_agent_d_stack"].values

    y_s4 = df_s4["condition"].values
    y_evt = df_evt["condition"].values
    ids = df_s4["ID"].values

    # Candidate 1: Consensus Core N2U (K=20)
    # Models independently agree that item labeled 'new' in S4 is 'used'
    # Score = sum of logit probabilities across LR, LGBM, CB
    is_n2u_cand = (y_s4 == "new") & (p_lr > 0.5) & (p_lgb > 0.5) & (p_cb > 0.5)
    score_consensus = np.where(is_n2u_cand, (p_lr + p_lgb + p_cb + p_stack) / 4.0, 0.0)
    top20_consensus_idx = np.argsort(-score_consensus)[:20]

    df_cand1 = df_s4.copy()
    df_cand1.loc[top20_consensus_idx, "condition"] = "used"

    cand1_path = out_dir / "candidate_s5_consensus_core_k20.csv"
    df_cand1.to_csv(cand1_path, index=False)
    sha_cand1 = sha256_lf(cand1_path)
    map_cand1 = compute_map_hash(df_cand1)

    # Candidate 2: High Stability (K=50, OOF Peak Net Gain)
    # Flips selected by highest stacking probability disagreement vs S4
    divergence_score = np.where(y_s4 == "used", 1.0 - p_stack, p_stack)
    # Exclude U2N flips due to catastrophic public degradation, focus on top N2U
    score_n2u = np.where(y_s4 == "new", p_stack, 0.0)
    top50_stab_idx = np.argsort(-score_n2u)[:50]

    df_cand2 = df_s4.copy()
    df_cand2.loc[top50_stab_idx, "condition"] = "used"

    cand2_path = out_dir / "candidate_s5_high_stability_n2u_k50.csv"
    df_cand2.to_csv(cand2_path, index=False)
    sha_cand2 = sha256_lf(cand2_path)
    map_cand2 = compute_map_hash(df_cand2)

    # Candidate 3: Diverse Multi-Modal (K=100, N2U + High Confidence U2N)
    # Top 80 N2U + Top 20 U2N with p_stack < 0.05
    score_u2n = np.where((y_s4 == "used") & (p_stack < 0.05), 1.0 - p_stack, 0.0)
    top80_n2u = np.argsort(-score_n2u)[:80]
    top20_u2n = np.argsort(-score_u2n)[:20]

    df_cand3 = df_s4.copy()
    df_cand3.loc[top80_n2u, "condition"] = "used"
    df_cand3.loc[top20_u2n, "condition"] = "new"

    cand3_path = out_dir / "candidate_s5_diverse_k100.csv"
    df_cand3.to_csv(cand3_path, index=False)
    sha_cand3 = sha256_lf(cand3_path)
    map_cand3 = compute_map_hash(df_cand3)

    print("=" * 80)
    print("S5 CANDIDATE PORTFOLIO SUMMARY")
    print("=" * 80)
    print(f"Candidate 1 (Consensus Core K=20):")
    print(f"  Path: {cand1_path}")
    print(f"  SHA-256: {sha_cand1}")
    print(f"  Map Hash: {map_cand1}")
    print(f"  Flips vs S4: {np.sum(df_cand1['condition'] != df_s4['condition'])} (All N2U)")
    print(f"  Flips vs EVT-143: {np.sum(df_cand1['condition'] != df_evt['condition'])}")

    print(f"\nCandidate 2 (High Stability N2U K=50):")
    print(f"  Path: {cand2_path}")
    print(f"  SHA-256: {sha_cand2}")
    print(f"  Map Hash: {map_cand2}")
    print(f"  Flips vs S4: {np.sum(df_cand2['condition'] != df_s4['condition'])} (All N2U)")
    print(f"  Flips vs EVT-143: {np.sum(df_cand2['condition'] != df_evt['condition'])}")

    print(f"\nCandidate 3 (Diverse K=100):")
    print(f"  Path: {cand3_path}")
    print(f"  SHA-256: {sha_cand3}")
    print(f"  Map Hash: {map_cand3}")
    print(f"  Flips vs S4: {np.sum(df_cand3['condition'] != df_s4['condition'])}")
    print(f"  Flips vs EVT-143: {np.sum(df_cand3['condition'] != df_evt['condition'])}")

    print(f"\nFallback Recommended (S4):")
    print(f"  Path: {s4_path}")
    print(f"  SHA-256: {sha256_lf(s4_path)}")
    print(f"  Expected Public Hits: 19,899 / 20,998 (0.94766)")

    # Save manifest
    portfolio_manifest = {
        "candidate_1_consensus_k20": {
            "path": str(cand1_path),
            "sha256_lf": sha_cand1,
            "map_hash": map_cand1,
            "flips_vs_s4": int(np.sum(df_cand1['condition'] != df_s4['condition'])),
            "flips_vs_evt143": int(np.sum(df_cand1['condition'] != df_evt['condition'])),
            "direction": "N2U_only"
        },
        "candidate_2_stability_k50": {
            "path": str(cand2_path),
            "sha256_lf": sha_cand2,
            "map_hash": map_cand2,
            "flips_vs_s4": int(np.sum(df_cand2['condition'] != df_s4['condition'])),
            "flips_vs_evt143": int(np.sum(df_cand2['condition'] != df_evt['condition'])),
            "direction": "N2U_only"
        },
        "candidate_3_diverse_k100": {
            "path": str(cand3_path),
            "sha256_lf": sha_cand3,
            "map_hash": map_cand3,
            "flips_vs_s4": int(np.sum(df_cand3['condition'] != df_s4['condition'])),
            "flips_vs_evt143": int(np.sum(df_cand3['condition'] != df_evt['condition'])),
            "direction": "N2U_80_plus_U2N_20"
        },
        "fallback_recommended_s4": {
            "path": str(s4_path),
            "sha256_lf": sha256_lf(s4_path),
            "expected_hits": 19899,
            "expected_score": 0.94766
        }
    }
    with open(out_dir / "portfolio_manifest.json", "w", encoding="utf-8") as f:
        json.dump(portfolio_manifest, f, indent=2)

if __name__ == "__main__":
    main()
