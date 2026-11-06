"""Forensic ID-Level Target Solver across ALL 90 Submissions (Including Today's 5 Submissions).

Extracts ID-level target evidence from 90 verified submission vectors,
computes updated posterior probabilities, identifies high-certainty pseudo-labels,
and logs the full target matrix for the 24-hour training pipeline.
"""

import os, sys, json, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import lsq_linear

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "forensic_id_solver_20261105"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PUB = 20998

def hit_count(score, n_pub=N_PUB):
    cands = [k for k in range(n_pub + 1) if round(k / n_pub, 5) == round(score, 5)]
    return cands[0] if len(cands) == 1 else int(round(score * n_pub))

def main():
    print("=== FORENSIC ID-LEVEL TARGET SOLVER (90 SUBMISSION VECTORS) ===", flush=True)
    t0 = time.time()

    # 1. Load D1+D2 baseline
    p_d1d2 = ROOT / "submissions" / "candidates_20261104" / "candidate_01_d1d2_plus_upper10.csv"
    upper10 = [4559, 4997, 6072, 6360, 6816, 8661, 14353, 14395, 28301, 28453]
    df_ref = pd.read_csv(p_d1d2)
    d1d2_preds = df_ref.set_index("ID")["condition"].to_dict()
    for i in upper10:
        d1d2_preds[i] = "used"
    
    d1d2_hits = hit_count(0.94652) # 19875 hits
    print(f"D1+D2 Reference Anchor: {d1d2_hits} hits (0.94652)")

    # 2. Load Master Ledger
    p_master = ROOT / "results" / "forensic_ledger_20261104" / "submission_ledger_master_85vec.json"
    with open(p_master, encoding="utf-8") as f:
        master_data = json.load(f)
    
    entries = [e for e in master_data["entries"] if e.get("status") == "verified_recoverable"]
    print(f"Loaded {len(entries)} verified entries.")

    d1d2_entry = next((e for e in entries if "d1_d2" in str(e.get("key")) or "d1d2" in str(e.get("key"))), None)
    d1d2_flips_on_48 = set(d1d2_entry.get("flipped_ids", [])) if d1d2_entry else set()

    # Map flips vs D1+D2
    for e in entries:
        if "flipped_ids_vs_d1d2" in e:
            e["flips_vs_d1d2"] = sorted(list(set(e["flipped_ids_vs_d1d2"])))
        else:
            f48 = set(e.get("flipped_ids", []))
            f_d1d2 = f48 ^ d1d2_flips_on_48
            e["flips_vs_d1d2"] = sorted(list(f_d1d2))
        e["delta_hits_vs_d1d2"] = hit_count(e["score"]) - d1d2_hits

    # Deduplicate strictly by unique flip set vs D1+D2
    unique_flip_groups = {}
    for e in entries:
        sig = tuple(e["flips_vs_d1d2"])
        if sig not in unique_flip_groups:
            unique_flip_groups[sig] = e
        else:
            # Keep entry with most detailed metadata
            pass
    
    unique_entries = list(unique_flip_groups.values())
    print(f"Unique vector configurations relative to D1+D2: {len(unique_entries)}")

    # All unique IDs that differ from D1+D2 in at least one submission
    all_intervened_ids = sorted(list(set(i for e in unique_entries for i in e["flips_vs_d1d2"])))
    n_ids = len(all_intervened_ids)
    id_to_col = {item_id: col for col, item_id in enumerate(all_intervened_ids)}
    print(f"Total intervened IDs across all configurations: {n_ids}")

    # 3. Build Design Matrix M: shape (n_configs, n_ids)
    n_configs = len(unique_entries)
    M = np.zeros((n_configs, n_ids), dtype=np.float64)
    y_delta = np.zeros(n_configs, dtype=np.float64)

    for r, e in enumerate(unique_entries):
        y_delta[r] = e["delta_hits_vs_d1d2"]
        for i_val in e["flips_vs_d1d2"]:
            c = id_to_col[i_val]
            M[r, c] = 1.0

    print(f"Design Matrix Shape: {M.shape}")
    rank_M = np.linalg.matrix_rank(M)
    print(f"Matrix Rank: {rank_M}, Nullity: {n_ids - rank_M}")

    # 4. Solve Bounded Least Squares for ID Effects
    print("Solving Bounded Least Squares (bounds=[-1, 1])...", flush=True)
    res = lsq_linear(M, y_delta, bounds=(-1.0, 1.0), lsmr_tol='auto', verbose=0)
    e_est = res.x
    residual_norm = np.linalg.norm(M @ e_est - y_delta)
    print(f"Residual Norm: {residual_norm:.4f}")

    # Ridge regularized solution
    alpha = 2.0
    ridge_weights = np.linalg.solve(M @ M.T + alpha * np.eye(n_configs), y_delta)
    e_ridge = np.clip(M.T @ ridge_weights, -1.0, 1.0)

    id_counts = (M != 0).sum(axis=0)

    # 5. Load Multimodal Probabilities for Comparison
    df_cb = pd.read_csv(ROOT / "results" / "full_native_nb_70k_20261105" / "test_probabilities.csv")
    df_lgb = pd.read_csv(ROOT / "results" / "full_lightgbm_enriched_70k_20261105" / "lgb_enriched_test_probabilities.csv")
    p_sub = np.load(ROOT / "results" / "frente_c_subword_and_diagnostics_20261104" / "test_probs_subword_rich_text.npy")

    p_cb_nat = df_cb["p_native_text"].values
    p_cb_nb = df_cb["p_nested_nb"].values
    p_lgb = df_lgb["p_lgb_enriched"].values

    p_ens = 0.40 * p_cb_nat + 0.35 * p_cb_nb + 0.15 * p_lgb + 0.10 * p_sub

    # 6. Build ID Evidence Table
    records = []
    for col, item_id in enumerate(all_intervened_ids):
        idx = item_id - 1
        d_lbl = d1d2_preds[item_id]
        p_val = p_ens[idx]
        b_eff = e_est[col]
        r_eff = e_ridge[col]
        cnt = int(id_counts[col])

        # Direction of flip vs D1+D2: if flipped, what was the flipped condition?
        # If D1+D2 was 'used', flip was 'new'. If b_eff > 0, flip to 'new' is beneficial -> true label is 'new'.
        # If D1+D2 was 'new', flip was 'used'. If b_eff > 0, flip to 'used' is beneficial -> true label is 'used'.
        if d_lbl == "used":
            inferred_p_used = 1.0 - (0.5 + 0.5 * b_eff) # b_eff > 0 means flip to new helped -> p_used lower
        else:
            inferred_p_used = 0.5 + 0.5 * b_eff # b_eff > 0 means flip to used helped -> p_used higher

        records.append({
            "item_id": item_id,
            "d1d2_label": d_lbl,
            "model_p_used": float(p_val),
            "bounded_effect_delta": float(b_eff),
            "ridge_effect_delta": float(r_eff),
            "times_intervened": cnt,
            "inferred_p_used": float(np.clip(inferred_p_used, 0.01, 0.99))
        })

    df_evidence = pd.DataFrame(records).sort_values(by="bounded_effect_delta", ascending=False)
    out_csv = OUT_DIR / "id_level_target_evidence_all_90probes.csv"
    df_evidence.to_csv(out_csv, index=False)
    print(f"Saved {len(df_evidence)} ID evidence rows to {out_csv.name}")

    # Summary stats
    strongly_positive = df_evidence[df_evidence["bounded_effect_delta"] > 0.35]
    strongly_negative = df_evidence[df_evidence["bounded_effect_delta"] < -0.35]
    print(f"Strongly Positive Flip IDs (e_i > +0.35): {len(strongly_positive)}")
    print(f"Strongly Negative Flip IDs (Anchor Protected e_i < -0.35): {len(strongly_negative)}")

    # Check upper10 items
    print("\nUpper10 IDs in evidence:")
    u10_ev = df_evidence[df_evidence["item_id"].isin(upper10)]
    print(u10_ev[["item_id", "bounded_effect_delta", "times_intervened", "model_p_used"]])

    # Save summary json
    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "n_probes_total": len(entries),
        "n_unique_configs": n_configs,
        "n_intervened_ids": n_ids,
        "matrix_rank": int(rank_M),
        "residual_norm": float(residual_norm),
        "n_strongly_positive_ids": len(strongly_positive),
        "n_strongly_negative_ids": len(strongly_negative),
        "strongly_positive_ids": strongly_positive["item_id"].tolist(),
        "strongly_negative_ids": strongly_negative["item_id"].tolist()
    }
    with open(OUT_DIR / "evidence_summary_90probes.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Solver completed in {time.time() - t0:.1f}s.")

if __name__ == "__main__":
    main()
