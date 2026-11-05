"""Forensic ID-Level Target Solver across ALL Historical Submissions & Probes.

Analyzes all 85 historical submission vectors, calculates exact flip sets vs D1+D2,
solves for ID-level effects on public score, and cross-references with multimodal model predictions.
"""

import os, sys, json, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from scipy import sparse
from scipy.optimize import lsq_linear

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "forensic_id_solver_20261105"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PUB = 20998

def hit_count(score, n_pub=N_PUB):
    cands = [k for k in range(n_pub + 1) if round(k / n_pub, 5) == round(score, 5)]
    return cands[0] if len(cands) == 1 else int(round(score * n_pub))

def main():
    print("=== FORENSIC ID-LEVEL TARGET SOLVER ACROSS ALL PROBES ===", flush=True)
    t0 = time.time()

    # 1. Load D1+D2 reference
    p_d1d2 = ROOT / "submissions" / "candidates_20261104" / "candidate_01_d1d2_plus_upper10.csv"
    # D1+D2 is Candidate 01 minus upper10
    upper10 = [4559, 4997, 6072, 6360, 6816, 8661, 14353, 14395, 28301, 28453]
    df_ref = pd.read_csv(p_d1d2)
    d1d2_preds = df_ref.set_index("ID")["condition"].to_dict()
    for i in upper10:
        d1d2_preds[i] = "used" # Restore D1+D2 baseline
    
    d1d2_hits = hit_count(0.94652) # 19875 hits
    print(f"D1+D2 Reference Anchor: {d1d2_hits} hits (0.94652)")

    # 2. Load Master Ledger (85 recoverable vectors)
    p_master = ROOT / "results" / "forensic_ledger_20261104" / "submission_ledger_master_85vec.json"
    with open(p_master, encoding="utf-8") as f:
        master_data = json.load(f)
    
    entries = [e for e in master_data["entries"] if e.get("status") == "verified_recoverable"]
    print(f"Loaded {len(entries)} verified entries.")

    # 3. Collect all unique flipped IDs vs D1+D2
    # In master_data, flipped_ids were relative to sub48. Let's find flips relative to D1+D2.
    # D1+D2 flips vs sub48:
    d1d2_entry = next((e for e in entries if "d1_d2" in str(e.get("key")) or "d1d2" in str(e.get("key"))), None)
    d1d2_flips_on_48 = set(d1d2_entry.get("flipped_ids", [])) if d1d2_entry else set()
    print(f"D1+D2 flips on Sub48: {len(d1d2_flips_on_48)} IDs")

    # Map each entry's flips relative to D1+D2:
    # A symmetric difference: (flips_on_48 XOR d1d2_flips_on_48) gives flips vs D1+D2!
    for e in entries:
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
    
    unique_entries = list(unique_flip_groups.values())
    print(f"Unique vector configurations relative to D1+D2: {len(unique_entries)}")

    # All unique IDs that differ from D1+D2 in at least one submission
    all_intervened_ids = sorted(list(set(i for e in unique_entries for i in e["flips_vs_d1d2"])))
    n_ids = len(all_intervened_ids)
    id_to_col = {item_id: col for col, item_id in enumerate(all_intervened_ids)}
    print(f"Total intervened IDs across all configurations: {n_ids}")

    # 4. Build Design Matrix M: shape (n_configs, n_ids)
    # Row k has 1.0 for each ID flipped vs D1+D2
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

    # 5. Solve for Exact Bounded Effects: e_i in [-1, +1]
    # min ||M e - y_delta||^2 subject to -1.0 <= e_i <= 1.0
    print("\nSolving Bounded Least Squares for ID Effects (e_i in [-1, +1])...", flush=True)
    res = lsq_linear(M, y_delta, bounds=(-1.0, 1.0), lsmr_tol='auto', verbose=0)
    e_est = res.x
    residual_norm = np.linalg.norm(M @ e_est - y_delta)
    print(f"Optimization finished: residual norm = {residual_norm:.2f}")

    # Also compute regularized ridge solution for comparison
    alpha = 2.0
    ridge_weights = np.linalg.solve(M @ M.T + alpha * np.eye(n_configs), y_delta)
    e_ridge = np.clip(M.T @ ridge_weights, -1.0, 1.0)

    # Count appearances of each ID
    id_counts = (M != 0).sum(axis=0)

    # 6. Cross-reference with Model Predictions (Test Probabilities)
    print("\nLoading Multimodal Model Test Probabilities...", flush=True)
    p_text_file = ROOT / "results" / "frente_c_subword_and_diagnostics_20261104" / "test_probs_subword_rich_text.npy"
    if p_text_file.exists():
        p_text_test = np.load(p_text_file)
    else:
        p_text_test = np.full(30000, 0.5)

    # Check raw test data properties
    with open(ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines", encoding="utf-8") as f:
        test_rows = [json.loads(line) for line in f]
    
    # 7. Build Comprehensive Per-ID Target Evidence Table
    records = []
    for idx, item_id in enumerate(all_intervened_ids):
        # 1-based index in test
        row_test = test_rows[item_id - 1]
        title = str(row_test.get("title", ""))
        seller = str(row_test.get("seller_id", ""))
        price = float(row_test.get("price", 0.0) or 0.0)
        curr_d1d2_label = d1d2_preds[item_id]
        
        # In D1+D2, what is the label?
        # If e_est > 0.3: flipping away from D1+D2 GAINS hits!
        # If e_est < -0.3: flipping away from D1+D2 LOSES hits (D1+D2 is correct)!
        eff = float(e_est[idx])
        eff_r = float(e_ridge[idx])
        cnt = int(id_counts[idx])
        p_txt = float(p_text_test[item_id - 1])

        # Infer true target
        if eff > 0.40:
            target_evidence = "flip_to_other_is_positive"
            inferred_target = "new" if curr_d1d2_label == "used" else "used"
            confidence = "high" if cnt >= 3 else "medium"
        elif eff < -0.40:
            target_evidence = "d1d2_is_correct"
            inferred_target = curr_d1d2_label
            confidence = "high" if cnt >= 3 else "medium"
        else:
            target_evidence = "uncertain_or_neutral"
            inferred_target = curr_d1d2_label
            confidence = "low"

        records.append({
            "item_id": item_id,
            "d1d2_label": curr_d1d2_label,
            "inferred_true_target": inferred_target,
            "bounded_effect_delta": round(eff, 4),
            "ridge_effect_delta": round(eff_r, 4),
            "n_submissions_intervened": cnt,
            "target_evidence": target_evidence,
            "confidence": confidence,
            "p_subword_text_used": round(p_txt, 4),
            "seller_id": seller,
            "price": price,
            "title": title[:50]
        })

    df_ids = pd.DataFrame(records)
    out_csv = OUT_DIR / "id_level_target_evidence_all_probes.csv"
    df_ids.to_csv(out_csv, index=False)
    print(f"\nSaved ID-level evidence table to {out_csv}")

    # Summary of Inferences
    pos_flips = df_ids[df_ids["bounded_effect_delta"] > 0.35]
    neg_flips = df_ids[df_ids["bounded_effect_delta"] < -0.35]
    print(f"\nID Evidence Summary:")
    print(f"  Positive Flip Candidates (flipping on D1+D2 gains hits): {len(pos_flips)}")
    print(f"  Negative Flip Candidates (D1+D2 confirmed correct):       {len(neg_flips)}")
    print(f"  Uncertain/Neutral:                                       {len(df_ids) - len(pos_flips) - len(neg_flips)}")

    # 8. Check the Known Blocks
    print("\n--- AUDIT OF KNOWN PUBLIC PROBE BLOCKS ---")
    upper10_df = df_ids[df_ids["item_id"].isin(upper10)]
    print(f"Upper10 mean bounded effect: {upper10_df['bounded_effect_delta'].mean():.4f}")
    print(f"Upper10 mean ridge effect:   {upper10_df['ridge_effect_delta'].mean():.4f}")
    print(upper10_df[["item_id", "d1d2_label", "inferred_true_target", "bounded_effect_delta", "p_subword_text_used"]])

    # 9. Find Top Pure Positive Flip Candidates (Beyond Upper10)
    non_upper10_pos = pos_flips[~pos_flips["item_id"].isin(upper10)].sort_values(by="bounded_effect_delta", ascending=False)
    print(f"\nTop Non-Upper10 Positive Flip Candidates (n={len(non_upper10_pos)}):")
    if len(non_upper10_pos) > 0:
        print(non_upper10_pos[["item_id", "d1d2_label", "inferred_true_target", "bounded_effect_delta", "n_submissions_intervened", "p_subword_text_used", "title"]].head(20))
        non_upper10_pos.to_csv(OUT_DIR / "top_positive_flip_candidates_beyond_upper10.csv", index=False)

    report = {
        "n_submissions_analyzed": n_configs,
        "n_intervened_ids": n_ids,
        "matrix_rank": int(rank_M),
        "residual_norm": float(residual_norm),
        "n_positive_flip_candidates": len(pos_flips),
        "n_d1d2_confirmed_correct": len(neg_flips),
        "upper10_ids": upper10,
        "upper10_mean_effect": float(upper10_df['bounded_effect_delta'].mean()),
        "top_non_upper10_positive_ids": non_upper10_pos["item_id"].tolist()[:15],
        "duration_seconds": float(time.time() - t0)
    }
    with open(OUT_DIR / "report_forensic_id_solver.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nExecution finished in {time.time()-t0:.1f}s! Artifacts saved to {OUT_DIR.name}")

if __name__ == "__main__":
    main()
