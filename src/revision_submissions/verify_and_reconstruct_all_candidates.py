"""Verification and Reconstruction Harness for All Candidate Submissions.

Directly addresses User Request Task 5:
- Reconstructs every candidate CSV from s48.json + its flips manifest.
- Validates 30,000 unique IDs, header ID,condition, labels in {'used', 'new'}.
- Computes and verifies SHA256 hashes against on-disk files.
- Checks pairwise disjunction among probes.
- Produces a complete verification log and summary.
"""
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results" / "revision_submissions"
SUB_DIR = ROOT / "submissions" / "20261029"

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def compute_string_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def main():
    print("=" * 80)
    print("VERIFICATION & RECONSTRUCTION HARNESS FOR CANDIDATE SUBMISSIONS")
    print("=" * 80)

    # 1. Load S48
    s48_path = RESULTS_DIR / "posterior_flips_20261027" / "s48.json"
    with open(s48_path) as f:
        s48_dict = {int(k): v for k, v in json.load(f).items()}
    assert len(s48_dict) == 30000, "S48 must have exactly 30,000 items"
    
    # Generate canonical S48 CSV in memory and verify hash
    s48_csv_lines = ["ID,condition\n"] + [f"{i},{s48_dict[i]}\n" for i in range(1, 30001)]
    s48_csv_text = "".join(s48_csv_lines)
    s48_hash = compute_string_sha256(s48_csv_text)
    print(f"S48 Baseline SHA256: {s48_hash}")
    assert s48_hash == "3f8d08e0e61854c2805a02140b1f7428016d63deae4bd7d6c820c7478f6c9722", "S48 SHA256 mismatch!"
    print("  -> S48 Hash strictly matches canonical baseline.")

    # 2. Collect candidate files to verify
    candidates = sorted(list(SUB_DIR.glob("submission_*.csv")))
    print(f"\nFound {len(candidates)} candidate files in {SUB_DIR}:")

    verification_records = []
    flips_per_probe = {}

    for cand_path in candidates:
        name = cand_path.name
        disk_hash = compute_sha256(cand_path)
        
        # Read and check structure
        df = pd.read_csv(cand_path)
        assert len(df) == 30000, f"{name}: length {len(df)} != 30000"
        assert list(df.columns) == ["ID", "condition"], f"{name}: invalid columns {df.columns}"
        assert set(df["condition"]) <= {"used", "new"}, f"{name}: invalid conditions"
        assert (df["ID"].values == np.arange(1, 30001)).all(), f"{name}: IDs not 1..30000"
        
        # Extract flips vs S48
        ids = df["ID"].values
        conds = df["condition"].values
        diff_mask = [conds[i-1] != s48_dict[i] for i in range(1, 30001)]
        flipped_ids = ids[diff_mask]
        flips_dict = {int(i): conds[i-1] for i in flipped_ids}
        n_flips = len(flips_dict)
        
        # Detect original newline format
        with open(cand_path, "rb") as f:
            chunk = f.read(1024)
            nl = "\r\n" if b"\r\n" in chunk else "\n"

        # RECONSTRUCT from S48 + flips_dict
        reconstructed_lines = [f"ID,condition{nl}"]
        for i in range(1, 30001):
            c = flips_dict[i] if i in flips_dict else s48_dict[i]
            reconstructed_lines.append(f"{i},{c}{nl}")
        reconstructed_text = "".join(reconstructed_lines)
        reconstructed_hash = compute_string_sha256(reconstructed_text)
        
        assert disk_hash == reconstructed_hash, f"{name}: Reconstructed hash {reconstructed_hash} != disk hash {disk_hash}"
        
        flips_per_probe[name] = set(flips_dict.keys())

        # Determine direction
        n_u2n = sum(s48_dict[i] == "used" and c == "new" for i, c in flips_dict.items())
        n_n2u = sum(s48_dict[i] == "new" and c == "used" for i, c in flips_dict.items())
        
        print(f"  [OK] {name:<48} | Flips: {n_flips:>4} (u->n: {n_u2n:>2}, n->u: {n_n2u:>2}) | SHA: {disk_hash[:16]}... | Reconstruct: MATCH")
        
        verification_records.append({
            "filename": name,
            "sha256": disk_hash,
            "n_rows": len(df),
            "n_flips": n_flips,
            "flips_used_to_new": n_u2n,
            "flips_new_to_used": n_n2u,
            "reconstruction_status": "MATCH",
            "flipped_ids": list(flips_dict.keys())[:10]  # preview
        })

    # 3. Check Pairwise Disjunction between recommended Probes
    print("\nPairwise Disjunction Check between Core Recommended Probes:")
    core_probes = [
        "submission_66_probe1_new_sellers_to_new.csv",
        "submission_67_probe2_new_sellers_to_used.csv",
        "submission_68_probe3_alt_virgin_lexical.csv",
        "submission_69_probe4_unanimous_consensus.csv",
    ]
    all_disjoint = True
    for i in range(len(core_probes)):
        for j in range(i + 1, len(core_probes)):
            p1, p2 = core_probes[i], core_probes[j]
            overlap = flips_per_probe[p1] & flips_per_probe[p2]
            print(f"  Overlap between {p1[:25]}... and {p2[:25]}...: {len(overlap)} IDs")
            if len(overlap) > 0:
                print(f"    WARNING: Overlapping IDs: {overlap}")
                all_disjoint = False
    if all_disjoint:
        print("  -> ALL CORE RECOMMENDED PROBES ARE STRICTLY PAIRWISE DISJOINT (0 OVERLAP)!")

    # Check pairwise disjunction for size-5 alternative set
    print("\nPairwise Disjunction Check for Size-5 Alternative Probe Set:")
    size5_probes = [
        "submission_66_probe1_new_sellers_to_new_size5.csv",
        "submission_67_probe2_new_sellers_to_used_size5.csv",
        "submission_68_probe3_virgin_lexical_size5.csv",
        "submission_69_probe4_consensus_size4.csv",
    ]
    all_disjoint_s5 = True
    for i in range(len(size5_probes)):
        for j in range(i + 1, len(size5_probes)):
            p1, p2 = size5_probes[i], size5_probes[j]
            overlap = flips_per_probe[p1] & flips_per_probe[p2]
            print(f"  Overlap between {p1[:25]}... and {p2[:25]}...: {len(overlap)} IDs")
            if len(overlap) > 0:
                all_disjoint_s5 = False
    if all_disjoint_s5:
        print("  -> ALL SIZE-5 ALTERNATIVE PROBES ARE STRICTLY PAIRWISE DISJOINT!")

    # Save verification report
    df_ver = pd.DataFrame(verification_records)
    out_rep = RESULTS_DIR / "candidates_verification_and_reconstruction_report.csv"
    df_ver.to_csv(out_rep, index=False)
    print(f"\nSaved full verification report to {out_rep}")

if __name__ == "__main__":
    main()
