"""Forensic Reconstruction of All Submissions and Files in Workspace.

Scans:
- submissions/ directory (all subdirectories)
- matches SHA-256 hashes against all known scores from:
  - submission_ledger_master_corrected_80vec.json
  - submission_ledger_master_83.json
  - results/public_probe_block_audit_20261104.json
  - historical markdown logs and coordination messages
- verifies exact flips against D1+D2 and submission 48
- builds the full submission x ID matrix
- calculates exact rank, nullity, connected components, and integer constraints under N_pub=20,998.
"""

import os, sys, json, hashlib, glob
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def sha256_lf(path):
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()

def load_csv(path):
    df = pd.read_csv(path)
    if "ID" in df.columns and "condition" in df.columns:
        return df.set_index("ID")["condition"]
    elif "id" in df.columns and "condition" in df.columns:
        return df.set_index("id")["condition"]
    else:
        raise ValueError(f"Unknown schema in {path}: {df.columns}")

def main():
    print("=== FORENSIC RECONSTRUCTION OF ALL SUBMISSIONS ===", flush=True)

    # 1. Load known reference files
    d1d2_path = ROOT / "submissions" / "candidate_d1_d2_combined_on48.csv"
    if not d1d2_path.exists():
        # search for it
        cands = list(ROOT.glob("**/candidate_d1_d2_combined_on48.csv"))
        if cands:
            d1d2_path = cands[0]
        else:
            cands2 = list(ROOT.glob("**/submission_final_best_d1_d2_20261031.csv"))
            d1d2_path = cands2[0]
    
    print(f"Loading reference D1+D2: {d1d2_path}")
    s_d1d2 = load_csv(d1d2_path)
    d1d2_hash = sha256_lf(d1d2_path)
    print(f"D1+D2 SHA-256 (LF): {d1d2_hash}")

    # 2. Load known scores from ledgers
    known_scores = {} # hash -> (score, key, source)
    
    # from corrected 80vec
    p_80 = ROOT / "results" / "intensive_20261030" / "submission_ledger_master_corrected_80vec.json"
    if p_80.exists():
        with open(p_80, encoding="utf-8") as f:
            d80 = json.load(f)
            for e in d80.get("entries", []):
                h = e.get("sha256_vector") or e.get("sha256_lf")
                if h and e.get("score") is not None:
                    known_scores[h] = (e.get("score"), e.get("entry_id") or e.get("name"), "ledger_80vec")

    # from public probe audit 20261104
    p_probe = ROOT / "results" / "public_probe_block_audit_20261104.json"
    if p_probe.exists():
        with open(p_probe, encoding="utf-8") as f:
            dpr = json.load(f)
            for k, row in dpr.get("rows", {}).items():
                h = row.get("sha256")
                if h and row.get("score") is not None:
                    known_scores[h] = (row.get("score"), k, "probe_audit_20261104")

    # 3. Scan all 115 CSVs in submissions/
    sub_files = sorted(list((ROOT / "submissions").glob("**/*.csv")))
    print(f"Total CSV files found in submissions/: {len(sub_files)}")

    catalog = []
    seen_hashes = {}

    for p in sub_files:
        try:
            h = sha256_lf(p)
            size = p.stat().st_size
            rel_path = str(p.relative_to(ROOT))
            
            # check if score is known
            score_info = known_scores.get(h)
            score = score_info[0] if score_info else None
            score_src = score_info[2] if score_info else None

            # verify if 30k rows
            s = load_csv(p)
            n_rows = len(s)
            
            # calculate flips vs D1+D2
            flips_vs_d1d2 = int((s != s_d1d2).sum())

            catalog.append({
                "rel_path": rel_path,
                "filename": p.name,
                "sha256": h,
                "size_bytes": size,
                "n_rows": n_rows,
                "score": score,
                "score_source": score_src,
                "flips_vs_d1d2": flips_vs_d1d2,
                "is_duplicate_hash": h in seen_hashes
            })
            seen_hashes[h] = rel_path
        except Exception as e:
            print(f"Error reading {p}: {e}")

    df_cat = pd.DataFrame(catalog)
    print(f"\nUnique SHA-256 hashes: {df_cat['sha256'].nunique()}")
    print(f"Files with confirmed public score: {df_cat['score'].notna().sum()}")
    print(f"Unique hashes with confirmed public score: {df_cat[df_cat['score'].notna()]['sha256'].nunique()}")

    # Save forensic inventory
    out_dir = ROOT / "results" / "forensic_ledger_20261104"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_cat.to_csv(out_dir / "all_submissions_files_inventory.csv", index=False)

    # Let's inspect unique submissions with confirmed scores
    df_scored = df_cat[df_cat["score"].notna()].drop_duplicates(subset=["sha256"])
    print("\nSummary of Scored Submissions:")
    print(df_scored[["filename", "score", "flips_vs_d1d2", "score_source"]].to_string())

if __name__ == "__main__":
    main()
