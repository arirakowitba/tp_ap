"""Duplicate Guard and Submission Safety Harness.

Directly prevents submitting duplicates or previously scored blocks:
- Validates structural schema: 30,000 rows, IDs 1..30000, header ID,condition, labels in {new, used}.
- BLOCKS (exit code 1) if file is an exact duplicate of 48 or any scored submission (like 76).
- BLOCKS (exit code 1) if the exact set of flips vs 48 equals any previously scored block.
- WARNS if flips overlap with refuted negative blocks (72, 73, 74, 59, 45, 46, 39, etc.) or tied block 75.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
REGISTRY_JSON = ROOT / "results" / "eval_20261030" / "scored_registry.json"
S48_JSON = ROOT / "results" / "revision_submissions" / "posterior_flips_20261027" / "s48.json"

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()

def sha256_vector(labels_dict):
    vec = "".join("1" if labels_dict[i] == "used" else "0" for i in range(1, 30001))
    return hashlib.sha256(vec.encode("ascii")).hexdigest()

def check_submission_file(csv_path, registry_data, s48_dict, verbose=True):
    path = Path(csv_path)
    if not path.exists():
        return False, f"File not found: {path}"

    # 1. Structural schema validation
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return False, f"CSV parse error: {e}"

    if len(df) != 30000:
        return False, f"Row count error: expected 30000, got {len(df)}"
    if list(df.columns) != ["ID", "condition"]:
        return False, f"Column error: expected ['ID', 'condition'], got {list(df.columns)}"
    if not (df["ID"].values == list(range(1, 30001))).all():
        return False, "ID sequencing error: IDs must be strictly 1..30000 sequential"
    
    conditions = df["condition"].astype(str).str.strip().tolist()
    unique_conds = set(conditions)
    if not unique_conds.issubset({"new", "used"}):
        return False, f"Invalid condition values: {unique_conds - {'new', 'used'}}"

    labels = {i + 1: c for i, c in enumerate(conditions)}
    file_bytes = path.read_bytes()
    raw_sha = sha256_bytes(file_bytes)
    vec_sha = sha256_vector(labels)

    # 2. Compare against S48
    flips_vs_48 = {}
    for i in range(1, 30001):
        if labels[i] != s48_dict[i]:
            flips_vs_48[i] = {"from_48": s48_dict[i], "to": labels[i]}
    n_flips = len(flips_vs_48)

    # BLOCK CHECK 1: Duplicate of S48
    if n_flips == 0 or vec_sha == registry_data["baseline_48"]["sha256"]:
        return False, (
            f"BLOCKED: Exact duplicate of Baseline Submission 48 (SHA256: {raw_sha})!\n"
            f"Submitting this would repeat the mistake of Submission 76, burning a slot for 0 information."
        )

    # BLOCK CHECK 2: Duplicate of any scored submission
    flipped_id_set = set(flips_vs_48.keys())
    for entry in registry_data["entries"]:
        sub_key = entry["key"]
        sub_score = entry["score"]
        
        # Check label vector match
        if vec_sha == entry["sha256_vector"]:
            return False, (
                f"BLOCKED: Exact label match with previously scored submission '{sub_key}' "
                f"(Score: {sub_score}, Delta: {entry['delta_vs_48']:+.5f})!\n"
                f"This prediction has ALREADY been evaluated on Kaggle."
            )
            
        # Check identical flip set vs 48
        entry_flips = set(entry["flipped_ids"])
        if len(entry_flips) > 0 and flipped_id_set == entry_flips:
            # Check if targets also match
            if all(labels[i] == entry["flips_dict"][str(i)] for i in flipped_id_set):
                return False, (
                    f"BLOCKED: Exact flip set match with previously scored submission '{sub_key}' "
                    f"({len(flipped_id_set)} flips, Score: {sub_score})!\n"
                    f"This exact block of changes has already been tested."
                )

    # WARN CHECK: Overlaps with refuted negative blocks or ties
    warnings = []
    refuted_keys = {"72", "73", "74", "59_full", "fin_stack_new150", "fin_stack_seen150", "45", "46", "49", "50", "39", "37", "38"}
    for entry in registry_data["entries"]:
        k = entry["key"]
        entry_flips = set(entry["flipped_ids"])
        overlap = flipped_id_set & entry_flips
        if overlap:
            matched_targets = sum(labels[i] == entry["flips_dict"][str(i)] for i in overlap)
            if k in refuted_keys:
                warnings.append(
                    f"WARNING: Overlaps with REFUTED negative submission '{k}' (Score: {entry['score']}) "
                    f"on {len(overlap)} IDs ({matched_targets} with same target direction)."
                )
            elif k == "75":
                warnings.append(
                    f"NOTICE: Overlaps with TIED submission '75' (Score: 0.94633) on {len(overlap)} IDs. "
                    f"Note that 75 did NOT demonstrate public improvement."
                )

    report = {
        "file": path.name,
        "valid": True,
        "n_rows": 30000,
        "n_flips_vs_48": n_flips,
        "sha256": raw_sha,
        "flips_used_to_new": sum(s48_dict[i] == "used" and labels[i] == "new" for i in flips_vs_48),
        "flips_new_to_used": sum(s48_dict[i] == "new" and labels[i] == "used" for i in flips_vs_48),
        "warnings": warnings,
    }
    return True, report

def main():
    parser = argparse.ArgumentParser(description="Submission Duplicate Guard & Validator")
    parser.add_argument("csv_files", nargs="+", help="Candidate CSV file(s) to check")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    args = parser.parse_args()

    if not REGISTRY_JSON.exists():
        sys.exit(f"Registry not found at {REGISTRY_JSON}. Run build_scored_registry.py first.")

    with open(REGISTRY_JSON, "r", encoding="utf-8") as f:
        registry_data = json.load(f)

    with open(S48_JSON, "r", encoding="utf-8") as f:
        s48_dict = {int(k): v for k, v in json.load(f).items()}

    all_passed = True
    for csv_file in args.csv_files:
        passed, res = check_submission_file(csv_file, registry_data, s48_dict, verbose=not args.quiet)
        if passed:
            print(f"[PASS] {res['file']}")
            print(f"       Flips vs 48: {res['n_flips_vs_48']} (used->new: {res['flips_used_to_new']}, new->used: {res['flips_new_to_used']})")
            print(f"       SHA256: {res['sha256']}")
            for w in res["warnings"]:
                print(f"       ! {w}")
        else:
            all_passed = False
            print(f"[FAIL] {Path(csv_file).name}")
            print(f"       Reason: {res}")
        print()

    if not all_passed:
        sys.exit(1)

if __name__ == "__main__":
    main()
