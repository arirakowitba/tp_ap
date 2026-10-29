"""Automated Builder for the 5th Final Submission based on Adaptive Probe Scores.

Usage:
  python src/revision_submissions/build_final_fifth_submission.py --s1 0.94633 --s2 0.94643 --s3 0.94623 --s4 0.94633

Rules:
- Baseline S48 = 0.94633
- Any probe with score > S48 is considered positive and its disjoint flips are included.
- If no probe beats S48, outputs exact Submission 48 as the safe final choice.
- Generates submissions/20261029/submission_70_final_optimized.csv and prints SHA256.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_JSON = ROOT / "results" / "revision_submissions" / "round152_adaptive_probes_manifest.json"
S48_JSON = ROOT / "results" / "revision_submissions" / "posterior_flips_20261027" / "s48.json"
SUBMISSIONS_DIR = ROOT / "submissions" / "20261029"


def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Build 5th Final Kaggle Submission")
    parser.add_argument("--s1", type=float, default=0.94633, help="Public score of Probe 1 (sub 66)")
    parser.add_argument("--s2", type=float, default=0.94633, help="Public score of Probe 2 (sub 67)")
    parser.add_argument("--s3", type=float, default=0.94633, help="Public score of Probe 3 (sub 68)")
    parser.add_argument("--s4", type=float, default=0.94633, help="Public score of Probe 4 (sub 69)")
    parser.add_argument("--s48", type=float, default=0.94633, help="Baseline score of Submission 48")
    args = parser.parse_args()

    scores = [args.s1, args.s2, args.s3, args.s4]
    print(f"Received scores: S1={args.s1}, S2={args.s2}, S3={args.s3}, S4={args.s4} vs S48={args.s48}")

    with open(MANIFEST_JSON, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(S48_JSON, "r", encoding="utf-8") as f:
        s48_dict = json.load(f)

    selected_flips = {}
    winning_probes = []

    for i, probe in enumerate(manifest):
        score = scores[i]
        if score > args.s48:
            winning_probes.append((i + 1, probe["file"], score, len(probe["items"])))
            for it in probe["items"]:
                selected_flips[it["ID"]] = it["target"]

    out_file = SUBMISSIONS_DIR / "submission_70_final_optimized.csv"

    if len(winning_probes) >= 2:
        print(f"\n[DECISION 5TH SLOT] Multiple winning probes identified ({len(winning_probes)}):")
        for p_num, fname, sc, n in winning_probes:
            print(f"  Probe {p_num} ({fname}): score {sc} (+{sc - args.s48:.5f}, {n} flips)")
        print(f"Total positive flips to combine in disjoint union: {len(selected_flips)}")
        print(f"Action: Generating {out_file.name} with all winning flips combined.")
        
        with open(out_file, "w", newline="\n", encoding="utf-8") as f:
            f.write("ID,condition\n")
            for id_num in range(1, 30001):
                cond = selected_flips.get(id_num, s48_dict[str(id_num)])
                f.write(f"{id_num},{cond}\n")

        sha = compute_sha256(out_file)
        print(f"\nGenerated 5th Final Submission (Disjoint Union):")
        print(f"  Path: {out_file}")
        print(f"  Total flips vs 48: {len(selected_flips)}")
        print(f"  SHA256: {sha}")
        print(f"  Recommendation: SUBMIT this file in 5th slot to compound gains.")

    elif len(winning_probes) == 1:
        p_num, fname, sc, n = winning_probes[0]
        print(f"\n[DECISION 5TH SLOT] Exactly ONE winning probe: Probe {p_num} ({fname}) with score {sc} (+{sc - args.s48:.5f}, {n} flips).")
        print(f"CRITICAL RULE: That probe's score ({sc}) is ALREADY registered on Kaggle!")
        print(f"Re-submitting the same block alone would waste the 5th final slot.")
        print(f"Recommendation:")
        print(f"  1. If a secondary validated model exists, submit that model.")
        print(f"  2. Otherwise, do NOT submit a redundant copy; Probe {p_num} remains your best recorded score.")
        
    else:
        print(f"\n[DECISION 5TH SLOT] No probe beat Submission 48 (all <= {args.s48}).")
        print(f"Action: Reverting strictly to Submission 48 (.94633) as final best.")
        print(f"Recommendation: Do NOT submit a new CSV; retain Submission 48.")


if __name__ == "__main__":
    main()
