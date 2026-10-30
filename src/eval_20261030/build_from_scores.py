"""Adaptive 5th Slot Decision Engine & Combiner.

Directly addresses User Request Task 6 & Strategy Plan:
- Takes explicit JSON mapping of evaluated probes {file, sha256, score}.
- Verifies file integrity, schema, and SHA256.
- Enforces strict decision rules:
  * Ties (score == 0.94633) are NOT promoted.
  * Negative probes (score < 0.94633) are excluded.
  * Disjoint union vs nested contrast resolution:
    - Prohibits summing deltas of overlapping blocks.
    - Resolves candidate 77, 78, 80 overlaps via conditioned nested contrast rules:
      * Sub 80 is a subset of 77 (15 of 20 flips). Marginal 5 flips (77 \\ 80) accepted only if Delta_77 > Delta_80.
      * Sub 78 shares 10 flips with 77/80. If 77/80 wins, Sub 78 only contributes its 15 consensus-unique flips.
      * Sub 79 is 100% disjoint (20 flips).
    - Unresolved overlaps outside nested contracts strictly trigger exit to prevent unsafe unions.
    - Seamlessly supports redesigned orthogonal blocks.
  * Exactly 1 winning probe: warns against redundant re-submission if identical to already scored probe.
  * >= 2 winning blocks: builds strictly disjoint union.
- Runs submission_guard to ensure the final output is safe, non-duplicate, and valid.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
S48_JSON = ROOT / "results" / "revision_submissions" / "posterior_flips_20261027" / "s48.json"
from src.eval_20261030.submission_guard import check_submission_file, REGISTRY_JSON

# Pre-computed structural relationships for legacy candidate suite
SUB77_MINUS_SUB80 = {732, 4969, 18735, 18790, 25747}
SUB78_CONSENSUS_UNIQUE = {
    3619, 4599, 5028, 5205, 6562, 7278, 12641, 17714,
    19545, 20200, 22327, 25326, 25600, 25940, 28448
}
SUB78_SHARED_WITH_77_80 = {
    732, 3802, 6128, 6487, 6940, 7575, 18790, 19403, 25976, 29552
}

def compute_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Adaptive 5th Slot Decision Engine & Combiner")
    parser.add_argument("--scores-json", type=str, required=True,
                        help="Path to JSON file with format: [{'file': 'path/to/cand.csv', 'sha256': '...', 'score': 0.94645}, ...]")
    parser.add_argument("--out-csv", type=str, default="submissions/20261030/submission_final_combined.csv",
                        help="Path to output combined submission CSV")
    parser.add_argument("--base-score", type=float, default=0.94633, help="Baseline score of S48")
    args = parser.parse_args()

    scores_path = Path(args.scores_json)
    if not scores_path.exists():
        sys.exit(f"Scores JSON file not found: {scores_path}")

    with open(scores_path, "r", encoding="utf-8") as f:
        probe_list = json.load(f)

    if not isinstance(probe_list, list) or len(probe_list) == 0:
        sys.exit("Scores JSON must contain a non-empty list of probe evaluation records.")

    # Load S48 reference
    with open(S48_JSON, "r", encoding="utf-8") as f:
        s48_dict = {int(k): v for k, v in json.load(f).items()}

    print("=" * 80)
    print(f"EVALUATING PROBE RESULTS (BASELINE S48 = {args.base_score:.5f})")
    print("=" * 80)

    winning_probes = []
    excluded_probes = []
    tied_probes = []

    for entry in probe_list:
        csv_file = Path(entry["file"])
        expected_sha = entry.get("sha256")
        score = entry.get("score")

        if score is None:
            sys.exit(f"Missing score for file {csv_file}. Do not pass fictitious scores!")

        if not csv_file.exists():
            sys.exit(f"Candidate file not found: {csv_file}")

        actual_sha = compute_sha256(csv_file)
        if expected_sha and actual_sha != expected_sha:
            sys.exit(f"Hash mismatch for {csv_file.name}: expected {expected_sha}, got {actual_sha}")

        df = pd.read_csv(csv_file)
        flips = {}
        for _, r in df.iterrows():
            i = int(r["ID"])
            c = str(r["condition"]).strip()
            if c != s48_dict[i]:
                flips[i] = c

        delta = round(score - args.base_score, 5)
        record = {
            "file": csv_file,
            "name": csv_file.name,
            "sha256": actual_sha,
            "score": score,
            "delta": delta,
            "n_flips": len(flips),
            "flips": flips
        }

        print(f"\nProbe: {csv_file.name}")
        print(f"  Flips vs S48: {len(flips)}")
        print(f"  Score: {score:.5f} (Delta vs S48: {delta:+.5f})")

        if score > args.base_score:
            print("  -> STATUS: POSITIVE WINNER (beats S48)")
            winning_probes.append(record)
        elif score == args.base_score:
            print("  -> STATUS: TIE (score == S48; NOT promoted)")
            tied_probes.append(record)
        else:
            print("  -> STATUS: NEGATIVE LOSS (excluded)")
            excluded_probes.append(record)

    print("\n" + "=" * 80)
    print("DECISION SYNTHESIS & NESTED CONTRAST RESOLUTION")
    print("=" * 80)

    if not winning_probes:
        print(f"NO PROBE BEAT SUBMISSION 48 (all <= {args.base_score:.5f}).")
        print("CRITICAL DIRECTIVE: Submission 48 remains the uncontested best known public benchmark.")
        print("RECOMMENDATION: Do NOT submit a new CSV; retain Submission 48.")
        return

    # Map probes by normalized name or identifier
    probes_by_tag = {}
    for p in winning_probes:
        fname = p["name"].lower()
        if "77" in fname:
            probes_by_tag["77"] = p
        elif "78" in fname:
            probes_by_tag["78"] = p
        elif "79" in fname:
            probes_by_tag["79"] = p
        elif "80" in fname:
            probes_by_tag["80"] = p
        else:
            probes_by_tag[fname] = p

    # Check for overlapping blocks among winners
    has_sub80 = "80" in probes_by_tag
    has_sub77 = "77" in probes_by_tag
    has_sub78 = "78" in probes_by_tag
    has_sub79 = "79" in probes_by_tag

    selected_flips = {}
    selected_sources = {}

    print("\nApplying Nested Contrast and Disjoint Partition Rules:")

    # 1. Evaluate Sub 80 vs Sub 77 nested contrast
    if has_sub80 and has_sub77:
        score_80 = probes_by_tag["80"]["score"]
        score_77 = probes_by_tag["77"]["score"]
        delta_margin = round(score_77 - score_80, 5)
        print(f"  [Nested Contrast 77 vs 80]: Sub 80 Score = {score_80:.5f}, Sub 77 Score = {score_77:.5f}")
        print(f"    Marginal Delta (Sub 77 - Sub 80) = {delta_margin:+.5f}")
        
        # Sub 80 core is accepted
        for i, c in probes_by_tag["80"]["flips"].items():
            selected_flips[i] = c
            selected_sources[i] = "Sub80_core"

        if delta_margin > 0:
            print("    -> DECISION: Marginal 5 items improved performance! Retaining all 20 flips of Sub 77.")
            for i, c in probes_by_tag["77"]["flips"].items():
                selected_flips[i] = c
                if i in SUB77_MINUS_SUB80:
                    selected_sources[i] = "Sub77_margin5"
        else:
            print("    -> DECISION: Marginal 5 items did NOT improve score (Delta <= 0). Retaining only Sub 80 (15 flips).")
    elif has_sub80:
        print("  [Sub 80 Solo Winner]: Retaining 15 core flips of Sub 80.")
        for i, c in probes_by_tag["80"]["flips"].items():
            selected_flips[i] = c
            selected_sources[i] = "Sub80_core"
    elif has_sub77:
        print("  [Sub 77 Solo Winner]: Retaining 20 flips of Sub 77.")
        for i, c in probes_by_tag["77"]["flips"].items():
            selected_flips[i] = c
            selected_sources[i] = "Sub77_full"

    # 2. Evaluate Sub 78
    if has_sub78:
        if has_sub80 or has_sub77:
            print("  [Sub 78 Consensus Disjoint]: Sub 77 or 80 also won.")
            print("    -> Retaining ONLY the 15 consensus-unique flips of Sub 78 (skipping 10 overlapping IDs).")
            print("    -> Prohibiting naive delta summation on overlapping items!")
            for i, c in probes_by_tag["78"]["flips"].items():
                if i in SUB78_CONSENSUS_UNIQUE:
                    selected_flips[i] = c
                    selected_sources[i] = "Sub78_consensus15"
        else:
            print("  [Sub 78 Solo Winner]: Retaining all 25 flips of Sub 78.")
            for i, c in probes_by_tag["78"]["flips"].items():
                selected_flips[i] = c
                selected_sources[i] = "Sub78_full"

    # 3. Evaluate Sub 79
    if has_sub79:
        print("  [Sub 79 Disjoint Block]: 100% disjoint from other probes. Retaining all 20 flips.")
        for i, c in probes_by_tag["79"]["flips"].items():
            selected_flips[i] = c
            selected_sources[i] = "Sub79_disjoint20"

    # 4. Handle any other arbitrary probes
    for tag, p in probes_by_tag.items():
        if tag not in {"77", "78", "79", "80"}:
            print(f"  [Additional Probe: {p['name']}]: Checking disjointness...")
            for i, c in p["flips"].items():
                if i in selected_flips:
                    sys.exit(f"Cannot combine overlapping winning blocks: ID {i} is shared between {p['name']} and {selected_sources[i]}. Aggregate score gains are not additive. Prepare a disjoint or explicitly nested experiment before generating a final CSV.")
                selected_flips[i] = c
                selected_sources[i] = p["name"]

    print("\n" + "=" * 80)
    print(f"FINAL COMBINED COMPOSITION: {len(selected_flips)} TOTAL FLIPS")
    print("=" * 80)
    source_counts = pd.Series(list(selected_sources.values())).value_counts()
    for src_name, cnt in source_counts.items():
        print(f"  - {src_name}: {cnt} items")

    print("\nCRITICAL MATHEMATICAL GUARD:")
    print("  DO NOT sum public deltas of overlapping candidate probes.")
    print("  The compound submission combines independent orthogonal signals without double-counting.")

    if len(winning_probes) == 1 and len(selected_flips) == winning_probes[0]["n_flips"]:
        win = winning_probes[0]
        print(f"\nNOTE: Exactly ONE probe won ({win['name']}) and no multi-probe union was formed.")
        print(f"This file has already been evaluated on Kaggle (Score: {win['score']:.5f}).")
        print("Resubmitting the identical file would waste the 5th adaptive slot.")
        print("Retain this submission as your current best candidate.")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["ID,condition\n"]
    for i in range(1, 30001):
        c = selected_flips.get(i, s48_dict[i])
        lines.append(f"{i},{c}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    final_sha = compute_sha256(out_path)
    print(f"\nGenerated combined submission file: {out_path} (SHA256: {final_sha})")

    # Validate with submission guard
    with open(REGISTRY_JSON, "r", encoding="utf-8") as f:
        reg_data = json.load(f)
    passed, guard_rep = check_submission_file(out_path, reg_data, s48_dict)
    if not passed:
        sys.exit(f"Submission guard check FAILED: {guard_rep}")
    print("Submission guard PASSED: Output is verified safe, non-duplicate, and strictly formatted.")

if __name__ == "__main__":
    main()
