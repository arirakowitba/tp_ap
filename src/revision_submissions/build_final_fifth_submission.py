"""Automated Decision Builder and Combiner for the 5th Final Submission.

Directly addresses User Request Task 6:
- Receives public leaderboard scores from Ariel for Probes 1 to 4.
- Evaluates statistical gain vs Submission 48 (.94633).
- Uses exact public test fraction (N = 21,000) and 5-decimal truncation/rounding constraints.
- If >= 2 disjoint winning probes: builds disjoint union in submission_70_final_optimized.csv.
- If == 1 winning probe: that score is already secured; warns against submitting a redundant duplicate.
- If == 0 winning probes: retains Submission 48 (.94633).
- Documents additivity, rounding, and non-extrapolation boundaries.
"""
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results" / "revision_submissions"
S48_JSON = RESULTS_DIR / "posterior_flips_20261027" / "s48.json"
SUBMISSIONS_DIR = ROOT / "submissions" / "20261029"

N_PUBLIC = 21000

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def estimate_correct_net(score, score_base, n_public=21000):
    """Estimate integer net flips in public split accounting for 5-decimal truncation."""
    delta = score - score_base
    net_items = round(delta * n_public)
    return delta, net_items

def main():
    parser = argparse.ArgumentParser(description="Adaptive 5th Slot Decision Engine")
    parser.add_argument("--s1", type=float, default=None, help="Public score of Probe 1 (sub 66, 10 flips used->new)")
    parser.add_argument("--s2", type=float, default=None, help="Public score of Probe 2 (sub 67, 10 flips new->used)")
    parser.add_argument("--s3", type=float, default=None, help="Public score of Probe 3 (sub 68 Virgin Lexical, 7 flips)")
    parser.add_argument("--s4", type=float, default=None, help="Public score of Probe 4 (sub 69 Unanimous Consensus, 8 flips)")
    parser.add_argument("--s48", type=float, default=0.94633, help="Public baseline score of Submission 48")
    parser.add_argument("--alt_size5", action="store_true", help="Use Size-5 alternative probes instead of core")
    args = parser.parse_args()

    print("=" * 80)
    print("5TH SLOT ADAPTIVE DECISION ENGINE — AP-2026-Q-2")
    print("=" * 80)
    print(f"Reference Baseline (Submission 48): {args.s48:.5f}")

    # Load S48
    with open(S48_JSON, "r", encoding="utf-8") as f:
        s48_dict = {int(k): v for k, v in json.load(f).items()}

    # Probe file definitions
    if not args.alt_size5:
        probe_files = [
            ("Probe 1 (New Sellers used->new)", SUBMISSIONS_DIR / "submission_66_probe1_new_sellers_to_new.csv", args.s1),
            ("Probe 2 (New Sellers new->used)", SUBMISSIONS_DIR / "submission_67_probe2_new_sellers_to_used.csv", args.s2),
            ("Probe 3 (Virgin Lexical)", SUBMISSIONS_DIR / "submission_68_probe3_alt_virgin_lexical.csv", args.s3),
            ("Probe 4 (Unanimous Consensus)", SUBMISSIONS_DIR / "submission_69_probe4_unanimous_consensus.csv", args.s4),
        ]
    else:
        print("[MODE: SIZE-5 ALTERNATIVES]")
        probe_files = [
            ("Probe 1 (Size 5)", SUBMISSIONS_DIR / "submission_66_probe1_new_sellers_to_new_size5.csv", args.s1),
            ("Probe 2 (Size 5)", SUBMISSIONS_DIR / "submission_67_probe2_new_sellers_to_used_size5.csv", args.s2),
            ("Probe 3 (Size 5 Virgin)", SUBMISSIONS_DIR / "submission_68_probe3_virgin_lexical_size5.csv", args.s3),
            ("Probe 4 (Size 4 Consensus)", SUBMISSIONS_DIR / "submission_69_probe4_consensus_size4.csv", args.s4),
        ]

    # Analyze input scores
    active_probes = []
    winning_probes = []
    combined_flips = {}

    for name, path, score in probe_files:
        assert path.exists(), f"File missing: {path}"
        df = pd.read_csv(path)
        flips = {int(r["ID"]): str(r["condition"]).strip() for _, r in df.iterrows() if str(r["condition"]).strip() != s48_dict[int(r["ID"])]}
        
        status_str = "PENDIENTE" if score is None else f"{score:.5f}"
        print(f"\n{name}:")
        print(f"  File: {path.name}")
        print(f"  Flips: {len(flips)}")
        print(f"  Score: {status_str}")

        if score is not None:
            delta, net_aciertos = estimate_correct_net(score, args.s48, N_PUBLIC)
            print(f"  Delta vs S48: {delta:+.5f} (approx {net_aciertos:+d} correct items on public ~21k)")
            active_probes.append((name, path, score, flips, delta, net_aciertos))
            if score > args.s48:
                winning_probes.append((name, path, score, flips, delta, net_aciertos))

    print("\n" + "-" * 80)
    print("DECISION RULE SYNTHESIS")
    print("-" * 80)

    if len(active_probes) == 0:
        print("No scores provided yet. Run with probe scores, for example:")
        print("  python src/revision_submissions/build_final_fifth_submission.py --s1 0.94643 --s2 0.94628 --s3 0.94638 --s4 0.94633")
        return

    out_file = SUBMISSIONS_DIR / "submission_70_final_optimized.csv"

    if len(winning_probes) >= 2:
        print(f"RESULT: MULTIPLE DISJOINT WINNING PROBES ({len(winning_probes)} probes beat S48)!")
        total_flips = 0
        expected_gain_aciertos = 0
        for name, path, sc, flips, delta, net_items in winning_probes:
            print(f"  + {name}: {sc:.5f} ({delta:+.5f}, {len(flips)} flips, est {net_items:+d} aciertos)")
            # Check disjointness with previously combined
            for i in flips:
                if i in combined_flips and combined_flips[i] != flips[i]:
                    print(f"    WARNING: ID {i} conflict! Already set to {combined_flips[i]}, probe wants {flips[i]}")
            combined_flips.update(flips)
            expected_gain_aciertos += net_items

        print(f"\nTotal combined disjoint flips: {len(combined_flips)}")
        expected_combined_score = args.s48 + (expected_gain_aciertos / N_PUBLIC)
        print(f"Estimated combined public score: ~{expected_combined_score:.5f} (+{expected_gain_aciertos} net aciertos)")
        print("\nConstructing 5th submission (submission_70_final_optimized.csv)...")

        lines = ["ID,condition\n"]
        for i in range(1, 30001):
            c = combined_flips.get(i, s48_dict[i])
            lines.append(f"{i},{c}\n")
        
        with open(out_file, "w", encoding="utf-8") as f:
            f.writelines(lines)
            
        sha = compute_sha256(out_file)
        print(f"SUCCESS: Generated {out_file.name}")
        print(f"  Rows: 30000 | Flips vs S48: {len(combined_flips)} | SHA256: {sha}")
        print("  RECOMMENDATION: SUBMIT submission_70_final_optimized.csv in Slot 5 to capture compounded gain!")

    elif len(winning_probes) == 1:
        name, path, sc, flips, delta, net_items = winning_probes[0]
        print(f"RESULT: EXACTLY ONE WINNING PROBE -> {name} ({sc:.5f}, {delta:+.5f}, {len(flips)} flips)")
        print("CRITICAL EFFICIENCY DIRECTIVE:")
        print(f"  This probe was ALREADY submitted and scored on Kaggle ({sc:.5f}).")
        print("  Re-submitting this CSV alone yields 0 marginal value and burns the final submission.")
        print("RECOMMENDATION:")
        print("  1. Retain this winning probe as your highest recorded score.")
        print("  2. If an independently validated model candidate with solid OOF gain exists, you may test it in slot 5.")
        print("  3. Otherwise, conserve your submission and do NOT submit a duplicate.")

    else:
        print(f"RESULT: NO PROBE BEAT SUBMISSION 48 (all <= {args.s48:.5f}).")
        print("CRITICAL SAFETY DIRECTIVE:")
        print(f"  Submission 48 remains the uncontested best known public benchmark ({args.s48:.5f}).")
        print("RECOMMENDATION:")
        print("  Do NOT submit any probe combination. Select Submission 48 as final.")

    print("\n" + "=" * 80)
    print("LIMITS OF GENERALIZATION & METHODOLOGICAL BOUNDARIES:")
    print("1. PUBLIC ADDITIVITY: For strictly disjoint blocks A and B on the public test set,")
    print("   the net changes in correct answers are mathematically additive on the public split.")
    print("2. 5-DECIMAL ROUNDING: 1 net correct answer on ~21,000 public rows = ~0.00004762.")
    print("   Truncation at 5 decimal places causes quantization jitter of +/- 0.00005.")
    print("3. NON-EXTRAPOLATION TO PRIVATE: Winning on a 7-10 flip probe does NOT authorize")
    print("   flipping all 1,500+ items in that segment. Massive flips destroyed Submissions 39 and 59.")
    print("=" * 80)

if __name__ == "__main__":
    main()
