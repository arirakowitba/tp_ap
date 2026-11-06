"""Line C: Cohort Error Analysis & High-Precision Text Specialization Rules.

Analyzes the 227 fixes and 1,350 breaks of Line A vs CatBoost Bag3 on confirm 20261143
to determine whether a localized rule can safely capture the text fixes without suffering the breaks.
"""
import os, sys, json, zipfile
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAIRS_PATH = ROOT / "results" / "line_a_text" / "paired_20261143.csv"
OUT_REPORT = ROOT / "results" / "line_c_cohorts" / "report_line_c_cohort_analysis_20261143.json"
OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
ZIP_PATH = ROOT / "data_local" / "ap-2026-q-2.zip"

def main():
    print("=== LINE C: ERROR ANALYSIS & COHORT SPECIALIZATION ===", flush=True)
    df = pd.read_csv(PAIRS_PATH)
    
    # Identify fixes (Line A correct, Bag3 wrong) and breaks (Line A wrong, Bag3 correct)
    df["line_a_correct"] = (df["pred_line_a"] == df["y_true"])
    df["bag3_correct"] = (df["pred_bag3"] == df["y_true"])
    
    df["is_fix"] = df["line_a_correct"] & (~df["bag3_correct"])
    df["is_break"] = (~df["line_a_correct"]) & df["bag3_correct"]
    df["is_both_correct"] = df["line_a_correct"] & df["bag3_correct"]
    df["is_both_wrong"] = (~df["line_a_correct"]) & (~df["bag3_correct"])

    print(f"Total confirm rows: {len(df)}")
    print(f"Fixes: {df['is_fix'].sum()}, Breaks: {df['is_break'].sum()}")
    print(f"Both correct: {df['is_both_correct'].sum()}, Both wrong: {df['is_both_wrong'].sum()}")

    # Margin analysis: margin = |p - 0.5|
    df["margin_line_a"] = (df["p_line_a"] - 0.5).abs()
    df["margin_bag3"] = (df["p_bag3"] - 0.5).abs()

    # Analyze error rate as a function of Line A confidence vs Bag3 uncertainty
    # E.g. where Bag3 is uncertain (|p_bag3 - 0.5| < 0.1) and Line A is very confident (|p_line_a - 0.5| > 0.35)
    cohort_uncertain_tree = df[df["margin_bag3"] < 0.1]
    print(f"\nCohort where Bag3 is uncertain (|p_bag3 - 0.5| < 0.1): {len(cohort_uncertain_tree)} items")
    for min_margin_text in [0.2, 0.3, 0.35, 0.4]:
        sub = cohort_uncertain_tree[cohort_uncertain_tree["margin_line_a"] >= min_margin_text]
        fixes = sub["is_fix"].sum()
        breaks = sub["is_break"].sum()
        net = fixes - breaks
        print(f"  When Line A margin >= {min_margin_text}: count={len(sub)}, fixes={fixes}, breaks={breaks}, net={net:+d}")

    # Segment by seller seen vs unseen
    for seen_val in [0, 1]:
        label = "Seen Sellers" if seen_val == 1 else "New Sellers"
        sub_df = df[df["is_seen"] == seen_val]
        unc_tree = sub_df[sub_df["margin_bag3"] < 0.1]
        print(f"\n[{label}] Total: {len(sub_df)}, Uncertain Bag3: {len(unc_tree)}")
        for min_margin_text in [0.2, 0.3, 0.35]:
            sub = unc_tree[unc_tree["margin_line_a"] >= min_margin_text]
            fixes = sub["is_fix"].sum()
            breaks = sub["is_break"].sum()
            net = fixes - breaks
            print(f"  [{label}] Line A margin >= {min_margin_text}: count={len(sub)}, fixes={fixes}, breaks={breaks}, net={net:+d}")

    out_data = {
        "n_confirm": len(df),
        "n_fixes": int(df["is_fix"].sum()),
        "n_breaks": int(df["is_break"].sum()),
        "n_both_correct": int(df["is_both_correct"].sum()),
        "n_both_wrong": int(df["is_both_wrong"].sum()),
        "uncertain_tree_cohort_size": len(cohort_uncertain_tree)
    }

    with open(OUT_REPORT, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved analysis to {OUT_REPORT}")

if __name__ == "__main__":
    main()
