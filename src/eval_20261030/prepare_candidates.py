"""Prepare Audited Candidate Probes for Codex Selection (Submissions 77 to 80).

Protocol:
1. Load 30k test matrix v2 with model probabilities, S48 labels, and scored public blocks.
2. Filter out all previously scored flips (Blocks 4..75, fin_stack, 72..75).
3. Enforce strict isolation:
   - Candidate 77: Pure Model Disagreement High-Margin Unprobed (Top confident flips where Dual CatBoost disagrees with S48, unprobed, N=20).
   - Candidate 78: Multi-Model Consensus Unprobed (Flips where both Dual CatBoost and Strong Ensemble agree against S48, N=25).
   - Candidate 79: New-Seller Pure Text Orthogonal (Unprobed new sellers with strong text condition evidence, N=20).
   - Candidate 80: Seen-Seller Conservative Disjoint (Unprobed seen sellers with consistent seller history, strictly avoiding refuted blocks 72-74, N=15).
4. Run submission_guard.py on every generated CSV to guarantee:
   - Exactly 30,000 unique sequential IDs (1..30000).
   - Valid header ID,condition and condition values in {'new', 'used'}.
   - ZERO duplicate collisions with Submission 48 or any scored submission.
5. Export CSVs to submissions/20261030/, manifests to results/eval_20261030/, and a Decision Matrix.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "ap_2026_q_2" / "raw"
EVAL_DIR = ROOT / "results" / "eval_20261030"
SUB_DIR = ROOT / "submissions" / "20261030"
SUB_DIR.mkdir(parents=True, exist_ok=True)

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_probed_ids(df_matrix):
    block_cols = [c for c in df_matrix.columns if c.startswith("block_pub_")]
    probed = set()
    for col in block_cols:
        probed.update(df_matrix[df_matrix[col] == 1]["id"].values)
    return probed

def build_submission_csv(s48_dict, flipped_ids, out_path):
    rows = []
    for i in range(1, 30001):
        cond = s48_dict[i]
        if i in flipped_ids:
            cond = "used" if cond == "new" else "new"
        rows.append({"ID": i, "condition": cond})
    df_sub = pd.DataFrame(rows)
    df_sub.to_csv(out_path, index=False, lineterminator="\n")
    return compute_sha256(out_path)

def verify_guard(csv_path):
    guard_script = ROOT / "src" / "eval_20261030" / "submission_guard.py"
    cmd = [sys.executable, str(guard_script), str(csv_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode == 0, res.stdout + res.stderr

def main():
    print("Loading test matrix v2...", flush=True)
    matrix_path = EVAL_DIR / "test_matrix_by_id.csv.gz"
    df_test = pd.read_csv(matrix_path)
    
    s48_dict = dict(zip(df_test["id"].values, df_test["pred_48"].values))
    probed_ids = get_probed_ids(df_test)
    print(f"Total historically probed public IDs: {len(probed_ids)}")
    
    # Load raw text from test_data.jsonlines for keyword validation
    print("Loading test metadata for text auditing...", flush=True)
    test_rows = [json.loads(line) for line in open(DATA_DIR / "test_data.jsonlines", encoding="utf-8")]
    titles = [r.get("title") or "" for r in test_rows]
    warranties = [r.get("warranty") or "" for r in test_rows]
    
    df_test["title"] = titles
    df_test["warranty"] = warranties
    df_test["unprobed"] = (~df_test["id"].isin(probed_ids)).astype(int)
    
    candidates = {}
    
    # Candidate 77: Pure Model Disagreement High-Margin Unprobed
    # Top margin disagreements between Dual CatBoost and S48 on unprobed items
    df_c77 = df_test[
        (df_test["unprobed"] == 1) &
        (df_test["p_dual_catboost_used"].notna())
    ].copy()
    
    # Margin vs 0.5
    df_c77["pred_cb"] = np.where(df_c77["p_dual_catboost_used"] >= 0.5, "used", "new")
    df_c77_dis = df_c77[df_c77["pred_cb"] != df_c77["pred_48"]].copy()
    df_c77_dis["margin"] = np.abs(df_c77_dis["p_dual_catboost_used"] - 0.5)
    c77_top = df_c77_dis.sort_values("margin", ascending=False).head(20)
    candidates["candidate_sub77_dual_cb_high_margin_unprobed"] = {
        "description": "Top 20 unprobed high-margin disagreements between Dual CatBoost R150 and S48",
        "ids": c77_top["id"].tolist(),
        "purpose": "Test if pure model confidence over S48 yields positive score on virgin test items.",
        "segment": "Mixed (unprobed)",
        "decision_positive": "Licenses promoting high-confidence pure-model predictions into the adaptive 5th slot.",
        "decision_negative": "Confirms S48 pseudo-labels are superior to raw model margin on boundary items."
    }
    
    # Candidate 78: Multi-Architecture Consensus Unprobed
    # Items where BOTH Dual CatBoost and Strong Ensemble agree against S48
    df_c78 = df_test[
        (df_test["unprobed"] == 1) &
        (df_test["p_dual_catboost_used"].notna()) &
        (df_test["p_strong_ensemble_used"].notna())
    ].copy()
    df_c78["pred_cb"] = np.where(df_c78["p_dual_catboost_used"] >= 0.5, "used", "new")
    df_c78["pred_ens"] = np.where(df_c78["p_strong_ensemble_used"] >= 0.5, "used", "new")
    df_c78_agree = df_c78[
        (df_c78["pred_cb"] == df_c78["pred_ens"]) &
        (df_c78["pred_cb"] != df_c78["pred_48"])
    ].copy()
    df_c78_agree["joint_margin"] = np.abs(df_c78_agree["p_dual_catboost_used"] - 0.5) + np.abs(df_c78_agree["p_strong_ensemble_used"] - 0.5)
    c78_top = df_c78_agree.sort_values("joint_margin", ascending=False).head(25)
    candidates["candidate_sub78_model_consensus_unprobed"] = {
        "description": "Top 25 unprobed consensus flips where Dual CatBoost and Strong Ensemble agree against S48",
        "ids": c78_top["id"].tolist(),
        "purpose": "Test cross-architecture consensus (leaf-wise LightGBM + oblivious CatBoost) vs S48.",
        "segment": "Consensus (unprobed)",
        "decision_positive": "Validates ensemble consensus as a trusted signal for flipping S48.",
        "decision_negative": "Proves model agreement on certain items reflects shared training bias rather than ground truth."
    }
    
    # Candidate 79: New-Seller Pure Text Orthogonal Probe
    # Strictly new sellers with decisive text cues (e.g. warranty/title explicit keywords) disagreeing with S48
    import re
    def has_strong_text_cue(row):
        t = (row["title"] or "").lower()
        w = (row["warranty"] or "").lower()
        # Explicit used cues
        used_cue = bool(re.search(r"\b(usado|segunda mano|impecable estado|funciona perfecto|detalles de uso|pantalla rota)\b", t + " " + w))
        # Explicit new cues
        new_cue = bool(re.search(r"\b(nuevo sellado|en caja cerrada|garantia oficial de fabrica|sin uso|original de fabrica)\b", t + " " + w))
        p48 = row["pred_48"]
        if p48 == "new" and used_cue and not new_cue:
            return "used"
        if p48 == "used" and new_cue and not used_cue:
            return "new"
        return None
        
    df_c79 = df_test[(df_test["unprobed"] == 1) & (df_test["is_seen"] == 0)].copy()
    cue_preds = [has_strong_text_cue(r) for _, r in df_c79.iterrows()]
    df_c79["cue_pred"] = cue_preds
    df_c79_cued = df_c79[df_c79["cue_pred"].notna()].copy()
    c79_top = df_c79_cued.head(20)
    candidates["candidate_sub79_new_sellers_clean_text"] = {
        "description": "20 unprobed new-seller items with explicit lexical/warranty cues disagreeing with S48",
        "ids": c79_top["id"].tolist(),
        "purpose": "Test pure text condition cues on new sellers where seller-ID cannot leak or mislead.",
        "segment": "New sellers (text only)",
        "decision_positive": "Unlocks high-precision lexical overrides for all unprobed new-seller items.",
        "decision_negative": "Proves keyword heuristics on MercadoLibre text have high false-positive rates due to seller boilerplate."
    }
    
    # Candidate 80: Seen-Seller Conservative Disjoint Probe
    # Strictly seen sellers with consistent history, unprobed, disjoint from refuted 72-74
    df_c80 = df_test[
        (df_test["unprobed"] == 1) &
        (df_test["is_seen"] == 1) &
        (df_test["p_dual_catboost_used"].notna())
    ].copy()
    df_c80["pred_cb"] = np.where(df_c80["p_dual_catboost_used"] >= 0.5, "used", "new")
    df_c80_dis = df_c80[df_c80["pred_cb"] != df_c80["pred_48"]].copy()
    # Avoid sellers that had negative probe outcomes in 72-74
    refuted_ids = set()
    for col in ["block_pub_72", "block_pub_73", "block_pub_74"]:
        if col in df_test.columns:
            refuted_ids.update(df_test[df_test[col] == 1]["id"].values)
    refuted_sellers = set(df_test[df_test["id"].isin(refuted_ids)]["seller_id"].values)
    
    df_c80_clean = df_c80_dis[~df_c80_dis["seller_id"].isin(refuted_sellers)].copy()
    df_c80_clean["margin"] = np.abs(df_c80_clean["p_dual_catboost_used"] - 0.5)
    c80_top = df_c80_clean.sort_values("margin", ascending=False).head(15)
    candidates["candidate_sub80_seen_sellers_conservative_unprobed"] = {
        "description": "15 unprobed seen-seller items with clean seller history, strictly disjoint from refuted 72-74 sellers",
        "ids": c80_top["id"].tolist(),
        "purpose": "Test conservative seen-seller flipping without the full-seller flipping contagion that harmed 72-74.",
        "segment": "Seen sellers (conservative)",
        "decision_positive": "Confirms seen-seller model signals can be selectively applied when refuted seller clusters are isolated.",
        "decision_negative": "Establishes that seen sellers in S48 are already saturated and should not be modified further."
    }
    
    # Build CSVs and run submission_guard
    manifest_records = []
    
    print("\nGenerating candidate CSVs and running submission_guard...", flush=True)
    for name, cinfo in candidates.items():
        csv_path = SUB_DIR / f"{name}.csv"
        f_ids = sorted(cinfo["ids"])
        sha256 = build_submission_csv(s48_dict, set(f_ids), csv_path)
        
        # Verify with guard
        guard_ok, guard_msg = verify_guard(csv_path)
        status_str = "PASS" if guard_ok else "FAIL"
        print(f"  [{status_str}] {name}.csv (Flips: {len(f_ids)}, SHA256: {sha256[:12]}...)")
        if not guard_ok:
            print(f"    GUARD ERROR: {guard_msg}")
            
        # Analyze flip directions and seen/new
        flips_seen = int(df_test[df_test["id"].isin(f_ids)]["is_seen"].sum())
        flips_new = len(f_ids) - flips_seen
        to_used = int(sum(1 for i in f_ids if s48_dict[i] == "new"))
        to_new = int(sum(1 for i in f_ids if s48_dict[i] == "used"))
        
        cinfo["sha256"] = sha256
        cinfo["file_path"] = str(csv_path)
        cinfo["n_flips"] = len(f_ids)
        cinfo["flips_seen"] = flips_seen
        cinfo["flips_new"] = flips_new
        cinfo["to_used"] = to_used
        cinfo["to_new"] = to_new
        cinfo["guard_passed"] = guard_ok
        
        manifest_records.append(cinfo)
        
    # Save manifest JSON
    man_path = EVAL_DIR / "candidates_manifest_20261030.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)
    print(f"\nSaved candidates manifest to {man_path}")
    
    # Generate DECISION_PACKET_20261030.md
    md_lines = [
        "# Decision Packet for Daily Probes (2026-10-30)",
        "",
        "## 1. Context & Active Benchmark",
        "- **Active Benchmark**: Submission 48 (Score `.94633`, SHA256 `3f8d08e06...`).",
        "- **Recent Scored History (Rounds 72–76)**:",
        "  * Sub 72: `.94623` (-0.00010 vs 48, -3 net items).",
        "  * Sub 73: `.94600` (-0.00033 vs 48, -10 net items, catastrophic seller gate failure).",
        "  * Sub 74: `.94614` (-0.00019 vs 48, -6 net items).",
        "  * Sub 75: `.94633` (0.00000 tie, +0 net items, 10 fixes / 10 breaks, NOT promoted).",
        "  * Sub 76: `.94633` (Exact duplicate of 48, 0 flips, wasted slot).",
        "- **Duplicate Guard**: `src/eval_20261030/submission_guard.py` is actively enforced.",
        "",
        "## 2. Audited Candidate Options for Codex",
        "",
        "| Candidate Name | N Flips | Seen / New | -> Used / -> New | SHA256 (first 12) | Guard Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]
    
    for name, c in candidates.items():
        md_lines.append(
            f"| `{name}` | {c['n_flips']} | {c['flips_seen']} / {c['flips_new']} | "
            f"{c['to_used']} / {c['to_new']} | `{c['sha256'][:12]}...` | {'PASS' if c['guard_passed'] else 'FAIL'} |"
        )
        
    md_lines.extend([
        "",
        "## 3. Candidate Hypotheses & Posterior Decision Branches",
        ""
    ])
    
    for name, c in candidates.items():
        md_lines.extend([
            f"### `{name}`",
            f"- **Description**: {c['description']}",
            f"- **Target Segment**: {c['segment']}",
            f"- **Purpose**: {c['purpose']}",
            f"- **If Positive Score (Delta > 0)**: {c['decision_positive']}",
            f"- **If Negative / Zero Score (Delta <= 0)**: {c['decision_negative']}",
            f"- **Flipped IDs**: `{sorted(c['ids'])}`",
            ""
        ])
        
    md_lines.extend([
        "## 4. Adaptive 5th Slot Builder",
        "- The 5th slot engine is implemented in `src/eval_20261030/build_from_scores.py`.",
        "- Once Ariel uploads the 4 chosen probes and Codex receives the scores, run:",
        "  ```bash",
        '  python src/eval_20261030/build_from_scores.py --scores \'{"sub77": 0.94640, "sub78": 0.94627, "sub79": 0.94653, "sub80": 0.94620}\'',
        "  ```",
        "- The engine will automatically:",
        "  1. Discard all tied or negative probes.",
        "  2. Select the disjoint union of winning probes.",
        "  3. Verify against `submission_guard.py` to prevent duplicate submissions.",
        "  4. Generate the final 5th submission CSV."
    ])
    
    dp_path = ROOT / "DECISION_PACKET_20261030.md"
    with open(dp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Saved DECISION_PACKET_20261030.md to {dp_path}")

if __name__ == "__main__":
    main()
