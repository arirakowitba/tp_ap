"""Analyze whether feature-conditioned subsets can achieve q_bar > 0.80 for 150+ items.
Tests interaction of Quad Consensus with explicit lexical cues and seller signals.
"""

import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(".")
ANCHOR_PATH = ROOT / "results/canonical_ledger_rebuild_20261108/evaluated_vectors_git_pinned/EVT-132_23ff472_11aabc76.csv"
TEST_JSONLINES = ROOT / "data/ap_2026_q_2/raw/test_data.jsonlines"

def main():
    print("=== HIGH-PRECISION SUBSET OPTIMIZATION ===")
    
    # Load test metadata
    test_rows = []
    with open(TEST_JSONLINES, "r", encoding="utf-8") as f:
        for line in f:
            test_rows.append(json.loads(line))
            
    df_anchor = pd.read_csv(ANCHOR_PATH)
    anchor_preds = df_anchor["condition"].values
    
    # Load models
    p_lgb = np.load("results/expanded_feedback_lgbm_5fold/test_probabilities.npy")
    p_xgb = np.load("results/expanded_feedback_xgb_5fold/test_probabilities.npy")
    p_cat = np.load("results/catboost_depth8_5fold/test_probabilities.npy")
    p_mlp = np.load("results/mlp_neural_5fold/test_probabilities.npy")
    
    # Domain shield
    NEW_REGEX = re.compile(
        r"\b(a estrenar|0km|0 km|sin uso|sellad[oa]s?|en blister|cerrad[oa] en caja|impecable sin uso|nuevo sellado|caja cerrada|nuevo sin uso)\b",
        re.IGNORECASE,
    )
    confirmed_true_new = {6308, 16515, 24735, 27016}
    
    # Positive USED cues
    USED_REGEX = re.compile(
        r"\b(como nuev[oa]s?|excelente estado|buen estado|muy buen estado|impecable estado|segunda mano|usad[oa]s?|antigu[oa]s?|antiguedades|vintage|retro|lote de|colecci[oó]n de|a reparar|para repuesto|despiece|permuto|canje)\b",
        re.IGNORECASE
    )
    
    # Extract cues for test items
    has_pos_cue = np.zeros(30000, dtype=bool)
    is_safe = np.ones(30000, dtype=bool)
    for idx, r in enumerate(test_rows):
        item_id = idx + 1
        if item_id in confirmed_true_new:
            is_safe[idx] = False
            continue
        t = str(r.get("title") or "")
        w = str(r.get("warranty") or "")
        full = f"{t} {w}"
        if NEW_REGEX.search(full):
            is_safe[idx] = False
        if USED_REGEX.search(full):
            has_pos_cue[idx] = True
            
    # Consensus scores
    consensus = (p_lgb + p_xgb + p_cat + p_mlp) / 4.0
    
    # Filter candidates
    virgin_mask = (anchor_preds == "new") & is_safe
    
    # Test 3 ranking methods:
    # A: Pure consensus
    # B: Consensus boosted by positive cue (+0.10)
    # C: Consensus gated (must have consensus > 0.60 AND positive cue)
    
    ranks = {
        "A_Pure_Consensus": consensus,
        "B_Cue_Boosted_Consensus": consensus + 0.10 * has_pos_cue.astype(float),
        "C_Multiplicative_Agreement": consensus * (1.0 + 0.25 * has_pos_cue.astype(float)),
    }
    
    print("\nCandidate counts with score > 0.50:")
    for name, score_arr in ranks.items():
        m = virgin_mask & (score_arr > 0.50)
        cands_cue = m & has_pos_cue
        print(f"  {name:<30}: {m.sum()} candidates (of which {cands_cue.sum()} have explicit USED cues)")
        
if __name__ == "__main__":
    main()
