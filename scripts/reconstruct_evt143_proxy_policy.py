"""Agente A & B: Reconstruct High-Fidelity OOF & Test Policy Proxy for EVT-143.

Lineage & Rule Decomposition:
1. Core Multi-Modal Ensemble:
   - High-capacity TF-IDF Text (Word 1-2 + Char 3-5, SVD 120, Logistic Regression)
   - Tabular LightGBM (Price, rel_price, title lengths, seller LOO)
   - Blended probability p_core.
2. Lexical Domain Overrides (derived from EVT-132/EVT-143 historical winning rules):
   - Strong used cues in title:
     ['para repuesto', 'para repuestos', 'para desarme', 'para piezas', 'no funciona', 'usado', 'usados', 'segunda mano', 'antiguo', 'antigua', 'vintage', 'impecable estado', 'muy buen estado', 'canje', 'permuto']
     -> If p_core in [0.25, 0.50] and has strong used cue -> predict 'used'.
   - Strong new cues in title:
     ['nuevo en caja', 'cerrado en caja', 'sellado', 'cerrado de fabrica', '100% nuevo', 'sin uso', 'con garantia', 'factura a', 'factura b', '0km', 'sellada']
     -> If p_core in [0.50, 0.75] and has strong new cue -> predict 'new'.
3. Category-specific hygiene / safety priors:
   - Piercings, lingerie, pacifiers, breast pumps, baby bottles -> bias to 'new' when p_core > 0.40.
4. Evaluation:
   - Measure agreement fraction between y_evt_proxy and EVT-143 on test (30,000 rows).
   - Evaluate cross-fitted OOF accuracy on 70,000 train rows across all 5 outer seller-group folds.
"""

import sys
import json
import time
import hashlib
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]

STRONG_USED_CUES = [
    "repuesto", "repuestos", "desarme", "piezas", "no funciona", "usado", "usados",
    "segunda mano", "antiguo", "antigua", "vintage", "impecable estado", "muy buen estado",
    "canje", "permuto", "año 19", "ano 19", "año 200", "ano 200"
]

STRONG_NEW_CUES = [
    "nuevo en caja", "cerrado en caja", "sellado", "cerrada", "cerrado de fabrica",
    "100% nuevo", "sin uso", "con garantia", "factura a", "factura b", "0km", "sellada"
]

def has_cue(title, cues):
    t_low = title.lower()
    return any(c in t_low for c in cues)

def apply_evt_rules(p_core, titles, categories):
    preds = np.where(p_core >= 0.5, "used", "new")
    
    n_modified = 0
    flips_to_used = 0
    flips_to_new = 0

    for i in range(len(p_core)):
        t = titles[i]
        p = p_core[i]

        # Rule 1: Strong used cues in borderline new items
        if 0.20 <= p < 0.50 and has_cue(t, STRONG_USED_CUES):
            preds[i] = "used"
            flips_to_used += 1
            n_modified += 1

        # Rule 2: Strong new cues in borderline used items
        elif 0.50 <= p <= 0.80 and has_cue(t, STRONG_NEW_CUES):
            preds[i] = "new"
            flips_to_new += 1
            n_modified += 1

    return preds, n_modified, flips_to_used, flips_to_new

def main():
    print("=" * 80)
    print("RECONSTRUCTING EVT-143 PROXY POLICY (AGENTE A & B)")
    print("=" * 80)

    splits_path = ROOT / "data/splits/canonical_nested_dual_regime_splits.csv"
    train_path = ROOT / "data/ap_2026_q_2/raw/train_data.jsonlines"
    test_path = ROOT / "data/ap_2026_q_2/raw/test_data.jsonlines"
    evt143_path = ROOT / "submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv"
    podium_oof_path = ROOT / "results/ultimate_podium_engine_23h/oof_predictions_podium_engine.csv"
    podium_test_path = ROOT / "results/ultimate_podium_engine_23h/test_probabilities_podium_engine.csv"

    df_splits = pd.read_csv(splits_path)
    df_evt143 = pd.read_csv(evt143_path)
    df_podium_oof = pd.read_csv(podium_oof_path)
    df_podium_test = pd.read_csv(podium_test_path)

    with open(train_path, "r", encoding="utf-8") as f:
        train_rows = [json.loads(line) for line in f]
    with open(test_path, "r", encoding="utf-8") as f:
        test_rows = [json.loads(line) for line in f]

    y_train = np.array([1 if r.get("condition") == "used" else 0 for r in train_rows], dtype=int)
    tr_titles = [r.get("title") or "" for r in train_rows]
    te_titles = [r.get("title") or "" for r in test_rows]
    tr_cats = [str(r.get("category_id") or "") for r in train_rows]
    te_cats = [str(r.get("category_id") or "") for r in test_rows]

    # Test baseline and EVT-143
    p_base_test = df_podium_test["p_optimal_blend"].values
    y_evt143 = df_evt143["condition"].values

    # Apply reconstructed rules to test base probabilities
    preds_test_proxy, n_mod_te, f_u_te, f_n_te = apply_evt_rules(p_base_test, te_titles, te_cats)

    # Agreement with EVT-143
    agree_raw_base = np.sum(np.where(p_base_test >= 0.5, "used", "new") == y_evt143)
    agree_proxy = np.sum(preds_test_proxy == y_evt143)

    print(f"\n[Test Set Fidelity Audit]")
    print(f"  Raw Baseline Agreement with EVT-143:       {agree_raw_base} / 30000 ({agree_raw_base/30000*100:.2f}%)")
    print(f"  Reconstructed Policy Agreement with EVT-143:{agree_proxy} / 30000 ({agree_proxy/30000*100:.2f}%)")
    print(f"  Net Disagreement Reduction:                {agree_proxy - agree_raw_base} items closer to EVT-143!")
    print(f"  Flips Applied by Reconstructed Rules:      {n_mod_te} (to_used={f_u_te}, to_new={f_n_te})")

    # Now evaluate on Train OOF across 5 folds
    p_base_train = df_podium_oof["oof_blend"].values
    preds_train_proxy, n_mod_tr, f_u_tr, f_n_tr = apply_evt_rules(p_base_train, tr_titles, tr_cats)
    bin_train_proxy = np.where(preds_train_proxy == "used", 1, 0)
    bin_base_train = (p_base_train >= 0.5).astype(int)

    acc_base = np.mean(bin_base_train == y_train)
    acc_proxy = np.mean(bin_train_proxy == y_train)
    fixes_proxy = np.sum((bin_train_proxy == y_train) & (bin_base_train != y_train))
    breaks_proxy = np.sum((bin_train_proxy != y_train) & (bin_base_train == y_train))
    net_proxy = fixes_proxy - breaks_proxy

    print(f"\n[Train OOF Performance Audit (70,000 rows)]")
    print(f"  Baseline Podium OOF Accuracy:              {acc_base*100:.3f}% ({int(acc_base*70000)} hits)")
    print(f"  Reconstructed EVT-Proxy OOF Accuracy:      {acc_proxy*100:.3f}% ({int(acc_proxy*70000)} hits)")
    print(f"  Rule Fixes: {fixes_proxy}, Breaks: {breaks_proxy}, Net Gain: {net_proxy:+d}")
    print(f"  Rule Precision:                            {fixes_proxy / (fixes_proxy + breaks_proxy)*100:.2f}%")

    # Residual Error Target Definition
    is_proxy_error = (bin_train_proxy != y_train)
    n_proxy_errors = np.sum(is_proxy_error)
    print(f"\n[Residual Target Definition]")
    print(f"  Total Residual Errors to Predict:          {n_proxy_errors} / 70000 ({n_proxy_errors/70000*100:.2f}%)")
    print(f"  - Residual U2N errors (proxy=used, true=new): {np.sum(is_proxy_error & (bin_train_proxy == 1))}")
    print(f"  - Residual N2U errors (proxy=new, true=used): {np.sum(is_proxy_error & (bin_train_proxy == 0))}")

    # Save reconstructed proxy OOF and test
    out_dir = ROOT / "results/evt143_reconstructed_policy"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_proxy_oof = pd.DataFrame({
        "id": df_splits["id"],
        "outer_unseen_fold": df_splits["outer_unseen_fold"],
        "y_true": y_train,
        "p_base": p_base_train,
        "pred_evt_proxy": preds_train_proxy,
        "is_proxy_error": is_proxy_error.astype(int)
    })
    df_proxy_oof.to_csv(out_dir / "oof_evt143_proxy_policy.csv", index=False)

    df_proxy_test = pd.DataFrame({
        "ID": df_evt143["ID"],
        "pred_evt_proxy": preds_test_proxy,
        "condition_evt143": y_evt143
    })
    df_proxy_test.to_csv(out_dir / "test_evt143_proxy_policy.csv", index=False)

    print(f"\nReconstructed Policy saved to {out_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
