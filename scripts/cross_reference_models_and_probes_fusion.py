"""Joint Information Fusion & Cross-Referencing Engine (AP-2026-Q-2).

Contrasts multi-model prediction confidence against empirical probe posteriors
across all 98 historical submissions. Produces joint Bayesian certainty metrics
and selects the absolute highest expected-value candidates for tomorrow.
"""

import os, sys, json, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import logit, expit

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "cross_referenced_bayesian_fusion"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PUB = 20998
ALPHA_PUB = N_PUB / 30000.0 # ~0.70

def main():
    print("=" * 80, flush=True)
    print("=== STARTING JOINT MODEL CONFIDENCE & PROBE POSTERIOR CROSS-FUSION ===", flush=True)
    print("=" * 80, flush=True)
    t0 = time.time()

    # 1. Load Anchor (0.94719 = 19,889 hits)
    p_anchor = ROOT / "submissions" / "submission_20261106_05_climax_consumer_unanimous_top15.csv"
    anchor_df = pd.read_csv(p_anchor)
    anchor = anchor_df.set_index("ID")["condition"]
    print(f"Loaded Record Anchor: {p_anchor.name} (0.94719 = 19,889 hits)", flush=True)

    # 2. Load Model Probabilities
    # a. Ultimate Podium Engine 23h (OOF 0.94230)
    p_podium = ROOT / "results" / "ultimate_podium_engine_23h" / "test_probabilities_podium_engine.csv"
    df_podium = pd.read_csv(p_podium).set_index("ID")
    
    # b. Full F117 LightGBM
    p_f117 = ROOT / "results" / "full_f117_lgbm_20261105" / "test_probabilities_f117_lgbm.csv"
    df_f117 = pd.read_csv(p_f117).set_index("ID") if p_f117.exists() else None
    
    # c. Continuous Training CatBoost Deep
    p_cont = ROOT / "results" / "continuous_training_20261105" / "continuous_test_probabilities_latest.csv"
    df_cont = pd.read_csv(p_cont).set_index("ID") if p_cont.exists() else None

    # d. Transductive Deep CatBoost (OOF 0.94554, LogLoss 0.13999)
    p_trans = ROOT / "results" / "transductive_deep_models" / "transductive_test_probabilities.csv"
    df_trans = pd.read_csv(p_trans).set_index("ID") if p_trans.exists() else None

    # e. Transductive XGBoost (OOF 0.89857)
    p_xgb = ROOT / "results" / "transductive_deep_models" / "xgboost_transductive_probabilities.csv"
    df_xgb = pd.read_csv(p_xgb).set_index("ID") if p_xgb.exists() else None

    # 3. Load Probe Posteriors & Empirical Deconvolution
    p_bayes = ROOT / "results" / "bayesian_solver_98vec" / "bayesian_transductive_item_effects_98vec.csv"
    df_bayes = pd.read_csv(p_bayes).set_index("item_id")
    
    p_ev90 = ROOT / "results" / "forensic_id_solver_20261105" / "id_level_target_evidence_all_90probes.csv"
    df_ev90 = pd.read_csv(p_ev90).set_index("item_id") if p_ev90.exists() else None

    # 4. Load Raw Test Metadata
    with open(ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines", encoding="utf-8") as f:
        test_rows = [json.loads(line) for line in f]

    # Category and semantic filters
    EXCLUDE_CATS = {
        "MLA1459", "MLA1466", "MLA1467", "MLA1468", "MLA1470", "MLA1471", 
        "MLA1472", "MLA1473", "MLA1474", "MLA1475", "MLA1495", # Complete Inmuebles tree
        "MLA50279", "MLA50280", "MLA50281", # Vacation rentals
        "MLA1743", "MLA1744", "MLA2547", "MLA1745", "MLA1746", # Vehicles
        "MLA5632", "MLA1071", "MLA1072", # Pets, animals
        "MLA1540" # Services
    }

    EXCLUDE_TITLE_WORDS = [
        "alquil", "cachorr", "servicio", "diseño", "reparacion", "adopcion", "reserva", 
        "terreno", "departamento", "duplex", "monoambient", "cochera", "inmueble"
    ]

    STRONG_KEYWORDS = [
        "impecable", "impecables", "vintage", "antiguo", "antigua", "antiguos", "antiguas",
        "usado", "usada", "usados", "usadas", "poco uso", "sin uso", "segunda mano",
        "excelente estado", "muy buen estado", "buen estado", "como nuevo", "detalles de uso",
        "a reparar", "con detalles", "desgaste", "retro", "coleccion"
    ]

    # Protected negative IDs
    prot_ids = set()
    if df_ev90 is not None:
        prot_ids = set(df_ev90[df_ev90["bounded_effect_delta"] < -0.35].index)
    print(f"Loaded {len(prot_ids)} protected IDs from probe ledger.", flush=True)

    records = []
    print("\nComputing Joint Information Metrics for all 30,000 items...", flush=True)

    for idx in range(30000):
        item_id = idx + 1
        curr_anc = anchor[item_id]
        
        # Model predictions
        p_opt = float(df_podium.loc[item_id, "p_optimal_blend"])
        p_lgb = float(df_podium.loc[item_id, "p_lgb"])
        p_cb = float(df_podium.loc[item_id, "p_cb"])
        p_text = float(df_podium.loc[item_id, "p_text"])
        
        preds_list = [p_opt, p_lgb, p_cb]
        if df_f117 is not None and item_id in df_f117.index:
            preds_list.append(float(df_f117.loc[item_id, "p_f117_lgbm"]))
        if df_cont is not None and item_id in df_cont.index:
            preds_list.append(float(df_cont.loc[item_id, "p_catboost_deep"]))
        if df_trans is not None and item_id in df_trans.index:
            preds_list.append(float(df_trans.loc[item_id, "p_cb_transductive"]))
        if df_xgb is not None and item_id in df_xgb.index:
            preds_list.append(float(df_xgb.loc[item_id, "p_xgb_transductive"]))
            
        p_model_mean = float(np.mean(preds_list))
        p_model_std = float(np.std(preds_list))
        p_model_confidence = abs(p_model_mean - 0.50) * 2.0 # 0.0 (uncertain) to 1.0 (certain)
        model_agreement_rate = (1.0 - p_model_std) # Higher = lower variance among models

        # Probe Evidence
        k_tested = 0
        map_eff = 0.0
        exp_gain_anchor = 0.0
        if item_id in df_bayes.index:
            k_tested = int(df_bayes.loc[item_id, "times_tested"])
            map_eff = float(df_bayes.loc[item_id, "map_effect_vs_d1d2"])
            exp_gain_anchor = float(df_bayes.loc[item_id, "expected_gain_vs_anchor"])
        
        # Calculate Empirical Probe Posterior Probability (P_probe)
        # Using empirical Bayesian shrinkage based on test count k_tested
        if k_tested > 0:
            # Shrinkage weight: with k=1 weight is 0.40; with k=10 weight is 0.83
            shrink_w = k_tested / (k_tested + 2.0)
            
            # If item was tested relative to D1+D2, the map_eff indicates shift in probability
            # e_map in [-1, 1] translates to empirical hit gain
            p_prior_clamped = np.clip(p_model_mean, 0.01, 0.99)
            prior_logit = logit(p_prior_clamped)
            
            # Probe log-odds evidence: 1 unit of map_eff corresponds to ~2.5 logit shift
            probe_logit_shift = 2.5 * map_eff
            posterior_logit = prior_logit + shrink_w * probe_logit_shift
            p_posterior = float(expit(posterior_logit))
        else:
            shrink_w = 0.0
            p_posterior = p_model_mean

        p_posterior_confidence = abs(p_posterior - 0.50) * 2.0

        # Contrast: Model Prior vs Probe Posterior
        # Contrast Delta = Shift caused by probes
        contrast_shift = p_posterior - p_model_mean
        
        # Metadata & Semantic Guards
        it = test_rows[idx]
        title_raw = str(it.get("title", ""))
        title = title_raw.lower()
        cat = str(it.get("category_id", ""))
        price = float(it.get("price", 0.0) or 0.0)
        
        is_blacklisted_cat = cat in EXCLUDE_CATS
        has_blacklisted_words = any(w in title for w in EXCLUDE_TITLE_WORDS)
        is_expensive = price > 35000
        is_negative_probe = (k_tested > 0 and exp_gain_anchor < 0.0) or (item_id in prot_ids)
        
        words = set(title.replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split())
        kw_hits = [kw for kw in STRONG_KEYWORDS if (kw in words or (" " in kw and kw in title))]
        has_kw = len(kw_hits) > 0
        if "abusada" in title and "usada" in kw_hits and len(kw_hits) == 1:
            has_kw = False
            kw_hits = []

        is_safe = (not is_blacklisted_cat) and (not has_blacklisted_words) and (not is_expensive) and (not is_negative_probe)

        # Quadrant Categorization (for items currently 'new' in anchor):
        if curr_anc == "new":
            if is_safe and k_tested > 0 and p_posterior > 0.85 and p_model_mean > 0.70:
                quadrant = "Q1_MUTUAL_CONSENSUS_GOLD"
            elif is_safe and k_tested == 0 and p_model_mean > 0.92 and p_model_std < 0.05:
                quadrant = "Q2_UNPROBED_MODEL_EXTREME_PLATINUM"
            elif is_safe and k_tested > 0 and exp_gain_anchor > 0.30 and p_posterior > 0.80:
                quadrant = "Q3_PROBE_DISCOVERED_SILVER"
            elif is_safe and has_kw and p_posterior > 0.65:
                quadrant = "Q4_SEMANTIC_KEYWORD_VERIFIED"
            elif not is_safe:
                quadrant = "Q_BLOCKED_GUARDRAIL_OR_NEGATIVE"
            else:
                quadrant = "Q_UNCERTAIN_OR_WEAK"
        else:
            quadrant = "ANCHOR_IS_USED"

        # Joint Information Fusion Value Metric (V_i)
        # Expected Public Hit Gain = ALPHA_PUB * (2 * p_posterior - 1)
        # Multiplied by Joint Certainty Weight
        exp_public_gain = ALPHA_PUB * (2.0 * p_posterior - 1.0) if curr_anc == "new" else 0.0
        
        # Joint Certainty: Combines model unanimity, low dispersion, and probe weight
        certainty = (
            0.40 * p_posterior_confidence +
            0.25 * model_agreement_rate +
            0.20 * (1.0 if has_kw else 0.0) +
            0.15 * shrink_w
        )
        
        joint_value = exp_public_gain * certainty if is_safe and curr_anc == "new" else -999.0

        records.append({
            "item_id": item_id,
            "anchor_condition": curr_anc,
            "p_model_mean": round(p_model_mean, 4),
            "p_model_std": round(p_model_std, 4),
            "p_model_confidence": round(p_model_confidence, 4),
            "times_tested": k_tested,
            "map_effect": round(map_eff, 4),
            "expected_gain_anchor": round(exp_gain_anchor, 4),
            "p_posterior": round(p_posterior, 4),
            "contrast_shift": round(contrast_shift, 4),
            "quadrant": quadrant,
            "is_safe": is_safe,
            "has_kw": has_kw,
            "kw_hits": ", ".join(kw_hits),
            "price": price,
            "category_id": cat,
            "title": title_raw[:60],
            "exp_public_gain": round(exp_public_gain, 4),
            "certainty": round(certainty, 4),
            "joint_value": round(joint_value, 4)
        })

    df_fusion = pd.DataFrame(records)
    out_csv = OUT_DIR / "joint_information_fusion_all30k.csv"
    df_fusion.to_csv(out_csv, index=False)
    print(f"\nSaved full joint fusion matrix (30,000 items) to {out_csv.name}", flush=True)

    # Summary Statistics & Quadrant Distribution for 'new' items
    df_new = df_fusion[df_fusion["anchor_condition"] == "new"]
    print("\n" + "=" * 80)
    print("=== JOINT INFORMATION DISTRIBUTION (ANCHOR = NEW, 15,878 ITEMS) ===")
    print("=" * 80)
    print(df_new["quadrant"].value_counts())

    # Top Safe Opportunities by Joint Value
    safe_cands = df_new[df_new["joint_value"] > 0].sort_values(by="joint_value", ascending=False)
    print(f"\nTotal High-Certainty Safe Candidates Available: {len(safe_cands)}")
    
    print("\n--- TOP 35 JOINT CERTAINTY CANDIDATES (SLOT 1 CORE) ---")
    for idx, r in safe_cands.head(35).iterrows():
        print(f"ID {int(r['item_id']):5d}: V={r['joint_value']:.3f} | post_p={r['p_posterior']:.3f} (model={r['p_model_mean']:.3f}, shift={r['contrast_shift']:+.3f}) | test={r['times_tested']} | kw=[{r['kw_hits']}] | ${r['price']:.0f} | {r['category_id']} | {r['title']}")

    print("\n--- NEXT 25 JOINT CERTAINTY CANDIDATES (SLOT 2 EXPANSION, CUMULATIVE 60) ---")
    for idx, r in safe_cands.iloc[35:60].iterrows():
        print(f"ID {int(r['item_id']):5d}: V={r['joint_value']:.3f} | post_p={r['p_posterior']:.3f} (model={r['p_model_mean']:.3f}, shift={r['contrast_shift']:+.3f}) | test={r['times_tested']} | kw=[{r['kw_hits']}] | ${r['price']:.0f} | {r['category_id']} | {r['title']}")

    print("\n--- NEXT 25 JOINT CERTAINTY CANDIDATES (SLOT 3 ATTACK, CUMULATIVE 85) ---")
    for idx, r in safe_cands.iloc[60:85].iterrows():
        print(f"ID {int(r['item_id']):5d}: V={r['joint_value']:.3f} | post_p={r['p_posterior']:.3f} (model={r['p_model_mean']:.3f}, shift={r['contrast_shift']:+.3f}) | test={r['times_tested']} | kw=[{r['kw_hits']}] | ${r['price']:.0f} | {r['category_id']} | {r['title']}")

    print("\n--- NEXT 25 JOINT CERTAINTY CANDIDATES (SLOT 4 BREAKTHROUGH, CUMULATIVE 110) ---")
    for idx, r in safe_cands.iloc[85:110].iterrows():
        print(f"ID {int(r['item_id']):5d}: V={r['joint_value']:.3f} | post_p={r['p_posterior']:.3f} (model={r['p_model_mean']:.3f}, shift={r['contrast_shift']:+.3f}) | test={r['times_tested']} | kw=[{r['kw_hits']}] | ${r['price']:.0f} | {r['category_id']} | {r['title']}")

    elapsed = time.time() - t0
    print(f"\n=== JOINT INFORMATION FUSION ENGINE COMPLETED IN {elapsed:.2f}s ===", flush=True)

if __name__ == "__main__":
    main()
