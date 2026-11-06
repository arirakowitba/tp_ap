import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".")
# 1. Load Anchor (0.94719)
p_anchor = ROOT / "submissions" / "submission_20261106_05_climax_consumer_unanimous_top15.csv"
anchor = pd.read_csv(p_anchor).set_index("ID")["condition"]

# 2. Load Model Probabilities from Ultimate Podium Engine (OOF 0.94230)
p_podium = ROOT / "results" / "ultimate_podium_engine_23h" / "test_probabilities_podium_engine.csv"
df_probs = pd.read_csv(p_podium).set_index("ID")

# 3. Load Test Metadata
with open(ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines", encoding="utf-8") as f:
    test_rows = [json.loads(line) for line in f]

# 4. Load Bayesian Solver Effects
p_bayes = ROOT / "results" / "bayesian_solver_98vec" / "bayesian_transductive_item_effects_98vec.csv"
df_bayes = pd.read_csv(p_bayes).set_index("item_id")

# Categories to strictly exclude from new -> used flips (Real estate, vehicles, services)
EXCLUDE_CATS = {
    "MLA1468", "MLA1474", "MLA1495", "MLA1459", "MLA1466", "MLA1467", # Real estate
    "MLA1743", "MLA1744", "MLA2547", # Cars, vehicles, boats
}

USED_STRONG_KEYWORDS = [
    "impecable", "impecables", "vintage", "antiguo", "antigua", "antiguos", "antiguas",
    "usado", "usada", "usados", "usadas", "poco uso", "sin uso", "segunda mano",
    "excelente estado", "muy buen estado", "buen estado", "como nuevo", "detalles de uso",
    "a reparar", "con detalles", "desgaste", "retro", "coleccion", "ano 19", "anos 70", "anos 80", "anos 90"
]

candidates = []
for idx in range(30000):
    item_id = idx + 1
    curr_lbl = anchor[item_id]
    
    # We only examine items currently predicted as 'new'
    if curr_lbl != "new":
        continue
        
    it = test_rows[idx]
    title = str(it.get("title", "")).lower()
    cat = str(it.get("category_id", ""))
    price = float(it.get("price", 0.0) or 0.0)
    
    # Exclude real estate and vehicles
    if cat in EXCLUDE_CATS:
        continue
    if price > 50000: # Expensive cars / property reservations
        continue
        
    # Check model probabilities
    p_blend = float(df_probs.loc[item_id, "p_optimal_blend"])
    p_lgb = float(df_probs.loc[item_id, "p_lgb"])
    p_cb = float(df_probs.loc[item_id, "p_cb"])
    p_text = float(df_probs.loc[item_id, "p_text"])
    
    # Check Bayesian solver
    b_gain = float(df_bayes.loc[item_id, "expected_gain_vs_anchor"]) if item_id in df_bayes.index else 0.0
    times_tested = int(df_bayes.loc[item_id, "times_tested"]) if item_id in df_bayes.index else 0
    
    # Keyword detection
    kw_hits = [w for w in USED_STRONG_KEYWORDS if w in title]
    has_strong_kw = len(kw_hits) > 0
    
    # Model unanimous for 'used'
    models_unanimous_used = (p_blend > 0.50 and p_lgb > 0.50 and p_cb > 0.50)
    
    # Extremely strong candidate criteria
    if models_unanimous_used or has_strong_kw or (p_blend > 0.70):
        candidates.append({
            "item_id": item_id,
            "title": it.get("title", "")[:60],
            "price": price,
            "category_id": cat,
            "p_blend": round(p_blend, 4),
            "p_lgb": round(p_lgb, 4),
            "p_cb": round(p_cb, 4),
            "p_text": round(p_text, 4),
            "b_gain": round(b_gain, 4),
            "times_tested": times_tested,
            "has_strong_kw": has_strong_kw,
            "kw_hits": ", ".join(kw_hits),
            "unanimous": models_unanimous_used
        })

df_c = pd.DataFrame(candidates)
print(f"Total potential 'new -> used' candidates identified: {len(df_c)}")

# Tier 1: Models unanimous (p_blend > 0.5, p_lgb > 0.5, p_cb > 0.5) AND contains strong keyword
tier1 = df_c[df_c["unanimous"] & df_c["has_strong_kw"]].sort_values(by="p_blend", ascending=False)
print(f"Tier 1 (Unanimous Models + Explicit Strong Keyword): {len(tier1)}")

# Tier 2: Models unanimous (p_blend > 0.60, p_lgb > 0.60, p_cb > 0.60) even without keyword
tier2 = df_c[df_c["unanimous"] & (~df_c["has_strong_kw"]) & (df_c["p_blend"] > 0.65)].sort_values(by="p_blend", ascending=False)
print(f"Tier 2 (Unanimous Models p > 0.65, Consumer Goods): {len(tier2)}")

# Tier 3: Strong Keyword (impecable, vintage, etc.) with p_blend > 0.40
tier3 = df_c[(~df_c["unanimous"]) & df_c["has_strong_kw"] & (df_c["p_blend"] > 0.40)].sort_values(by="p_blend", ascending=False)
print(f"Tier 3 (Strong Keyword + p_blend > 0.40): {len(tier3)}")

out_csv = ROOT / "results" / "new_to_used_candidates_deep_audit.csv"
df_c.to_csv(out_csv, index=False)
print(f"Saved full candidates to {out_csv}")
