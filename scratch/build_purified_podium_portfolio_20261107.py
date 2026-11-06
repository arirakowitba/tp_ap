import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".")
# 1. Load Anchor (0.94719 = 19,889 hits)
p_anchor = ROOT / "submissions" / "submission_20261106_05_climax_consumer_unanimous_top15.csv"
anchor_df = pd.read_csv(p_anchor)
anchor = anchor_df.set_index("ID")["condition"]

# 2. Load Model Probabilities (0.94230 Multi-Modal Ensemble)
p_podium = ROOT / "results" / "ultimate_podium_engine_23h" / "test_probabilities_podium_engine.csv"
df_probs = pd.read_csv(p_podium).set_index("ID")

# 3. Load Test Metadata
with open(ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines", encoding="utf-8") as f:
    test_rows = [json.loads(line) for line in f]

# 4. Load Bayesian Solver Effects
p_bayes = ROOT / "results" / "bayesian_solver_98vec" / "bayesian_transductive_item_effects_98vec.csv"
df_bayes = pd.read_csv(p_bayes).set_index("item_id")

# Categories to strictly exclude
EXCLUDE_CATS = {
    "MLA1468", "MLA1474", "MLA1495", "MLA1459", "MLA1466", "MLA1467", # Real estate
    "MLA50279", "MLA50280", "MLA50281", # Vacation rentals
    "MLA1743", "MLA1744", "MLA2547", "MLA1745", "MLA1746", # Vehicles
    "MLA5632", "MLA1071", "MLA1072", # Pets, animals
    "MLA1540" # Services
}

EXCLUDE_TITLE_WORDS = [
    "alquiler", "cachorr", "servicio", "diseño", "reparacion", "adopcion", "reserva", "terreno", "departamento", "duplex"
]

# Protected IDs from historical negative evidence
df_ev = pd.read_csv(ROOT / "results" / "forensic_id_solver_20261105" / "id_level_target_evidence_all_90probes.csv")
prot_ids = set(df_ev[df_ev["bounded_effect_delta"] < -0.35]["item_id"])

# Condition keywords
STRONG_KEYWORDS = [
    "impecable", "impecables", "vintage", "antiguo", "antigua", "antiguos", "antiguas",
    "usado", "usada", "usados", "usadas", "poco uso", "sin uso", "segunda mano",
    "excelente estado", "muy buen estado", "buen estado", "como nuevo", "detalles de uso",
    "a reparar", "con detalles", "desgaste", "retro", "coleccion"
]

candidates = []
for idx in range(30000):
    item_id = idx + 1
    curr_lbl = anchor[item_id]
    
    # Strictly examine items currently labeled 'new' in anchor
    if curr_lbl != "new":
        continue
    if item_id in prot_ids:
        continue
        
    it = test_rows[idx]
    title_raw = str(it.get("title", ""))
    title = title_raw.lower()
    cat = str(it.get("category_id", ""))
    price = float(it.get("price", 0.0) or 0.0)
    
    if cat in EXCLUDE_CATS:
        continue
    if price > 35000: # Protect against expensive vehicles/property
        continue
    if any(w in title for w in EXCLUDE_TITLE_WORDS):
        continue
        
    p_b = float(df_probs.loc[item_id, "p_optimal_blend"])
    p_l = float(df_probs.loc[item_id, "p_lgb"])
    p_c = float(df_probs.loc[item_id, "p_cb"])
    p_t = float(df_probs.loc[item_id, "p_text"])
    
    b_gain = float(df_bayes.loc[item_id, "expected_gain_vs_anchor"]) if item_id in df_bayes.index else 0.0
    times_tested = int(df_bayes.loc[item_id, "times_tested"]) if item_id in df_bayes.index else 0
    
    # Exclude any item with negative historical Bayesian evidence
    if times_tested > 0 and b_gain < 0.0:
        continue
    
    # Check whole word keywords
    words = set(title.replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split())
    kw_hits = [kw for kw in STRONG_KEYWORDS if (kw in words or (" " in kw and kw in title))]
    has_kw = len(kw_hits) > 0
    
    # Filter out false matches like "revista abusada"
    if "abusada" in title and "usada" in kw_hits and len(kw_hits) == 1:
        has_kw = False
        kw_hits = []
        
    models_unanimous = (p_b > 0.60 and p_l > 0.50 and p_c > 0.50)
    
    if not (models_unanimous or (has_kw and p_b > 0.45)):
        continue
        
    # Composite score prioritizing unanimous models, positive bayesian evidence, and semantic keywords
    # Normalize components
    norm_prob = (p_b - 0.50) / 0.50 # 0.0 to 1.0
    norm_gain = max(0.0, min(1.0, b_gain)) if times_tested > 0 else 0.50 # neutral if not tested
    kw_bonus = 1.0 if has_kw else 0.0
    
    composite_rank = 0.50 * norm_prob + 0.30 * norm_gain + 0.20 * kw_bonus
    
    candidates.append({
        "item_id": item_id,
        "title": title_raw[:60],
        "category_id": cat,
        "price": price,
        "p_blend": round(p_b, 4),
        "p_lgb": round(p_l, 4),
        "p_cb": round(p_c, 4),
        "b_gain": round(b_gain, 4),
        "times_tested": times_tested,
        "has_kw": has_kw,
        "kw_hits": ", ".join(kw_hits),
        "composite_rank": round(composite_rank, 4)
    })

df_c = pd.DataFrame(candidates).sort_values(by="composite_rank", ascending=False)
print(f"Total Purified 'new -> used' Candidates Available: {len(df_c)}")

print("\n--- TOP 35 CANDIDATES (SLOT 1) ---")
for idx, r in df_c.head(35).iterrows():
    print(f"ID {int(r['item_id']):5d}: rank={r['composite_rank']:.3f} | p={r['p_blend']:.3f} | tested={r['times_tested']} (gain={r['b_gain']:+.2f}) | kw=[{r['kw_hits']}] | ${r['price']:.0f} | {r['category_id']} | {r['title']}")

print("\n--- NEXT 25 CANDIDATES (SLOT 2, CUMULATIVE 60) ---")
for idx, r in df_c.iloc[35:60].iterrows():
    print(f"ID {int(r['item_id']):5d}: rank={r['composite_rank']:.3f} | p={r['p_blend']:.3f} | tested={r['times_tested']} (gain={r['b_gain']:+.2f}) | kw=[{r['kw_hits']}] | ${r['price']:.0f} | {r['category_id']} | {r['title']}")

print("\n--- NEXT 25 CANDIDATES (SLOT 3, CUMULATIVE 85) ---")
for idx, r in df_c.iloc[60:85].iterrows():
    print(f"ID {int(r['item_id']):5d}: rank={r['composite_rank']:.3f} | p={r['p_blend']:.3f} | tested={r['times_tested']} (gain={r['b_gain']:+.2f}) | kw=[{r['kw_hits']}] | ${r['price']:.0f} | {r['category_id']} | {r['title']}")

print("\n--- NEXT 25 CANDIDATES (SLOT 4, CUMULATIVE 110) ---")
for idx, r in df_c.iloc[85:110].iterrows():
    print(f"ID {int(r['item_id']):5d}: rank={r['composite_rank']:.3f} | p={r['p_blend']:.3f} | tested={r['times_tested']} (gain={r['b_gain']:+.2f}) | kw=[{r['kw_hits']}] | ${r['price']:.0f} | {r['category_id']} | {r['title']}")
