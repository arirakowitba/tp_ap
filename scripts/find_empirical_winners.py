"""Find all flips with both strong empirical Kaggle probe evidence AND multi-model confidence."""
import json
import pandas as pd
import numpy as np

# 1. Load Bayesian 108vec effects
df_effects = pd.read_csv("results/bayesian_solver_108vec/bayesian_transductive_item_effects_108vec.csv")

# 2. Load EVT-132
df_evt = pd.read_csv("submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv")
preds_evt = df_evt.set_index("ID")["condition"].to_dict()

# 3. Load raw test data
with open("data/ap_2026_q_2/raw/test_data.jsonlines", encoding="utf-8") as f:
    items = [json.loads(line) for line in f]
item_dict = {it["id"]: it for it in items}

# Find candidates with:
# - expected_gain_vs_s4 >= 0.35
# - map_effect_vs_d1d2 >= 0.20
# - times_tested >= 2
# - alignment with models:
#     if used -> new: model_p_used < 0.40
#     if new -> used: model_p_used > 0.65

positive_candidates = []
for _, r in df_effects.iterrows():
    item_id = int(r["item_id"])
    evt_c = preds_evt[item_id]
    target_c = r["flipped_condition"]
    if evt_c == target_c:
        continue
    
    gain = r["expected_gain_vs_s4"]
    map_eff = r["map_effect_vs_d1d2"]
    tested = int(r["times_tested"])
    p_model = float(r["model_p_used"])
    
    # Filter for positive gain and empirical test support
    if gain < 0.35 or map_eff < 0.20 or tested < 2:
        continue
        
    direction = f"{evt_c} -> {target_c}"
    
    # Check model alignment
    if direction == "used -> new" and p_model > 0.40:
        continue
    if direction == "new -> used" and p_model < 0.65:
        continue
        
    it_raw = item_dict.get(r["item_id"], {})
    title = it_raw.get("title", r["title"])
    price = it_raw.get("price", r["price"])
    cat = it_raw.get("category_id", r["category_id"])
    
    positive_candidates.append({
        "item_id": item_id,
        "direction": direction,
        "expected_gain": gain,
        "map_effect": map_eff,
        "times_tested": tested,
        "model_p_used": p_model,
        "price": price,
        "category_id": cat,
        "title": title
    })

df_pos = pd.DataFrame(positive_candidates)
df_pos = df_pos.sort_values(by=["expected_gain", "times_tested"], ascending=[False, False])
print(f"Total High-Confidence Positive Candidates: {len(df_pos)}")
print("\nCandidates Breakdown by Direction:")
print(df_pos["direction"].value_counts())

print("\n--- ALL CANDIDATES ---")
for i, (_, r) in enumerate(df_pos.iterrows(), 1):
    print(f"#{i:2d} | ID {r['item_id']:5d} | {r['direction']:12s} | Gain: +{r['expected_gain']:.3f} | MAP: +{r['map_effect']:.3f} | Tests: {r['times_tested']:2d} | P_used: {r['model_p_used']:.3f} | ${r['price']:7.1f} | {r['title'][:45]}")
