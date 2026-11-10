"""Find all obvious false 'used' in EVT-132 with explicit NEW markers."""
import json
import re
import pandas as pd

with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    items = [json.loads(line) for line in f]

df_evt = pd.read_csv("submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv")

# Strong NEW patterns
PATTERNS = [
    (r"\b0\s*km\b", "0km"),
    (r"\bokm\b", "okm"),
    (r"\bcaja sellada\b", "caja sellada"),
    (r"\bcaja cerrada\b", "caja cerrada"),
    (r"\bnuevo sellado\b", "nuevo sellado"),
    (r"\bnuevo cerrado\b", "nuevo cerrado"),
    (r"\ben blister\b", "en blister"),
    (r"\bcon etiqueta\b", "con etiqueta"),
    (r"\bcon etiquetas\b", "con etiquetas"),
    (r"\ba estrenar\b", "a estrenar"),
    (r"\bgarantia de fabrica\b", "garantia de fabrica"),
    (r"\bgarantia oficial\b", "garantia oficial"),
]

# Exclusion patterns (fake new)
EXCLUDE = [
    r"\bigual a\b",
    r"\bcomo nuevo\b",
    r"\bparece nuevo\b",
    r"\bpara repuesto\b",
    r"\broto\b",
    r"\bno funciona\b",
    r"\busad[oa]\b",
]
exclude_regex = re.compile("|".join(EXCLUDE), re.IGNORECASE)

candidates = []
for idx, item in enumerate(items):
    cond = df_evt.loc[idx, "condition"]
    if cond != "used":
        continue  # only look at items EVT predicted as used
    
    title = item.get("title", "")
    title_lower = title.lower()
    
    if exclude_regex.search(title_lower):
        continue
    
    matched = []
    for pat, name in PATTERNS:
        if re.search(pat, title_lower):
            matched.append(name)
            
    if matched:
        candidates.append({
            "idx": idx,
            "id": idx + 1,
            "title": title,
            "price": item.get("price", 0),
            "category_id": item.get("category_id", ""),
            "cues": ", ".join(matched),
            "tags": " ".join(item.get("tags") or []),
            "warranty": str(item.get("warranty") or "")
        })

df_c = pd.DataFrame(candidates)
print(f"Total obvious false 'used' in EVT-132: {len(df_c)}")
df_c.to_csv("results/false_used_in_evt132.csv", index=False)

print("\nSample top candidates:")
for _, r in df_c.head(30).iterrows():
    print(f"ID {r['id']:5d} | ${r['price']:9.1f} | Cue: {r['cues']:18s} | {r['title'][:55]}")
