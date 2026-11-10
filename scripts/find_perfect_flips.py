"""Find the complete list of 98%+ precision lexical discrepancies against EVT-132."""
import json
import re
import pandas as pd

with open("data/ap_2026_q_2/raw/test_data.jsonlines", encoding="utf-8") as f:
    items = [json.loads(line) for line in f]

df_evt = pd.read_csv("submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv")

used_patterns = re.compile(r"\b(para repuestos?|para reparar|no funciona|para desarme)\b", re.IGNORECASE)
new_patterns = re.compile(r"\b(caja sellada|nuevo cerrado|en blister)\b", re.IGNORECASE)

exclude_new = re.compile(r"\b(como nuevo|igual a|abierto|usad[oa]|para repuesto)\b", re.IGNORECASE)

n2u = []
u2n = []

for idx, it in enumerate(items):
    cond = df_evt.loc[idx, "condition"]
    title = it.get("title", "")
    
    if cond == "new" and used_patterns.search(title):
        n2u.append({
            "id": idx + 1,
            "title": title,
            "price": it.get("price", 0),
            "cue": used_patterns.search(title).group(0)
        })
        
    if cond == "used" and new_patterns.search(title) and not exclude_new.search(title):
        u2n.append({
            "id": idx + 1,
            "title": title,
            "price": it.get("price", 0),
            "cue": new_patterns.search(title).group(0)
        })

print(f"Total new -> used (98% precision): {len(n2u)}")
for c in n2u:
    print(f"  ID {c['id']:5d} | ${c['price']:7.1f} | Cue: {c['cue']:15s} | {c['title']}")

print(f"\nTotal used -> new (98% precision): {len(u2n)}")
for c in u2n:
    print(f"  ID {c['id']:5d} | ${c['price']:7.1f} | Cue: {c['cue']:15s} | {c['title']}")
