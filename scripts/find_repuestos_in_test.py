"""Find all test items with 'para repuesto/para reparar/no funciona' where EVT-132 said NEW."""
import json
import re
import pandas as pd

with open("data/ap_2026_q_2/raw/test_data.jsonlines", encoding="utf-8") as f:
    items = [json.loads(line) for line in f]

pattern = re.compile(r"\b(para repuestos?|para reparar|no funciona|para desarme)\b", re.IGNORECASE)

df_evt = pd.read_csv("submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv")

candidates = []
for idx, it in enumerate(items):
    cond = df_evt.loc[idx, "condition"]
    title = it.get("title", "")
    if cond == "new" and pattern.search(title):
        candidates.append({
            "idx": idx,
            "id": idx + 1,
            "title": title,
            "price": it.get("price", 0),
            "category_id": it.get("category_id")
        })

print(f"Total test items with 'para repuesto/no funciona' where EVT said NEW: {len(candidates)}")
for c in candidates:
    print(f"ID {c['id']:5d} | ${c['price']:7.1f} | Cat: {c['category_id']} | {c['title']}")
