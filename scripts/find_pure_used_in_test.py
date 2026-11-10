"""Find all test items with 98-100% precision used cues where EVT-132 said NEW."""
import json
import re
import pandas as pd

with open("data/ap_2026_q_2/raw/test_data.jsonlines", encoding="utf-8") as f:
    items = [json.loads(line) for line in f]

pattern = re.compile(r"\b(muy buen estado|poco uso|excelente estado|para repuestos?|para reparar|para desarme)\b", re.IGNORECASE)

df_evt = pd.read_csv("submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv")

candidates = []
for idx, it in enumerate(items):
    cond = df_evt.loc[idx, "condition"]
    title = it.get("title", "")
    if cond == "new" and pattern.search(title):
        m = pattern.search(title).group(0).lower()
        candidates.append({
            "idx": idx,
            "id": idx + 1,
            "cue": m,
            "title": title,
            "price": it.get("price", 0),
            "category_id": it.get("category_id")
        })

print(f"Total test items where EVT said NEW but contains 98-100% used cue: {len(candidates)}")
for c in candidates:
    print(f"ID {c['id']:5d} | Cue: {c['cue']:16s} | ${c['price']:7.1f} | {c['title']}")
