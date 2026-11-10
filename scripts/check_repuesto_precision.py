"""Check precision of 'para repuesto', 'para reparar', 'no funciona' in train_data."""
import json
import re

with open("data/ap_2026_q_2/raw/train_data.jsonlines", encoding="utf-8") as f:
    items = [json.loads(line) for line in f]

pattern = re.compile(r"\b(para repuestos?|para reparar|no funciona|para desarme)\b", re.IGNORECASE)

matches = []
for it in items:
    title = it.get("title", "")
    if pattern.search(title):
        matches.append(it.get("condition"))

n_total = len(matches)
n_used = sum(1 for c in matches if c == "used")
n_new = sum(1 for c in matches if c == "new")

print(f"Total items in train with 'para repuesto/para reparar/no funciona': {n_total}")
print(f"Used: {n_used} ({n_used/n_total*100:.2f}%)")
print(f"New:  {n_new} ({n_new/n_total*100:.2f}%)")
