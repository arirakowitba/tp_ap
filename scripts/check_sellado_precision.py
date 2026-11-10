"""Check precision of 'caja sellada' and 'nuevo cerrado' in train."""
import json
import re

with open("data/ap_2026_q_2/raw/train_data.jsonlines", encoding="utf-8") as f:
    items = [json.loads(line) for line in f]

pattern = re.compile(r"\b(caja sellada|nuevo cerrado)\b", re.IGNORECASE)

matches = []
for it in items:
    title = it.get("title", "")
    if pattern.search(title):
        matches.append(it.get("condition"))

print(f"Total in train with 'caja sellada' or 'nuevo cerrado': {len(matches)}")
print(f"New:  {sum(1 for c in matches if c == 'new')} ({sum(1 for c in matches if c == 'new')/max(len(matches),1)*100:.2f}%)")
print(f"Used: {sum(1 for c in matches if c == 'used')} ({sum(1 for c in matches if c == 'used')/max(len(matches),1)*100:.2f}%)")
