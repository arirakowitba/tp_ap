"""Check precision of strong used cues in train."""
import json
import re

with open("data/ap_2026_q_2/raw/train_data.jsonlines", encoding="utf-8") as f:
    items = [json.loads(line) for line in f]

cues = [
    (r"\bcomo nuevo\b", "como nuevo"),
    (r"\bexcelente estado\b", "excelente estado"),
    (r"\bmuy buen estado\b", "muy buen estado"),
    (r"\bbien cuidado\b", "bien cuidado"),
    (r"\bcon detalles\b", "con detalles"),
    (r"\bpoco uso\b", "poco uso"),
    (r"\bsegunda mano\b", "segunda mano"),
]

for pat, name in cues:
    regex = re.compile(pat, re.IGNORECASE)
    matched = [it.get("condition") for it in items if regex.search(it.get("title", ""))]
    total = len(matched)
    used = sum(1 for c in matched if c == "used")
    pct = (used / total * 100) if total > 0 else 0
    print(f"Cue: {name:20s} | Total in train: {total:4d} | Used: {used:4d} | Precision: {pct:.2f}%")
