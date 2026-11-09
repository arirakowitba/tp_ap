"""Analyze tag correlation with condition."""
import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data/ap_2026_q_2/raw")
tags_new = Counter()
tags_used = Counter()

with open(DATA_DIR / "train_data.jsonlines", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        cond = item.get("condition")
        tags = item.get("tags") or []
        if cond == "new":
            for t in tags:
                tags_new[t] += 1
        elif cond == "used":
            for t in tags:
                tags_used[t] += 1

all_tags = set(tags_new.keys()) | set(tags_used.keys())
print(f"{'TAG':35s} | {'NEW COUNT':10s} | {'USED COUNT':10s} | {'USED %':8s}")
print("-" * 70)
for t in sorted(all_tags, key=lambda x: tags_new[x] + tags_used[x], reverse=True)[:15]:
    c_n = tags_new[t]
    c_u = tags_used[t]
    pct_u = c_u / (c_n + c_u) * 100
    print(f"{t:35s} | {c_n:10d} | {c_u:10d} | {pct_u:6.1f}%")
