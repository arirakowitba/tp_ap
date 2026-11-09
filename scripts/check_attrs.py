"""Analyze attributes correlation with condition."""
import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data/ap_2026_q_2/raw")
attr_new = Counter()
attr_used = Counter()

with open(DATA_DIR / "train_data.jsonlines", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        cond = item.get("condition")
        attrs = item.get("attributes") or []
        for a in attrs:
            if isinstance(a, dict):
                name = a.get("id") or a.get("name")
                val = a.get("value_name") or a.get("value_id")
                pair = f"{name}={val}"
                if cond == "new":
                    attr_new[pair] += 1
                elif cond == "used":
                    attr_used[pair] += 1

all_attrs = set(attr_new.keys()) | set(attr_used.keys())
print(f"{'ATTRIBUTE PAIR':45s} | {'NEW':6s} | {'USED':6s} | {'USED %':8s}")
print("-" * 75)
for p in sorted(all_attrs, key=lambda x: attr_new[x] + attr_used[x], reverse=True)[:25]:
    c_n = attr_new[p]
    c_u = attr_used[p]
    pct_u = c_u / (c_n + c_u) * 100
    print(f"{p:45s} | {c_n:6d} | {c_u:6d} | {pct_u:6.1f}%")
