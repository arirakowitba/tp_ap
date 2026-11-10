"""Inspect the 6 items between EVT-132 and EVT-135."""
import json

with open("data/ap_2026_q_2/raw/test_data.jsonlines", encoding="utf-8") as f:
    items = [json.loads(line) for line in f]

for fid in [7048, 10123, 15673, 18412, 21122, 26668]:
    it = items[fid - 1]
    print(f"ID {fid:5d} | Price: ${it.get('price', 0):7.1f} | Cat: {it.get('category_id')} | Title: {it.get('title')}")
