import pandas as pd
import json

with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    items = [json.loads(line) for line in f]
test_items = {idx+1: it for idx, it in enumerate(items)}

slot1_ids = [2204, 5616, 6072, 6822, 7936, 8461, 9822, 10356, 12726, 14405, 16515, 16846, 17934, 19025, 22499, 22539, 22770, 25371, 27683, 28010]
slot2_ids = [1471, 2531, 4637, 6619, 6688, 7671, 9732, 14227, 16982, 18423, 18830, 19591, 20220, 24000, 24858, 25885, 28321, 29306]

print("=== SLOT 1 FLIPS (new -> used, Delta = -6 hits) ===")
for cid in slot1_ids:
    it = test_items[cid]
    cat = it.get('category_id', '')
    p = it.get('price', 0)
    t = it.get('title', '')
    print(f"ID {cid:5d} | Cat: {cat} | ${p:8.1f} | {t[:60]}")

print("\n=== SLOT 2 FLIPS (used -> new, Delta = -5 hits) ===")
for cid in slot2_ids:
    it = test_items[cid]
    cat = it.get('category_id', '')
    p = it.get('price', 0)
    t = it.get('title', '')
    print(f"ID {cid:5d} | Cat: {cat} | ${p:8.1f} | {t[:60]}")
