import json
import re

with open('data/ap_2026_q_2/raw/train_data.jsonlines', encoding='utf-8') as f:
    train_items = [json.loads(line) for line in f]

test_queries = [
    r'\bantigu[oa]s?\b',
    r'\blampara antigua\b',
    r'\bretro vintage\b',
    r'\blote de\b',
    r'\bcoleccion\b',
    r'\bdesarme\b',
    r'\bchatarra\b',
    r'\bfalla\b',
    r'\bdetalles\b',
    r'\busad[oa]s?\b'
]

print("=== CUES PRECISION IN TRAIN ===")
for pat in test_queries:
    c = re.compile(pat, re.IGNORECASE)
    matches = [it['condition'] for it in train_items if c.search(it['title'])]
    tot = len(matches)
    n_used = sum(1 for m in matches if m == 'used')
    n_new = sum(1 for m in matches if m == 'new')
    pct_used = (n_used / tot * 100) if tot else 0
    print(f"Pattern: {pat:25s} | Total: {tot:5d} | Used: {n_used:4d} ({pct_used:5.1f}%) | New: {n_new:4d}")
