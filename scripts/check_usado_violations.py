import json
import re
import pandas as pd

anchor_df = pd.read_csv('submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv')
anchor_p = anchor_df.set_index('ID')['condition'].to_dict()

with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    items = [json.loads(line) for line in f]
test_items = {idx+1: it for idx, it in enumerate(items)}

c = re.compile(r'\busad[oa]s?\b', re.IGNORECASE)

violations = []
for cid, it in test_items.items():
    title = it.get('title', '')
    if c.search(title):
        cur = anchor_p[cid]
        if cur == 'new':
            violations.append((cid, it.get('price', 0), it.get('category_id', ''), title))

print(f"Total test items containing 'usado/a/s': {sum(1 for it in test_items.values() if c.search(it.get('title', '')))}")
print(f"Items where title says 'usado/a/s' BUT EVT-143 predicts NEW: {len(violations)}")
for cid, price, cat, title in violations:
    print(f"ID {cid:5d} | Cat: {cat} | ${price:8.1f} | {title}")
