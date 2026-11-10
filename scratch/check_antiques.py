import sys
sys.path.insert(0, '.')
import json
from scratch.analyze_all_vectors import anchor

with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    test_lines = [json.loads(line) for line in f]

antique_candidates = []
for idx, d in enumerate(test_lines):
    iid = idx + 1
    t = d.get('title', '').lower()
    anch = anchor['vec'][iid]
    cat = d.get('category_id')
    
    # Check numismatics category MLA44260 or antique coins / banknotes / stamps
    if cat == 'MLA44260' or 'numismatica' in t or 'billete antiguo' in t or 'moneda antigua' in t or 'estampillas' in t:
        if anch == 'new':
            antique_candidates.append((iid, cat, d.get('price'), anch, d.get('title')))

print(f"Total antique items where anchor predicts NEW: {len(antique_candidates)}")
for c in antique_candidates:
    print(f"ID {c[0]:5d} | Cat: {c[1]:10s} | Price: {c[2]:8} | Title: {c[4]}")
