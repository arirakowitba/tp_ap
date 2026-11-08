import pandas as pd, json, re
from pathlib import Path

p_s4_y = 'submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv'
p_test = 'data/ap_2026_q_2/raw/test_data.jsonlines'

s4 = pd.read_csv(p_s4_y).set_index('ID')['condition']

with open(p_test, encoding='utf-8') as f:
    test_rows = [json.loads(line) for line in f]

EXPLICIT_NEW_PATTERNS = [
    r'\b0\s*km\b', r'\bcero\s*km\b', r'\bsin rodar\b', r'\bnuevo cerrado\b',
    r'\bnueva cerrada\b', r'\bcaja sellada\b', r'\ben blister\b', r'\ba estrenar\b',
    r'\bcon etiquetas\b', r'\bcon etiqueta\b', r'\bde fabrica sellad\b'
]

found = []
for i, c in s4.items():
    if c == 'used':
        r = test_rows[i-1]
        t = str(r.get('title', '')).lower()
        price = float(r.get('price', 0) or 0)
        
        for pat in EXPLICIT_NEW_PATTERNS:
            if re.search(pat, t):
                found.append((i, r.get('title'), price, pat))
                break

print(f"Total items currently predicted as USED in S4 with explicit NEW tokens: {len(found)}")
for i, title, price, pat in found:
    print(f"ID={i:5d} | Price=${price:6.0f} | Pat={pat:20s} | {title}")
