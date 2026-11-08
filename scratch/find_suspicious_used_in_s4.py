import pandas as pd, json, re
from pathlib import Path

p_s4_y = 'submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv'
p_test = 'data/ap_2026_q_2/raw/test_data.jsonlines'

s4 = pd.read_csv(p_s4_y).set_index('ID')['condition']

with open(p_test, encoding='utf-8') as f:
    test_rows = [json.loads(line) for line in f]

EXCLUDE_CATS = {
    "MLA1459", "MLA1466", "MLA1467", "MLA1468", "MLA1470", "MLA1471", 
    "MLA1472", "MLA1473", "MLA1474", "MLA1475", "MLA1495", # Real Estate
    "MLA50279", "MLA50280", "MLA50281", # Vacation rentals
    "MLA1743", "MLA1744", "MLA2547", "MLA1745", "MLA1746", # Vehicles
    "MLA5632", "MLA1071", "MLA1072", # Pets
    "MLA1540" # Services
}

BAD_PATTERNS = [
    r'\b0\s*km\b', r'\bcero\s*km\b', r'\bnuevo cerrado\b', r'\bnueva cerrada\b',
    r'\bcaja sellada\b', r'\ben blister\b', r'\balquiler\b', r'\bdepartamento\b',
    r'\bposter afiche\b', r'\bafiche poster\b'
]

suspicious_used_in_s4 = []
for i, c in s4.items():
    if c == 'used':
        r = test_rows[i-1]
        t = str(r.get('title', '')).lower()
        cat = str(r.get('category_id', ''))
        price = float(r.get('price', 0) or 0)
        
        is_bad_cat = cat in EXCLUDE_CATS
        matched_pat = None
        for pat in BAD_PATTERNS:
            if re.search(pat, t):
                matched_pat = pat
                break
                
        if is_bad_cat or matched_pat:
            suspicious_used_in_s4.append((i, r.get('title'), price, cat, matched_pat, is_bad_cat))

print(f"Total suspicious items predicted as USED in S4: {len(suspicious_used_in_s4)}")
for i, title, price, cat, pat, bad_cat in suspicious_used_in_s4:
    reason = f"BadCat:{cat}" if bad_cat else f"Keyword:{pat}"
    print(f"ID={i:5d} | Price=${price:6.0f} | Reason={reason:20s} | {title}")
