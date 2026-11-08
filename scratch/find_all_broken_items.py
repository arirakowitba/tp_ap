import pandas as pd, json, re
from pathlib import Path

p_s4_y = 'submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv'
p_probs = 'results/transductive_deep_models_103vec/transductive_test_probabilities_103vec.csv'
p_test = 'data/ap_2026_q_2/raw/test_data.jsonlines'

s4 = pd.read_csv(p_s4_y).set_index('ID')['condition']
probs = pd.read_csv(p_probs).set_index('ID')

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

BROKEN_PARTS_PATTERNS = [
    r'\bno funciona\b', r'\ba reparar\b', r'\bpara reparar\b',
    r'\bpara repuesto[s]*\b', r'\bdesarme\b', r'\ba revisar\b'
]

found = []
for i, c in s4.items():
    if c == 'new':
        r = test_rows[i-1]
        t = str(r.get('title', '')).lower()
        cat = str(r.get('category_id', ''))
        price = float(r.get('price', 0) or 0)
        iq = int(r.get('initial_quantity', 1) or 1)
        
        if cat in EXCLUDE_CATS or iq > 1 or price > 1500 or price < 20:
            continue
            
        matched = None
        for pat in BROKEN_PARTS_PATTERNS:
            if re.search(pat, t):
                matched = pat
                break
                
        if matched:
            pb = probs.loc[i, 'p_blend_103vec'] if i in probs.index else 0.5
            found.append({
                'ID': i,
                'title': r.get('title'),
                'price': price,
                'marker': matched,
                'p_blend': pb
            })

df_found = pd.DataFrame(found).sort_values(by='p_blend', ascending=False)
print(f"Total Broken / Scrap / Repair items currently predicted as NEW in S4: {len(df_found)}")
for idx, r in df_found.iterrows():
    print(f"ID={r['ID']:5d} | Pb={r['p_blend']:.3f} | ${r['price']:4.0f} | {r['marker']:18s} | {r['title'][:55]}")
