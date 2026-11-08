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

BAD_PATTERNS = [
    r'\b0\s*km\b', r'\bcero\s*km\b', r'\bnuevo\b', r'\bnueva\b', r'\bnuevos\b', r'\bnuevas\b',
    r'\bsellad[oas]*\b', r'\bcerrad[oas]*\b', r'\bblister\b', r'\bestrenar\b',
    r'\ben caja\b', r'\bde fabrica\b', r'\bsin uso\b', r'\bposter\b', r'\bconcesionari[oa]\b',
    r'\bmoto\b', r'\bmotosport\b', r'\bplanta\b', r'\bsuculenta\b', r'\baeonium\b',
    r'\blote de\b', r'\bmayorista\b', r'\bdistribuidor\b', r'\bfabrica\b'
]

WINNING_MARKERS = [
    r'\bno funciona\b', r'\ba reparar\b', r'\bpara reparar\b', r'\bpara repuesto[s]*\b',
    r'\ba revisar\b', r'\bsin probar\b', r'\bdesarme\b',
    r'\bantigu[oas]\b', r'\bdecada del \d\d\b', r'\baos \d\d\b',
    r'\bvendo por mudanza\b', r'\bun solo uso\b', r'\bpoco uso\b'
]

candidates = []
for i, c in s4.items():
    if c == 'new':
        r = test_rows[i-1]
        t = str(r.get('title', '')).lower()
        cat = str(r.get('category_id', ''))
        price = float(r.get('price', 0) or 0)
        iq = int(r.get('initial_quantity', 1) or 1)
        
        if cat in EXCLUDE_CATS or iq > 1 or price > 2000 or price < 20:
            continue
            
        if any(re.search(bp, t) for bp in BAD_PATTERNS):
            continue
            
        matched_win = None
        for wm in WINNING_MARKERS:
            if re.search(wm, t):
                matched_win = wm
                break
                
        if matched_win:
            p_b = probs.loc[i, 'p_blend_103vec'] if i in probs.index else 0.5
            candidates.append({
                'ID': i,
                'title': r.get('title'),
                'price': price,
                'marker': matched_win,
                'p_blend': p_b
            })

df_cands = pd.DataFrame(candidates).sort_values(by='p_blend', ascending=False)
print(f"Total Winning-Profile Candidates found: {len(df_cands)}")
for idx, r in df_cands.iloc[:40].iterrows():
    print(f"ID={r['ID']:5d} | Pb={r['p_blend']:.3f} | ${r['price']:4.0f} | {r['marker']:20s} | {r['title'][:55]}")
