import sys
sys.path.insert(0, '.')
import json
import pandas as pd
from scratch.analyze_all_vectors import anchor

print("Scanning test data attributes for vehicles...")
with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    test_lines = [json.loads(line) for line in f]

vehicle_candidates = []

for idx, d in enumerate(test_lines):
    iid = idx + 1
    attrs = {a['name']: a['value_name'] for a in d.get('attributes', [])}
    km_str = attrs.get('Kilómetros')
    ano_str = attrs.get('Año')
    title = d.get('title', '').lower()
    anchor_pred = anchor['vec'][iid]
    
    if km_str is not None:
        try:
            km = float(str(km_str).replace('.', '').replace(',', '').replace(' km', '').strip())
        except:
            km = -1
        
        # If km == 0 or km == 1, or title clearly states 0km / 0 km
        if km in [0, 1] or ('0 km' in title or '0km' in title):
            # Check for negation: igual a 0km, como 0km, etc.
            is_negated = any(neg in title for neg in ['igual a', 'como 0', 'parece 0', 'como nuevo', 'igual a nuevo'])
            vehicle_candidates.append({
                'id': iid,
                'title': d.get('title'),
                'km': km,
                'ano': ano_str,
                'negated': is_negated,
                'anchor': anchor_pred,
                'recommended': 'used' if is_negated else ('new' if km in [0, 1] else 'check')
            })

print(f"Total vehicle 0km / low-km candidates: {len(vehicle_candidates)}")
for c in vehicle_candidates:
    print(f"ID: {c['id']:5d} | KM: {c['km']:6} | Ano: {str(c['ano']):4s} | Neg: {c['negated']} | Anchor: {c['anchor']:4s} | Rec: {c['recommended']:4s} | Title: {c['title'][:50]}")
