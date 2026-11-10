import sys
sys.path.insert(0, '.')
import json
import pandas as pd
from scratch.analyze_all_vectors import anchor

print("Scanning test items for high-confidence lexical mismatches against anchor EVT-143...")
with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    test_lines = [json.loads(line) for line in f]

used_cues = [
    'buen estado', 'excelente estado', 'muy buen estado', 'impecable estado',
    'desarme', 'para repuesto', 'para repuestos', 'no funciona', 'a reparar',
    'para reparar', 'usado', 'usada', 'usadas', 'usados', 'con detalles'
]

new_cues = [
    'a estrenar', 'sin rodar', 'en caja cerrada', 'sellado de fabrica',
    'en blister cerrado', 'sobres cerrados', 'sobre cerrado', '0km en motolandia'
]

mismatches = []
for idx, d in enumerate(test_lines):
    iid = idx + 1
    title = d.get('title', '').lower()
    anch = anchor['vec'][iid]
    
    # Check used cues where anchor is new
    if anch == 'new':
        for cue in used_cues:
            if cue in title:
                mismatches.append({
                    'id': iid,
                    'anchor': 'new',
                    'rec': 'used',
                    'cue': cue,
                    'title': d.get('title'),
                    'price': d.get('price'),
                    'cat': d.get('category_id')
                })
                break
                
    # Check new cues where anchor is used
    if anch == 'used':
        for cue in new_cues:
            if cue in title:
                mismatches.append({
                    'id': iid,
                    'anchor': 'used',
                    'rec': 'new',
                    'cue': cue,
                    'title': d.get('title'),
                    'price': d.get('price'),
                    'cat': d.get('category_id')
                })
                break

print(f"Total pure lexical mismatches found: {len(mismatches)}")
df_m = pd.DataFrame(mismatches)
print(df_m['cue'].value_counts())
print("\n--- NEW TO USED (Anchor predicted NEW, title says USED) ---")
for idx, r in df_m[df_m['rec'] == 'used'].iterrows():
    print(f"ID {r['id']:5d} | Cue: {r['cue']:18s} | Cat: {r['cat']:10s} | Price: {r['price']:8} | Title: {r['title']}")

print("\n--- USED TO NEW (Anchor predicted USED, title says NEW) ---")
for idx, r in df_m[df_m['rec'] == 'new'].iterrows():
    print(f"ID {r['id']:5d} | Cue: {r['cue']:18s} | Cat: {r['cat']:10s} | Price: {r['price']:8} | Title: {r['title']}")
