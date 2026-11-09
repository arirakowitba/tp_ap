import json
import pandas as pd

anchor = pd.read_csv('submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv').set_index('ID')['condition']

# Ingest test items
with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    test_rows = [json.loads(line) for line in f]

obvious_used_tokens = [
    'para repuestos', 'para repuesto', 'para reparar', 'para desarme', 
    'no funciona', 'con detalles', 'detalles de uso', 'marcas de uso',
    'usado en buen estado', 'usada en buen estado', 'segunda mano',
    'antiguo en buen estado', 'antigua en buen estado', 'impecable estado usado',
    'usado muy buen estado', 'usada muy buen estado', 'muy buen estado usado',
    'vendo por mudanza'
]

flips = []
for idx, it in enumerate(test_rows, 1):
    curr = anchor.loc[idx]
    if curr == 'new':
        t_low = it.get('title', '').lower()
        matched = [tok for tok in obvious_used_tokens if tok in t_low]
        if matched:
            flips.append({
                'ID': idx,
                'title': it.get('title'),
                'price': it.get('price'),
                'matched': matched[0]
            })

df_flips = pd.DataFrame(flips)
print(f"Total obvious used items currently marked NEW in anchor: {len(df_flips)}")
print(df_flips.to_string())
