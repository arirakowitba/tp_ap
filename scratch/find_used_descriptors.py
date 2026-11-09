import json
import pandas as pd

anchor = pd.read_csv('submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv').set_index('ID')['condition']

with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    test_rows = [json.loads(line) for line in f]

used_descriptors = [
    'funciona perfecto', 'funciona impecable', 'funciona 10 puntos', 'funcionando bien',
    'en muy buen estado', 'muy buen estado', 'en buen estado', 'buen estado',
    'en impecable estado', 'impecable estado', 'en excelente estado',
    'poco uso', 'sin uso en caja', 'como nuevo'
]

flips = []
for idx, it in enumerate(test_rows, 1):
    curr = anchor.loc[idx]
    if curr == 'new':
        t_low = it.get('title', '').lower()
        matched = [tok for tok in used_descriptors if tok in t_low]
        # Filter out if it has explicit '0 km' or 'a estrenar' or 'nuevo en caja sellada'
        if matched and not any(k in t_low for k in ['0 km', '0km', 'a estrenar', 'nuevo cerrado', 'nueva con garantia']):
            flips.append({
                'ID': idx,
                'title': it.get('title'),
                'price': it.get('price'),
                'matched': matched[0],
                'iq': it.get('initial_quantity')
            })

df_flips = pd.DataFrame(flips)
print(f"Total used descriptors currently marked NEW in anchor: {len(df_flips)}")
# Filter for iq == 1 (individual single item)
df_iq1 = df_flips[df_flips['iq'] == 1]
print(f"Of which initial_quantity == 1: {len(df_iq1)}")
print(df_iq1.head(25).to_string())
