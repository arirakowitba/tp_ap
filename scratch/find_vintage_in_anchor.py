import json
import pandas as pd

anchor = pd.read_csv('submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv').set_index('ID')['condition']

with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    test_rows = [json.loads(line) for line in f]

vintage_tokens = [
    'antiguo ', 'antigua ', 'antiguos ', 'antiguas ',
    'decada del 70', 'decada del 80', 'decada del 60', 'decada del 50',
    'decada 70', 'decada 80', 'decada 60', 'decada 50',
    'de coleccion', 'coleccionista'
]

flips = []
for idx, it in enumerate(test_rows, 1):
    curr = anchor.loc[idx]
    if curr == 'new':
        t_low = it.get('title', '').lower()
        matched = [tok for tok in vintage_tokens if tok in t_low]
        if matched and it.get('initial_quantity') == 1 and not any(k in t_low for k in ['nuevo', '0 km', 'nueva']):
            flips.append({
                'ID': idx,
                'title': it.get('title'),
                'price': it.get('price'),
                'matched': matched[0],
                'cat': it.get('category_id')
            })

df_v = pd.DataFrame(flips)
print(f"Total vintage/antique items currently marked NEW in anchor (iq=1): {len(df_v)}")
print(df_v.head(25).to_string())
