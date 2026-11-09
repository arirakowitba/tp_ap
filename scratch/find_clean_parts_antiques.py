import json
import pandas as pd

s7_4 = pd.read_csv('submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv').set_index('ID')['condition']

with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    test_rows = [json.loads(line) for line in f]

exact_used_tokens = [
    'para repuestos', 'para repuesto', 'para reparar', 'para desarme', 
    'no funciona', 'permuto', 'antigua foto', 'foto antigua', 'antiguo cenicero',
    'pinza antigua', 'antiguo farol', 'antigua lampara', 'antiguo reloj',
    'billete antiguo', 'moneda antigua'
]

matches = []
for idx, it in enumerate(test_rows, 1):
    curr = s7_4.loc[idx]
    if curr == 'new':
        t = it.get('title', '')
        t_low = t.lower()
        matched = [tok for tok in exact_used_tokens if tok in t_low]
        if matched:
            iq = it.get('initial_quantity', 1)
            p = float(it.get('price', 0) or 0)
            if iq == 1 and p < 1500 and not any(k in t_low for k in ['0 km', '0km', 'a estrenar', 'nuevo en blister', 'nuevo cerrado', 'sellado']):
                matches.append({
                    'ID': idx,
                    'title': t,
                    'price': p,
                    'matched': matched[0],
                    'cat': it.get('category_id')
                })

df_m = pd.DataFrame(matches)
print(f"Total pure physical parts/antiques currently NEW in anchor: {len(df_m)}")
print(df_m.to_string())
