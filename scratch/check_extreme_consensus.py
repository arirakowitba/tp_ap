import pandas as pd
import json

df_probs = pd.read_csv('results/transductive_deep_models/transductive_test_probabilities.csv').set_index('ID')
anchor = pd.read_csv('submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv').set_index('ID')['condition']

with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    test_dict = {i+1: json.loads(line) for i, line in enumerate(f)}

cands = []
for idx in range(1, 30001):
    if anchor.loc[idx] == 'new':
        p_b = df_probs.loc[idx, 'p_transductive_blend']
        p_cb = df_probs.loc[idx, 'p_cb_transductive']
        p_lgb = df_probs.loc[idx, 'p_lgb_transductive']
        if p_b >= 0.92 and p_cb >= 0.90 and p_lgb >= 0.90:
            it = test_dict[idx]
            iq = it.get('initial_quantity', 1)
            p = float(it.get('price', 0) or 0)
            t = str(it.get('title', ''))
            t_low = t.lower()
            if iq == 1 and p < 5000 and not any(k in t_low for k in ['0 km', '0km', 'a estrenar', 'nuevo en blister', 'nuevo cerrado', 'sellado']):
                cands.append({
                    'ID': idx,
                    'p_blend': p_b,
                    'p_cb': p_cb,
                    'p_lgb': p_lgb,
                    'price': p,
                    'title': t,
                    'cat': it.get('category_id')
                })

df_c = pd.DataFrame(cands).sort_values(by='p_blend', ascending=False)
print(f"Total extreme consensus items (P >= 0.92, currently NEW in anchor): {len(df_c)}")
print(df_c.head(25).to_string())
