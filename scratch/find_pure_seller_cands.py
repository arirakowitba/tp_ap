import json
import pandas as pd

anchor = pd.read_csv('submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv').set_index('ID')['condition']

# Ingest train sellers
with open('data/ap_2026_q_2/raw/train_data.jsonlines', encoding='utf-8') as f:
    train_sellers = {}
    for line in f:
        it = json.loads(line)
        s = it['seller_id']
        c = it['condition']
        if s not in train_sellers:
            train_sellers[s] = {'used': 0, 'new': 0}
        train_sellers[s][c] += 1

# Ingest test items
with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    test_rows = [json.loads(line) for line in f]

bad_tokens = ['nuevo', 'nueva', 'sellado', 'cerrado', '0 km', '0km', 'a estrenar', 'garantia', 'en blister', 'original cerrado']

matches = []
for idx, it in enumerate(test_rows, 1):
    curr = anchor.loc[idx]
    if curr == 'new':
        s = it.get('seller_id')
        s_hist = train_sellers.get(s, {'used': 0, 'new': 0})
        u = s_hist['used']
        n = s_hist['new']
        tot = u + n
        if tot >= 8 and (u / tot) >= 0.80:
            iq = it.get('initial_quantity', 1)
            p = float(it.get('price', 0) or 0)
            t_low = it.get('title', '').lower()
            if iq == 1 and p < 2500 and not any(tok in t_low for tok in bad_tokens):
                matches.append({
                    'ID': idx,
                    'title': it.get('title'),
                    'price': p,
                    'seller_id': s,
                    'train_used': u,
                    'train_new': n,
                    'pct_used': round(u / tot, 3),
                    'cat': it.get('category_id')
                })

df_matches = pd.DataFrame(matches)
print(f"Total seller-verified candidates currently marked NEW in anchor: {len(df_matches)}")
df_matches = df_matches.sort_values(by=['pct_used', 'train_used'], ascending=[False, False])
print(df_matches.head(30).to_string())
