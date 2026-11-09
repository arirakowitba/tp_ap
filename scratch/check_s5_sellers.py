import json
import pandas as pd

items_in_s5 = [815, 2037, 2349, 6900, 7255, 7936, 9562, 16515, 19854, 21530, 21659, 22237, 22499, 23291, 24779, 25371, 25490, 28295, 29859]

with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    test_dict = {i+1: json.loads(line) for i, line in enumerate(f)}

# Load train sellers
with open('data/ap_2026_q_2/raw/train_data.jsonlines', encoding='utf-8') as f:
    train_sellers = {}
    for line in f:
        it = json.loads(line)
        s = it['seller_id']
        c = it['condition']
        if s not in train_sellers:
            train_sellers[s] = {'used': 0, 'new': 0}
        train_sellers[s][c] += 1

records = []
for idx in items_in_s5:
    it = test_dict[idx]
    s = it['seller_id']
    s_hist = train_sellers.get(s, {'used': 0, 'new': 0})
    u = s_hist['used']
    n = s_hist['new']
    tot = u + n
    pct_u = u / tot if tot > 0 else 0.5
    records.append({
        'ID': idx,
        'title': it.get('title')[:45],
        'price': it.get('price'),
        'iq': it.get('initial_quantity'),
        'train_used': u,
        'train_new': n,
        'pct_used': round(pct_u, 3)
    })

df = pd.DataFrame(records)
print(df.to_string())
