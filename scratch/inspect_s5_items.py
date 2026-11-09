import json
import pandas as pd

items_in_s5 = [815, 2037, 2349, 6900, 7255, 7936, 9562, 16515, 19854, 21530, 21659, 22237, 22499, 23291, 24779, 25371, 25490, 28295, 29859]

with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    test_dict = {i+1: json.loads(line) for i, line in enumerate(f)}

records = []
for idx in items_in_s5:
    it = test_dict[idx]
    records.append({
        'ID': idx,
        'title': it.get('title'),
        'price': it.get('price'),
        'iq': it.get('initial_quantity'),
        'sq': it.get('sold_quantity'),
        'seller': it.get('seller_id'),
        'cat': it.get('category_id')
    })

df_s5 = pd.DataFrame(records)
print(df_s5.to_string())
