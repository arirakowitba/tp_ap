import json
import pandas as pd
from analyze_todays_5_subs import all_diff_ids

all_ids_int = set(int(i) for i in all_diff_ids)

# Let's inspect test set items by 1-based line index
records = []
with open('data/ap_2026_q_2/raw/test_data.jsonlines', 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        item_id = idx + 1
        if item_id in all_ids_int:
            r = json.loads(line)
            records.append({
                'ID': item_id,
                'mla_id': r.get('id'),
                'title': r.get('title'),
                'price': r.get('price'),
                'currency_id': r.get('currency_id'),
                'category_id': r.get('category_id'),
                'seller_id': str(r.get('seller_id')),
                'warranty': r.get('warranty'),
                'tags': str(r.get('tags')),
            })

print(f"Matched {len(records)} / {len(all_ids_int)} test items")
df_info = pd.DataFrame(records).sort_values('ID')
df_info.to_csv('scratch/todays_35_items_metadata.csv', index=False)
for _, row in df_info.iterrows():
    print(f"ID {row['ID']:5d} | {str(row['title']):55s} | ${row['price']:8.2f} {row['currency_id']} | seller={row['seller_id']}")
