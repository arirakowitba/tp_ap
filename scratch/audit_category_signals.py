import json
from collections import defaultdict
from pathlib import Path
import pandas as pd

p_train = Path('data/ap_2026_q_2/raw/train_data.jsonlines')
p_test = Path('data/ap_2026_q_2/raw/test_data.jsonlines')
p_s4 = Path('submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv')

s4 = pd.read_csv(p_s4).set_index('ID')['condition']

cat_stats = defaultdict(lambda: {'new': 0, 'used': 0})
with open(p_train, encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        cat_stats[r['category_id']][r['condition']] += 1

print(f"Total categories in Train: {len(cat_stats)}")

pure_used_cats = {k: v for k, v in cat_stats.items() if v['used'] >= 5 and v['new'] == 0}
pure_new_cats = {k: v for k, v in cat_stats.items() if v['new'] >= 10 and v['used'] == 0}

print(f"Categories with >= 5 train items and 100% USED: {len(pure_used_cats)}")
print(f"Categories with >= 10 train items and 100% NEW: {len(pure_new_cats)}")

# Check test items in pure used categories
test_items = []
with open(p_test, encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        r = json.loads(line)
        cat = r.get('category_id')
        if cat in pure_used_cats:
            test_items.append({
                'ID': i,
                'cat': cat,
                'title': r.get('title'),
                'price': float(r.get('price', 0) or 0),
                'train_used': pure_used_cats[cat]['used'],
                'curr_s4': s4[i]
            })

df_used_cat_test = pd.DataFrame(test_items).set_index('ID')
print(f"\nTest items in 100% USED categories (>=5 in train): {len(df_used_cat_test)}")
print("S4 predictions for these items:")
print(df_used_cat_test['curr_s4'].value_counts())

discrep_used_cat = df_used_cat_test[df_used_cat_test['curr_s4'] == 'new']
print(f"\nDiscrepancies: S4 predicts NEW in 100% USED category: {len(discrep_used_cat)}")
for idx, r in discrep_used_cat.iloc[:20].iterrows():
    print(f"ID={idx:5d} Cat={r['cat']} (Train used:{r['train_used']}) ${r['price']:.0f} | {r['title'][:60]}")
