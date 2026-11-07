import json
from collections import defaultdict
from pathlib import Path
import pandas as pd

p_train = Path('data/ap_2026_q_2/raw/train_data.jsonlines')
p_test = Path('data/ap_2026_q_2/raw/test_data.jsonlines')
p_s4 = Path('submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv')

s4 = pd.read_csv(p_s4).set_index('ID')['condition']

# Train seller profile
seller_stats = defaultdict(lambda: {'new': 0, 'used': 0})
with open(p_train, encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        s_id = r.get('seller_id')
        if s_id:
            seller_stats[s_id][r['condition']] += 1

print(f"Total profiled sellers in Train: {len(seller_stats)}")

# Analyze test items by seller profile
test_items = []
with open(p_test, encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        r = json.loads(line)
        s_id = r.get('seller_id')
        lt = r.get('listing_type_id')
        iq = r.get('initial_quantity', 1)
        w = str(r.get('warranty') or '').lower()
        off = r.get('official_store_id')
        
        n_train_new = seller_stats[s_id]['new'] if s_id else 0
        n_train_used = seller_stats[s_id]['used'] if s_id else 0
        tot_seller = n_train_new + n_train_used
        
        test_items.append({
            'ID': i,
            'title': r.get('title'),
            'price': float(r.get('price', 0) or 0),
            'cat': r.get('category_id'),
            'seller_id': s_id,
            'listing_type': lt,
            'iq': iq,
            'warranty': w,
            'official_store': off,
            'seller_train_new': n_train_new,
            'seller_train_used': n_train_used,
            'seller_tot': tot_seller,
            'curr_s4': s4[i]
        })

df_test = pd.DataFrame(test_items).set_index('ID')
print(f"Total Test Items: {len(df_test)}")

# 1. Check purely used sellers (at least 3 items in train, 100% used)
pure_used_sellers = df_test[(df_test['seller_train_used'] >= 3) & (df_test['seller_train_new'] == 0)]
print(f"\nTest items belonging to sellers with >= 3 train items (100% used): {len(pure_used_sellers)}")
print("Current predictions in S4 for these items:")
print(pure_used_sellers['curr_s4'].value_counts())

# Where S4 predicted 'new' but the seller is 100% used!
s4_new_pure_used = pure_used_sellers[pure_used_sellers['curr_s4'] == 'new']
print(f"\nDiscrepancies: S4 predicts NEW but seller is 100% USED in train: {len(s4_new_pure_used)}")
for idx, r in s4_new_pure_used.iloc[:20].iterrows():
    print(f"ID={idx:5d} Seller={r['seller_id']} (Used:{r['seller_train_used']} New:0) LT={r['listing_type']} IQ={r['iq']} ${r['price']:.0f} | {r['title'][:55]}")

# 2. Check purely new sellers (at least 5 items in train, 100% new)
pure_new_sellers = df_test[(df_test['seller_train_new'] >= 5) & (df_test['seller_train_used'] == 0)]
print(f"\nTest items belonging to sellers with >= 5 train items (100% new): {len(pure_new_sellers)}")
print("Current predictions in S4 for these items:")
print(pure_new_sellers['curr_s4'].value_counts())
s4_used_pure_new = pure_new_sellers[pure_new_sellers['curr_s4'] == 'used']
print(f"\nDiscrepancies: S4 predicts USED but seller is 100% NEW in train: {len(s4_used_pure_new)}")
for idx, r in s4_used_pure_new.iloc[:20].iterrows():
    print(f"ID={idx:5d} Seller={r['seller_id']} (Used:0 New:{r['seller_train_new']}) LT={r['listing_type']} IQ={r['iq']} ${r['price']:.0f} | {r['title'][:55]}")
