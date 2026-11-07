import json
from collections import Counter
from pathlib import Path

p_train = Path('data/ap_2026_q_2/raw/train_data.jsonlines')
p_test = Path('data/ap_2026_q_2/raw/test_data.jsonlines')

print("Analyzing Train Metadata Fields...")
train_sellers = Counter()
seller_cond = Counter()
official_stores = Counter()
initial_qty_cond = Counter()
shipping_mode_cond = Counter()
listing_type_cond = Counter()
warranty_cond = Counter()

train_count = 0
train_used = 0
with open(p_train, encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        train_count += 1
        c = r['condition']
        if c == 'used':
            train_used += 1
        
        s_id = r.get('seller_id')
        if s_id:
            train_sellers[s_id] += 1
            seller_cond[(s_id, c)] += 1
        
        off = r.get('official_store_id')
        if off:
            official_stores[c] += 1
            
        iq = r.get('initial_quantity', 1)
        if iq > 1:
            initial_qty_cond[('iq>1', c)] += 1
        else:
            initial_qty_cond[('iq=1', c)] += 1
            
        sm = (r.get('shipping') or {}).get('mode', 'none')
        shipping_mode_cond[(sm, c)] += 1
        
        lt = r.get('listing_type_id', 'none')
        listing_type_cond[(lt, c)] += 1
        
        w = r.get('warranty')
        if w:
            w_str = str(w).lower()
            if any(k in w_str for k in ['fabrica', 'meses', 'año', 'oficial']):
                warranty_cond[('w_new_like', c)] += 1
            elif any(k in w_str for k in ['prueba', 'funciona', 'sin', '48']):
                warranty_cond[('w_used_like', c)] += 1

print(f"Train Count: {train_count}, Used: {train_used} ({train_used/train_count:.3%})")
print("\nOfficial Stores condition breakdown:", dict(official_stores))
print("\nInitial Quantity breakdown:")
for k, v in initial_qty_cond.items():
    print(f"  {k}: {v}")

print("\nShipping Mode breakdown:")
for k, v in sorted(shipping_mode_cond.items()):
    print(f"  {k}: {v}")

print("\nListing Type breakdown:")
for k, v in sorted(listing_type_cond.items()):
    print(f"  {k}: {v}")

print("\nWarranty Keyword breakdown:")
for k, v in sorted(warranty_cond.items()):
    print(f"  {k}: {v}")

# Now check seller overlap with test!
print("\nChecking Seller Overlap between Train and Test...")
test_sellers = Counter()
with open(p_test, encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        s_id = r.get('seller_id')
        if s_id:
            test_sellers[s_id] += 1

overlap = set(train_sellers.keys()).intersection(set(test_sellers.keys()))
print(f"Unique Train Sellers: {len(train_sellers)}, Unique Test Sellers: {len(test_sellers)}")
print(f"Overlapping Sellers: {len(overlap)}")
