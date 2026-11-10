import json, os
import pandas as pd
from collections import Counter

with open('results/canonical_ledger_rebuild_20261110/canonical_events_catalog.json') as f:
    cat = json.load(f)

events_by_id = {e['event_id']: e for e in cat}

ladder = ['EVT-118', 'EVT-120', 'EVT-127', 'EVT-128', 'EVT-129', 'EVT-132', 'EVT-143']

with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    test_rows = [json.loads(line) for line in f]

all_ladder_flips = []

for i in range(len(ladder) - 1):
    e1_id = ladder[i]
    e2_id = ladder[i+1]
    e1 = events_by_id[e1_id]
    e2 = events_by_id[e2_id]
    
    df1 = pd.read_csv(e1['vector_path'])
    df2 = pd.read_csv(e2['vector_path'])
    
    col1 = 'condition' if 'condition' in df1.columns else df1.columns[1]
    col2 = 'condition' if 'condition' in df2.columns else df2.columns[1]
    
    for idx, (c1, c2) in enumerate(zip(df1[col1], df2[col2])):
        if c1 != c2:
            item_id = idx + 1 # 1-based ID
            r = test_rows[idx]
            all_ladder_flips.append({
                'transition': f"{e1_id}->{e2_id}",
                'item_id': item_id,
                'from_cond': c1,
                'to_cond': c2,
                'title': str(r.get('title') or '').strip(),
                'category_id': str(r.get('category_id') or ''),
                'price': float(r.get('price') or 0.0),
                'seller_id': str(r.get('seller_id') or ''),
                'warranty': str(r.get('warranty') or ''),
                'delta_hits_step': e2['public_hits'] - e1['public_hits'],
                'step_flips_count': (df1[col1] != df2[col2]).sum()
            })

df_flips = pd.DataFrame(all_ladder_flips)
print(f"Total ladder flip events recorded: {len(df_flips)}")
print(f"Unique items flipped across the entire ladder: {df_flips['item_id'].nunique()}")

print("\n--- FLIPS BREAKDOWN BY STEP ---")
for t, grp in df_flips.groupby('transition', sort=False):
    d_hits = grp['delta_hits_step'].iloc[0]
    n_flips = len(grp)
    n2u = (grp['from_cond'] == 'new').sum()
    u2n = (grp['from_cond'] == 'used').sum()
    print(f"{t}: ΔHits={d_hits:+2d} | Flips={n_flips:2d} (new->used: {n2u}, used->new: {u2n})")

print("\n--- TOP CATEGORIES IN SUCCESSFUL FLIPS ---")
print(df_flips['category_id'].value_counts().head(10))

print("\n--- SAMPLE SUCCESSFUL FLIP ITEMS (EVT-128 -> EVT-129, +4 hits) ---")
for idx, r in df_flips[df_flips['transition'] == 'EVT-128->EVT-129'].head(10).iterrows():
    print(f"ID {r['item_id']:5d} | {r['from_cond']}->{r['to_cond']} | ${r['price']:7.1f} | {r['category_id']} | {r['title']}")

print("\n--- SAMPLE SUCCESSFUL FLIP ITEMS (EVT-129 -> EVT-132, +3 hits) ---")
for idx, r in df_flips[df_flips['transition'] == 'EVT-129->EVT-132'].head(10).iterrows():
    print(f"ID {r['item_id']:5d} | {r['from_cond']}->{r['to_cond']} | ${r['price']:7.1f} | {r['category_id']} | {r['title']}")

print("\n--- SUCCESSFUL FLIP ITEMS (EVT-132 -> EVT-143, +2 hits) ---")
for idx, r in df_flips[df_flips['transition'] == 'EVT-132->EVT-143'].iterrows():
    print(f"ID {r['item_id']:5d} | {r['from_cond']}->{r['to_cond']} | ${r['price']:7.1f} | {r['category_id']} | {r['title']}")
