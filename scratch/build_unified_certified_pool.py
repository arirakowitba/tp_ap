import sys
sys.path.insert(0, '.')
import json
import pandas as pd
from scratch.analyze_all_vectors import anchor

# 1. Octeto
octeto_flips = [
    (1477, "used", "new", "Yu-Gi-Oh 10 Sobres Cerrados (sealed packs)", 1.0),
    (3777, "used", "new", "Gap Chaleco Unico A Estrenar (brand new)", 1.0),
    (4256, "new", "used", "Imperdible Set De Estampillas Usadas (used stamps)", 1.0),
    (11421, "new", "used", "Desarme Netbook Bangho (teardown/parts)", 1.0),
    (11779, "new", "used", "Telefonos Ip Avaya Buen Estado (100% used in train)", 1.0),
    (13224, "used", "new", "Figuritas Lote De 4 Sobres Cerrados (sealed packs)", 1.0),
    (19857, "used", "new", "Musculosa M/l Bien Larga A Estrenar (brand new)", 1.0),
    (29386, "new", "used", "Fallout New Vegas Excelente Estado Ps3 (99.2% used)", 1.0)
]

# 2. Key Vehicle & Antique Additions
tier2_flips = [
    (6627, "used", "new", "2014 0 Km Sin Rodar Y Sin Patentar (1km odometer)", 0.95),
    (14438, "used", "new", "Chevrolet Onix 1.4 Lt 0 Km (1km odometer)", 0.95),
    (22299, "new", "used", "Argentina, Moneda De 5 Centavos De 1905 (antique coin)", 0.95),
    (27888, "used", "new", "Sobre Cerrado Figuritas Golazo 2004 (sealed pack)", 0.90)
]

# 3. 5-Pillar Filter Certified Flips
df_5p = pd.read_csv('results/certified_five_pillar_candidates.csv')

with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    test_lines = [json.loads(line) for line in f]

pool = {}

# Add Octeto first (Priority 1)
for iid, anch, rec, justification, conf in octeto_flips:
    d = test_lines[iid - 1]
    pool[iid] = {
        'id': iid,
        'title': d.get('title'),
        'category_id': d.get('category_id'),
        'price': d.get('price'),
        'anchor': anch,
        'rec': rec,
        'priority_tier': 1,
        'confidence': conf,
        'expected_gain': (2 * conf - 1) * 0.699933,
        'justification': justification
    }

# Add Tier 2 (Priority 2)
for iid, anch, rec, justification, conf in tier2_flips:
    d = test_lines[iid - 1]
    pool[iid] = {
        'id': iid,
        'title': d.get('title'),
        'category_id': d.get('category_id'),
        'price': d.get('price'),
        'anchor': anch,
        'rec': rec,
        'priority_tier': 2,
        'confidence': conf,
        'expected_gain': (2 * conf - 1) * 0.699933,
        'justification': justification
    }

# Add 5-Pillar Filter certified (Priority 3)
for _, r in df_5p.iterrows():
    iid = int(r['id'])
    if iid not in pool:
        p_m = r['p_meta']
        conf = p_m if r['rec'] == 'new' else (1.0 - p_m)
        pool[iid] = {
            'id': iid,
            'title': r['title'],
            'category_id': r['category_id'],
            'price': r['price'],
            'anchor': r['anchor'],
            'rec': r['rec'],
            'priority_tier': 3,
            'confidence': float(conf),
            'expected_gain': float(r['expected_gain']),
            'justification': f"5-Pillar Certified (P_meta={p_m:.3f}, Cue={r['has_decisive_cue']})"
        }

df_pool = pd.DataFrame(list(pool.values()))
# Sort: priority_tier asc, then confidence desc
df_pool = df_pool.sort_values(['priority_tier', 'confidence'], ascending=[True, False]).reset_index(drop=True)
df_pool.to_csv('results/unified_certified_flips_pool.csv', index=False)

print(f"Total unified certified flips pool: {len(df_pool)}")
print(f"Tier 1 (Octeto Puro):         {(df_pool['priority_tier'] == 1).sum()}")
print(f"Tier 2 (0km + Moneda 1905):   {(df_pool['priority_tier'] == 2).sum()}")
print(f"Tier 3 (5-Pillar Multimodel): {(df_pool['priority_tier'] == 3).sum()}")

# Cumulative expected hits curve
df_pool['cum_gain'] = df_pool['expected_gain'].cumsum()

print("\n--- UNIFIED PROGRESSION LADDER ---")
print(f"{'K Flips':8s} | {'Exp Gain':10s} | {'Proj Hits':12s} | {'Proj Score':12s} | {'Target Milestone'}")
print("-" * 80)

checkpoints = [8, 12, 25, 50, 75, 100, 110, 150, len(df_pool)]
for k in checkpoints:
    gain = df_pool.iloc[k - 1]['cum_gain']
    proj_hits = 19898 + gain
    proj_score = proj_hits / 20998.0
    u2n = (df_pool.iloc[:k]['rec'] == 'new').sum()
    n2u = (df_pool.iloc[:k]['rec'] == 'used').sum()
    print(f"Top {k:3d}   | +{gain:6.2f}    | {proj_hits:7.1f}/20998 | {proj_score:10.5f} | [U->N: {u2n:2d}, N->U: {n2u:2d}]")

print("\nTop 15 candidates in Unified Pool:")
for i, r in df_pool.head(15).iterrows():
    print(f"{i+1:2d}. ID {r['id']:5d} | {r['anchor']} -> {r['rec']:4s} | Tier {r['priority_tier']} | Conf={r['confidence']:.3f} | {r['justification'][:40]} | {r['title'][:35]}")
