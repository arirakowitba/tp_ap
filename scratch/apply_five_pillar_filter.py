import sys
sys.path.insert(0, '.')
import json, re
import pandas as pd
import numpy as np
from collections import Counter
from scratch.analyze_all_vectors import anchor, vectors

print("Loading candidate flips and datasets for 5-Pillar Verification...")
df_flips = pd.read_csv('results/ranked_candidate_flips.csv')

with open('data/ap_2026_q_2/raw/train_data.jsonlines', encoding='utf-8') as f:
    train_data = [json.loads(line) for line in f]

with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    test_lines = [json.loads(line) for line in f]

# Compute category base rates in train
cat_counts = Counter(d['category_id'] for d in train_data)
cat_new = Counter(d['category_id'] for d in train_data if d['condition'] == 'new')
cat_rates = {c: cat_new[c] / cat_counts[c] for c in cat_counts if cat_counts[c] >= 5}

# Regex definitions
used_strong_cues = [
    r'\bbuen estado\b', r'\bexcelente estado\b', r'\bmuy buen estado\b',
    r'\bimpecable estado\b', r'\bdesarme\b', r'\bpara repuestos?\b',
    r'\bdespiece\b', r'\bno funciona\b', r'\ba reparar\b', r'\bpara reparar\b',
    r'\bestampillas usadas\b', r'\busad[oa]s?\b', r'\bcon detalles de uso\b',
    r'\blote de revistas\b', r'\blibro usado\b', r'\bsegunda mano\b'
]

new_strong_cues = [
    r'\ba estrenar\b', r'\bsin rodar\b', r'\ben caja cerrada\b',
    r'\bsellad[oa] de fabrica\b', r'\bsobres cerrados\b', r'\bsobre cerrado\b',
    r'\ben blister cerrado\b', r'\b0\s*km\b', r'\bcero\s*km\b',
    r'\blibro digital\b', r'\be-?book\b'
]

negations_of_new = [
    r'\bigual a 0\s*km\b', r'\bcomo 0\s*km\b', r'\bparece 0\s*km\b',
    r'\bigual a nuevo\b', r'\bcomo nuevo\b', r'\bparece nuevo\b',
    r'\bcon baja\b', r'\bdado de baja\b', r'\bpara desarme\b'
]

passed_candidates = []
quarantined = []

for idx, r in df_flips.iterrows():
    iid = int(r['id'])
    d = test_lines[iid - 1]
    title = str(d.get('title', '')).lower()
    cid = d.get('category_id')
    price = float(d.get('price') or 0.0)
    flip = f"{r['anchor']} -> {r['rec']}"
    target_label = r['rec']
    
    attrs = {a['name']: a['value_name'] for a in d.get('attributes', [])}
    km_str = attrs.get('Kilómetros')
    ano_str = attrs.get('Año')
    
    # Pillar 1: Multi-model conviction
    p_meta = r['p_meta']
    p_xgb = r['p_xgb']
    p_lgb = r['p_lgb']
    
    # Must have both trees pointing in same direction
    if target_label == 'new' and (p_xgb < 0.50 or p_lgb < 0.50):
        quarantined.append((iid, flip, "Pillar 1: Disagreement between XGB and LGB", title))
        continue
    if target_label == 'used' and (p_xgb > 0.50 or p_lgb > 0.50):
        quarantined.append((iid, flip, "Pillar 1: Disagreement between XGB and LGB", title))
        continue
        
    # Pillar 2: Category empirical check
    cat_rate = cat_rates.get(cid, 0.5)
    # If category is extreme and flip contradicts it without decisive lexical cue
    has_decisive_cue = False
    if target_label == 'new':
        has_decisive_cue = any(bool(re.search(pat, title)) for pat in new_strong_cues)
    else:
        has_decisive_cue = any(bool(re.search(pat, title)) for pat in used_strong_cues)
        
    # Block pet services / stud from being flipped to used
    if cid in ['MLA8041', 'MLA8323'] and target_label == 'used':
        quarantined.append((iid, flip, "Pillar 2: Pet/Stud category (100% new in train)", title))
        continue
        
    # Pillar 3: Negation and false positive filter
    if target_label == 'new':
        is_negated = any(bool(re.search(pat, title)) for pat in negations_of_new)
        if is_negated:
            quarantined.append((iid, flip, "Pillar 3: Negation of new ('igual a 0km', etc.)", title))
            continue
            
    # Check false positive words
    if 'abusada' in title and target_label == 'used' and not any(bool(re.search(pat, title)) for pat in used_strong_cues if pat != r'\busad[oa]s?\b'):
        quarantined.append((iid, flip, "Pillar 3: False positive substring (abusada)", title))
        continue
        
    # Pillar 4: Vehicle odometer check
    if km_str is not None:
        try:
            km_val = float(str(km_str).replace('.', '').replace(',', '').replace(' km', '').strip())
        except:
            km_val = -1
        if target_label == 'new' and km_val > 10:
            quarantined.append((iid, flip, f"Pillar 4: Vehicle has {km_val} km > 10", title))
            continue

    # Pillar 5: Elite Probe Quarantine
    # If this item was tested in the Top 17 vectors and failed
    # Check if Top 17 unanimously chose anchor
    # If Top 17 has 17/0 for anchor and confidence is < 0.85, quarantine
    if (r['top17_new_cnt'] == 17 or r['top17_used_cnt'] == 17) and abs(p_meta - 0.5) < 0.35 and not has_decisive_cue:
        quarantined.append((iid, flip, "Pillar 5: Top 17 elite consensus against flip without decisive cue", title))
        continue

    # If it passed all pillars:
    pillar_score = abs(p_meta - 0.5) + (0.20 if has_decisive_cue else 0.0)
    passed_candidates.append({
        'id': iid,
        'title': d.get('title'),
        'category_id': cid,
        'price': price,
        'anchor': r['anchor'],
        'rec': r['rec'],
        'p_meta': p_meta,
        'p_xgb': p_xgb,
        'p_lgb': p_lgb,
        'expected_gain': r['expected_gain'],
        'has_decisive_cue': has_decisive_cue,
        'pillar_score': pillar_score,
        'top17_new_cnt': r['top17_new_cnt'],
        'top17_used_cnt': r['top17_used_cnt']
    })

print(f"\nAudit Complete:")
print(f"Total candidates evaluated: {len(df_flips)}")
print(f"Total passed 5 pillars:     {len(passed_candidates)}")
print(f"Total quarantined:          {len(quarantined)}")

df_passed = pd.DataFrame(passed_candidates).sort_values('pillar_score', ascending=False).reset_index(drop=True)
df_passed.to_csv('results/certified_five_pillar_candidates.csv', index=False)
print(f"Saved certified candidates to results/certified_five_pillar_candidates.csv")

print("\n--- SAMPLE OF QUARANTINED CANDIDATES (False positives avoided) ---")
for q in quarantined[:10]:
    print(f"ID {q[0]:5d} | Flip: {q[1]:13s} | Reason: {q[2]:55s} | Title: {q[3][:40]}")

print("\n--- TOP 25 CERTIFIED CANDIDATES (Highest Conviction) ---")
for i, r in df_passed.head(25).iterrows():
    cue_str = " [CUE]" if r['has_decisive_cue'] else ""
    print(f"{i+1:2d}. ID {r['id']:5d} | {r['anchor']} -> {r['rec']} | P={r['p_meta']:.3f} | Score={r['pillar_score']:.3f}{cue_str} | Cat: {r['category_id']} | ${r['price']} | {r['title'][:40]}")
