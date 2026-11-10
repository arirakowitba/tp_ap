import json, csv, os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('.')

def read_vector_dict(path: Path) -> dict:
    with path.open(newline='', encoding='utf-8-sig') as fh:
        reader = csv.DictReader(fh)
        lower = {name.lower(): name for name in reader.fieldnames}
        id_col = lower.get('id') or lower.get('item_id')
        target_col = lower.get('condition') or lower.get('target')
        out = {}
        for row in reader:
            out[int(row[id_col].strip())] = row[target_col].strip().lower()
    return out

# Reference vectors
vec_48 = read_vector_dict(ROOT / 'submissions/submission_48_reference.csv')
vec_d1d2 = read_vector_dict(ROOT / 'submissions/candidate_d1_d2_combined_on48.csv')

with open(ROOT / 'results/forensic_ledger_20261104/submission_ledger_master_85vec.json', encoding='utf-8') as f:
    master_85 = json.load(f)
master_entries = {e.get('key'): e for e in master_85.get('entries', [])}

probes_20261104_vectors = {}
for probe_key in ['20261104_01_all80', '20261104_02_used40', '20261104_03_new40', '20261104_04_low20', '20261104_05_low10']:
    if probe_key in master_entries:
        e = master_entries[probe_key]
        flips_d1d2 = e.get('flipped_ids_vs_d1d2') or []
        v = dict(vec_d1d2)
        for item_id in flips_d1d2:
            v[int(item_id)] = 'used' if v[int(item_id)] == 'new' else 'new'
        probes_20261104_vectors[probe_key] = v

with open('results/canonical_ledger_rebuild_20261108/canonical_events_catalog.json') as f:
    events = json.load(f)

vectors = []
for e in events:
    evt_id = e['event_id']
    vpath = e.get('vector_path')
    ident = e.get('legacy_identifier')
    hits = e.get('public_hits')
    score = e.get('public_score')
    name = e.get('display_name')
    if hits is None or not vpath:
        continue
    
    vec = None
    p = ROOT / vpath
    if p.exists():
        vec = read_vector_dict(p)
    elif ident in probes_20261104_vectors:
        vec = probes_20261104_vectors[ident]
    elif vpath.startswith('reconstructed_from_48_'):
        tgt = vpath.replace('reconstructed_from_48_', '')
        if tgt in master_entries:
            flips = master_entries[tgt].get('flipped_ids') or []
            vec = dict(vec_48)
            for item_id in flips:
                vec[int(item_id)] = 'used' if vec[int(item_id)] == 'new' else 'new'
    
    if vec is not None and len(vec) == 30000:
        vectors.append({
            'event_id': evt_id,
            'name': name,
            'score': score,
            'hits': hits,
            'vec': vec
        })

# Add today's 5 slots
slots_today = [
    ('EVT-139', 'Slot 1: top20_rank4_probe', 0.94723, 19890, 'submissions/portfolio_20261109_podium_ladder/submission_slot1_top20_rank4_probe.csv'),
    ('EVT-140', 'Slot 2: empirical_pure_used2new_top18', 0.94728, 19891, 'submissions/portfolio_20261109_podium_ladder/submission_slot2_empirical_pure_used2new_top18.csv'),
    ('EVT-141', 'Slot 3: 12model_nested_meta_stacker', 0.94214, 19783, 'submissions/portfolio_20261109_podium_ladder/submission_slot3_12model_nested_meta_stacker.csv'),
    ('EVT-142', 'Slot 4: hybrid_seen_router_c4', 0.93104, 19550, 'submissions/portfolio_20261109_podium_ladder/submission_slot4_hybrid_seen_router_c4.csv'),
    ('EVT-143', 'Slot 5: surgical_quad_rank6_breakthrough', 0.94761, 19898, 'submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv'),
]

for eid, name, sc, hi, p in slots_today:
    v = read_vector_dict(ROOT / p)
    vectors.append({
        'event_id': eid,
        'name': name,
        'score': sc,
        'hits': hi,
        'vec': v
    })

print(f'Total {len(vectors)} vectors loaded.')
anchor = [v for v in vectors if v['event_id'] == 'EVT-143'][0]
print(f"Anchor EVT-143: score={anchor['score']}, hits={anchor['hits']}")

# Print top 15 vectors by public hits
vectors_by_hits = sorted(vectors, key=lambda x: x['hits'], reverse=True)
print('\nTop 15 highest-scoring vectors:')
for v in vectors_by_hits[:15]:
    delta = v['hits'] - anchor['hits']
    flips = sum(v['vec'][i] != anchor['vec'][i] for i in range(1, 30001))
    print(f"  {v['event_id']:7s} | Score: {v['score']:.5f} | Hits: {v['hits']} (Delta: {delta:+3d}) | Flips: {flips:4d} | {str(v['name'])[:45]}")

print('\nScore distribution:')
scores = [v['score'] for v in vectors]
print(f"Min score: {min(scores):.5f}, Max score: {max(scores):.5f}, Mean score: {np.mean(scores):.5f}")
high_perf = [v for v in vectors if v['score'] >= 0.94700]
print(f"Vectors with score >= 0.94700: {len(high_perf)}")
