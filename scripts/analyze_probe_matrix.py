import json
import pandas as pd
import numpy as np
import hashlib
import os

# 1. Load catalog
with open('results/canonical_ledger_rebuild_20261108/canonical_events_catalog.json', 'r', encoding='utf-8') as f:
    events = json.load(f)

# Collect all scored events with valid vectors
loaded_events = []
for e in events:
    score = e.get('public_score')
    hits = e.get('public_hits')
    vpath = e.get('vector_path')
    if score is not None and hits is not None and vpath and os.path.exists(vpath):
        loaded_events.append({
            'id': e.get('event_id'),
            'name': e.get('display_name'),
            'score': score,
            'hits': hits,
            'path': vpath
        })

# Add today's 5 slots
today_slots = [
    ('EVT-139', 'submission_slot1_top20_rank4_probe.csv', 0.94723, 19890, 'submissions/portfolio_20261109_podium_ladder/submission_slot1_top20_rank4_probe.csv'),
    ('EVT-140', 'submission_slot2_empirical_pure_used2new_top18.csv', 0.94728, 19891, 'submissions/portfolio_20261109_podium_ladder/submission_slot2_empirical_pure_used2new_top18.csv'),
    ('EVT-141', 'submission_slot3_12model_nested_meta_stacker.csv', 0.94214, 19783, 'submissions/portfolio_20261109_podium_ladder/submission_slot3_12model_nested_meta_stacker.csv'),
    ('EVT-142', 'submission_slot4_hybrid_seen_router_c4.csv', 0.93104, 19550, 'submissions/portfolio_20261109_podium_ladder/submission_slot4_hybrid_seen_router_c4.csv'),
    ('EVT-143', 'submission_slot5_surgical_quad_rank6_breakthrough.csv', 0.94761, 19898, 'submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv'),
]

for eid, name, score, hits, path in today_slots:
    if os.path.exists(path):
        loaded_events.append({
            'id': eid,
            'name': name,
            'score': score,
            'hits': hits,
            'path': path
        })

print(f"Total scored events loaded with vector files: {len(loaded_events)}")

# Load all vectors into a matrix (N_events x 30000)
# 'new' -> 1, 'used' -> 0
vectors = []
event_metadata = []
for ev in loaded_events:
    df = pd.read_csv(ev['path'])
    # Ensure correct column
    col = df.columns[1]
    v = (df[col].values == 'new').astype(np.int8)
    vectors.append(v)
    event_metadata.append(ev)

V = np.array(vectors) # shape: (M, 30000)
hits = np.array([ev['hits'] for ev in event_metadata])

print(f"Matrix V shape: {V.shape}")

# Compute number of items that vary across all these submissions
item_variance = np.var(V, axis=0)
varying_items = np.where(item_variance > 0)[0]
constant_items = np.where(item_variance == 0)[0]
print(f"Constant items across all submissions: {len(constant_items)}")
print(f"Items tested/flipped across submissions: {len(varying_items)}")
