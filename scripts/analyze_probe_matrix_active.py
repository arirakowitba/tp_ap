import pandas as pd
import numpy as np
import json
import os

# Load the 54 verified vectors
with open('results/canonical_ledger_rebuild_20261108/canonical_events_catalog.json', 'r', encoding='utf-8') as f:
    events = json.load(f)

loaded = []
for e in events:
    score = e.get('public_score')
    hits = e.get('public_hits')
    vpath = e.get('vector_path')
    if score is not None and hits is not None and vpath and os.path.exists(vpath):
        loaded.append((e.get('event_id'), e.get('display_name'), hits, vpath))

today_slots = [
    ('EVT-139', 'submission_slot1_top20_rank4_probe.csv', 19890, 'submissions/portfolio_20261109_podium_ladder/submission_slot1_top20_rank4_probe.csv'),
    ('EVT-140', 'submission_slot2_empirical_pure_used2new_top18.csv', 19891, 'submissions/portfolio_20261109_podium_ladder/submission_slot2_empirical_pure_used2new_top18.csv'),
    ('EVT-141', 'submission_slot3_12model_nested_meta_stacker.csv', 19783, 'submissions/portfolio_20261109_podium_ladder/submission_slot3_12model_nested_meta_stacker.csv'),
    ('EVT-142', 'submission_slot4_hybrid_seen_router_c4.csv', 19550, 'submissions/portfolio_20261109_podium_ladder/submission_slot4_hybrid_seen_router_c4.csv'),
    ('EVT-143', 'submission_slot5_surgical_quad_rank6_breakthrough.csv', 19898, 'submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv'),
]
for t in today_slots:
    loaded.append(t)

# Reference anchor: EVT-143 (our current best, 19898 hits)
ref_idx = len(loaded) - 1 # EVT-143 is last
ref_df = pd.read_csv(loaded[ref_idx][3])
col = ref_df.columns[1]
ref_v = (ref_df[col].values == 'new').astype(np.int8)
ref_hits = loaded[ref_idx][2]

diff_matrix = []
delta_hits = []
event_names = []

for eid, name, h, path in loaded:
    df = pd.read_csv(path)
    v = (df[col].values == 'new').astype(np.int8)
    diff = v - ref_v # +1 if changed to new, -1 if changed to used, 0 if same
    diff_matrix.append(diff)
    delta_hits.append(h - ref_hits)
    event_names.append(f"{eid}: {name}")

D = np.array(diff_matrix) # shape: (54, 30000)
y = np.array(delta_hits)  # shape: (54,)

# Filter to columns that have at least one non-zero difference
active_cols = np.where(np.any(D != 0, axis=0))[0]
print(f"Active items with at least one flip vs EVT-143: {len(active_cols)}")

D_active = D[:, active_cols] # (54, N_active)
print(f"D_active shape: {D_active.shape}")
print(f"Rank of D_active: {np.linalg.matrix_rank(D_active)}")

# For each active item, how many times was it tested?
flip_counts = np.sum(np.abs(D_active), axis=0)
print(f"Items tested exactly 1 time: {np.sum(flip_counts == 1)}")
print(f"Items tested 2-5 times: {np.sum((flip_counts >= 2) & (flip_counts <= 5))}")
print(f"Items tested > 5 times: {np.sum(flip_counts > 5)}")
print(f"Max times an item was tested: {np.max(flip_counts)}")
