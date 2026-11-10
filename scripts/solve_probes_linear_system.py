import pandas as pd
import numpy as np
import json
from scipy.optimize import linprog

# Load test items
with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    items = [json.loads(line) for line in f]
test_items = {idx+1: it for idx, it in enumerate(items)}

vectors_info = [
    ('EVT-129', 19893, 'results/canonical_ledger_rebuild_20261108/evaluated_vectors_git_pinned/EVT-129_b8aa4c0_7b5dcb6f.csv'),
    ('EVT-130', 19893, 'results/canonical_ledger_rebuild_20261108/evaluated_vectors_git_pinned/EVT-130_78e5dd5_e80b5750.csv'),
    ('EVT-131', 19887, 'results/canonical_ledger_rebuild_20261108/evaluated_vectors_git_pinned/EVT-131_dc6d9ae_cb6c478c.csv'),
    ('EVT-132', 19896, 'results/canonical_ledger_rebuild_20261108/evaluated_vectors_git_pinned/EVT-132_23ff472_11aabc76.csv'),
    ('EVT-133', 19893, 'results/canonical_ledger_rebuild_20261108/evaluated_vectors_git_pinned/EVT-133_44e02fb_ffb7c7c7.csv'),
    ('EVT-134', 19892, 'results/canonical_ledger_rebuild_20261108/evaluated_vectors_git_pinned/EVT-134_b50a5e3_7ae780e9.csv'),
    ('EVT-135', 19896, 'results/canonical_ledger_rebuild_20261108/evaluated_vectors_git_pinned/EVT-135_3fa6c87_909a979d.csv'),
    ('EVT-136', 19895, 'results/canonical_ledger_rebuild_20261108/evaluated_vectors_git_pinned/EVT-136_c421bef_ab5ee340.csv'),
    ('EVT-137', 19892, 'results/canonical_ledger_rebuild_20261108/evaluated_vectors_git_pinned/EVT-137_4ae1b03_6c26e32d.csv'),
    ('EVT-138', 19893, 'results/canonical_ledger_rebuild_20261108/evaluated_vectors_git_pinned/EVT-138_6155654_11900c11.csv'),
    ('EVT-139', 19890, 'submissions/portfolio_20261109_podium_ladder/submission_slot1_top20_rank4_probe.csv'),
    ('EVT-140', 19891, 'submissions/portfolio_20261109_podium_ladder/submission_slot2_empirical_pure_used2new_top18.csv'),
    ('EVT-141', 19783, 'submissions/portfolio_20261109_podium_ladder/submission_slot3_12model_nested_meta_stacker.csv'),
    ('EVT-142', 19550, 'submissions/portfolio_20261109_podium_ladder/submission_slot4_hybrid_seen_router_c4.csv'),
    ('EVT-143', 19898, 'submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv'),
]

# Build diff matrix vs EVT-143
anchor_df = pd.read_csv('submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv')
anchor_p = anchor_df.set_index('ID')['condition'].to_dict()

# Matrix of flips:
# Let x_i in {0, 1} be whether EVT-143 is CORRECT on item i in public test.
# If EVT-143 is correct on item i (x_i = 1):
#   any flip of item i loses 1 hit (-1).
# If EVT-143 is wrong on item i (x_i = 0):
#   any flip of item i gains 1 hit (+1).
# So for any item i in public test flipped in submission k:
# delta_k = sum_{i in flips_k and i in public} ( (1 - x_i) - x_i ) = sum ( 1 - 2*x_i )
# If an item is NOT flipped, contribution is 0.

print("=== SUBSET FLIP OVERLAPS ===")
# Let's inspect the small flip vectors: EVT-132, 134, 135, 136, 137, 138, 139, 140, 143
small_events = [e for e in vectors_info if e[0] not in ['EVT-141', 'EVT-142']]

all_flips = {}
for eid, hits, path in small_events:
    df = pd.read_csv(path)
    p = df.set_index('ID')['condition'].to_dict()
    diffs = {k: (anchor_p[k], p[k]) for k in p if p[k] != anchor_p[k]}
    all_flips[eid] = diffs

# Let's count how many times each item is flipped across these 13 focused events
item_flips = {}
for eid, diffs in all_flips.items():
    for iid, (c_anc, c_other) in diffs.items():
        if iid not in item_flips:
            item_flips[iid] = []
        item_flips[iid].append((eid, c_anc, c_other))

print(f"Total unique items in focused events: {len(item_flips)}")
frequent_items = sorted(item_flips.items(), key=lambda x: len(x[1]), reverse=True)
for iid, flip_list in frequent_items[:25]:
    it = test_items[iid]
    print(f"ID {iid:5d} | Flipped {len(flip_list):2d} times | EVT-143={anchor_p[iid]:4s} | ${it.get('price', 0):8.1f} | {it.get('title', '')[:50]}")
