import pandas as pd
import numpy as np
import os

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

# Load all into dictionary
dfs = {}
preds = {}
for eid, hits, path in vectors_info:
    df = pd.read_csv(path)
    # Check column names
    col_id = [c for c in df.columns if c.lower() == 'id'][0]
    col_cond = [c for c in df.columns if c.lower() == 'condition'][0]
    p = df.set_index(col_id)[col_cond].to_dict()
    dfs[eid] = df
    preds[eid] = p

print("Loaded all 15 recent evaluated vectors.")

# Base anchor is EVT-143
anchor = 'EVT-143'
p_anchor = preds[anchor]

# Find all items that differ from anchor in at least one vector
all_diff_ids = set()
for eid, hits, path in vectors_info:
    p = preds[eid]
    diffs = [k for k in p_anchor if p_anchor[k] != p.get(k)]
    all_diff_ids.update(diffs)
    delta = hits - 19898
    print(f"{eid:8s} | Hits: {hits:5d} (Delta vs EVT-143: {delta:+3d}) | Flips vs EVT-143: {len(diffs):4d}")

all_diff_ids = sorted(list(all_diff_ids))
print(f"\nTotal unique items flipped across these 15 vectors: {len(all_diff_ids)}")
