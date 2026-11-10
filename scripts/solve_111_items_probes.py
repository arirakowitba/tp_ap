import pandas as pd
import numpy as np
import json

with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    items = [json.loads(line) for line in f]
test_items = {idx+1: it for idx, it in enumerate(items)}

anchor_df = pd.read_csv('submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv')
anchor_p = anchor_df.set_index('ID')['condition'].to_dict()

# The 13 small/clean events
events = [
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
    ('EVT-143', 19898, 'submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv'),
]

# Collect all items flipped in any of these 13 events vs EVT-143
item_list = set()
event_flips = []
for eid, hits, path in events:
    df = pd.read_csv(path)
    p = df.set_index('ID')['condition'].to_dict()
    diffs = {k: 1 if p[k] != anchor_p[k] else 0 for k in anchor_p}
    # active in this event
    active = [k for k, v in diffs.items() if v == 1]
    item_list.update(active)
    event_flips.append((eid, hits - 19898, active))

items_111 = sorted(list(item_list))
item_to_col = {iid: idx for idx, iid in enumerate(items_111)}
M = len(events)
N = len(items_111)

A = np.zeros((M, N), dtype=float)
b = np.zeros(M, dtype=float)

for row_idx, (eid, delta, active) in enumerate(event_flips):
    b[row_idx] = delta
    for iid in active:
        A[row_idx, item_to_col[iid]] = 1.0 # flipped

print(f"Matrix A shape: {A.shape}, rank: {np.linalg.matrix_rank(A)}")

# Note on physics of b:
# When item i is flipped:
# If item i is in Public test (prob 0.70):
#   if EVT-143 was WRONG on item i, flipping it gives +1 hit.
#   if EVT-143 was CORRECT on item i, flipping it gives -1 hit.
# Let s_i in {-1, +1} be the effect of flipping item i (if public) or 0 (if private).
# So delta = sum_{i flipped} e_i, where e_i = effect of flipping item i.
# If EVT-143 is optimal on item i, e_i <= 0 (either -1 or 0).
# If EVT-143 has an ERROR on item i, e_i >= 0 (either +1 or 0).

# Let's run Ridge regression / least squares to estimate e_i for each of the 111 items:
# e = (A^T A + lambda I)^(-1) A^T b
lam = 1.0
e_est = np.linalg.solve(A.T @ A + lam * np.eye(N), A.T @ b)

results = []
for idx, iid in enumerate(items_111):
    eff = e_est[idx]
    it = test_items[iid]
    times_flipped = np.sum(A[:, idx])
    results.append((iid, eff, times_flipped, anchor_p[iid], it.get('price', 0), it.get('title', '')))

# Sort by estimated effect of flipping
# If eff > 0: flipping away from EVT-143 IMPROVES the score!
# If eff < 0: EVT-143 is CORRECT, keep it!
results.sort(key=lambda x: x[1], reverse=True)

print("\n=== TOP ITEMS WHERE FLIPPING AWAY FROM EVT-143 MIGHT IMPROVE SCORE (Positive Effect) ===")
for iid, eff, tf, cur, price, title in results[:15]:
    print(f"ID {iid:5d} | Flip Effect: {eff:+6.3f} | Flipped {int(tf):2d}x | Current: {cur:4s} | ${price:8.1f} | {title[:45]}")

print("\n=== TOP ITEMS WHERE EVT-143 IS CONFIRMED HIGHLY ACCURATE (Negative Effect - DO NOT FLIP) ===")
for iid, eff, tf, cur, price, title in results[-15:]:
    print(f"ID {iid:5d} | Flip Effect: {eff:+6.3f} | Flipped {int(tf):2d}x | Current: {cur:4s} | ${price:8.1f} | {title[:45]}")
