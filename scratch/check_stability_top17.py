import sys
sys.path.insert(0, '.')
from scratch.analyze_all_vectors import vectors, anchor
import numpy as np

top_17 = [v for v in vectors if v['score'] >= 0.94700]
print(f"Top {len(top_17)} vectors with score >= 0.94700:")
for v in top_17:
    flips = sum(v['vec'][i] != anchor['vec'][i] for i in range(1, 30001))
    print(f"  {v['event_id']:7s} | {v['score']:.5f} | {v['hits']} | Flips vs EVT-143: {flips:3d} | {str(v['name'])[:40]}")

# Check which items are invariant across top 17
flips_in_top17 = np.zeros(30001, dtype=int)
for v in top_17:
    for i in range(1, 30001):
        if v['vec'][i] != anchor['vec'][i]:
            flips_in_top17[i] += 1

n_inv_top17 = int(np.sum(flips_in_top17[1:] == 0))
n_flips_top17 = int(np.sum(flips_in_top17[1:] > 0))
print(f"\nItems strictly invariant across all 17 top vectors (>94.7%): {n_inv_top17}")
print(f"Items flipped in at least one top vector: {n_flips_top17}")
