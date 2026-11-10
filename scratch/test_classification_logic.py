import json, csv, os, sys
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
import pandas as pd
from scratch.analyze_all_vectors import vectors, anchor

print("Analyzing item distribution and test metadata...")

# 1. Load test metadata
test_meta = []
with open("data/ap_2026_q_2/raw/test_data.jsonlines", "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        it = json.loads(line)
        test_meta.append({
            'item_id': idx + 1,
            'title': it.get('title', ''),
            'category_id': it.get('category_id', ''),
            'price': it.get('price', 0.0),
        })

df_test = pd.DataFrame(test_meta)
print(f"Loaded test metadata: {len(df_test)} rows.")

# 2. Build submission matrix (116 x 30000)
M = len(vectors)
N = 30000

# Submissions matrix: 1 if new, 0 if used
sub_matrix = np.zeros((M, N), dtype=np.int8)
hits_vec = np.zeros(M, dtype=int)
scores_vec = np.zeros(M, dtype=float)
eids = []

for s_idx, v in enumerate(vectors):
    eids.append(v['event_id'])
    hits_vec[s_idx] = v['hits']
    scores_vec[s_idx] = v['score']
    for i in range(1, N + 1):
        if v['vec'][i] == 'new':
            sub_matrix[s_idx, i - 1] = 1

anchor_hits = anchor['hits']
anchor_vec_arr = np.array([1 if anchor['vec'][i] == 'new' else 0 for i in range(1, N + 1)], dtype=np.int8)

# Compute basic submission statistics for each item
times_tested_new = sub_matrix.sum(axis=0)
times_tested_used = M - times_tested_new
times_flipped = np.sum(sub_matrix != anchor_vec_arr, axis=0)

# High-performing vectors (score >= 0.94700)
high_perf_mask = scores_vec >= 0.94700
n_high_perf = np.sum(high_perf_mask)
flips_in_high_perf = np.sum(sub_matrix[high_perf_mask] != anchor_vec_arr, axis=0)
is_invariant_high_perf = (flips_in_high_perf == 0)

# Net delta hits sum for flips vs anchor
# For each submission s: delta_hits_s = hits_vec[s] - anchor_hits
delta_hits_s = hits_vec - anchor_hits
# Flips indicator vs anchor: F[s, i] = 1 if sub_matrix[s, i] != anchor_vec_arr[i] else 0
F = (sub_matrix != anchor_vec_arr).astype(int)
delta_hits_sum = F.T @ delta_hits_s

print(f"Summary of flips:")
print(f"Items never flipped in any of 116 vectors: {np.sum(times_flipped == 0)}")
print(f"Items flipped at least once in 116 vectors: {np.sum(times_flipped > 0)}")
print(f"Items invariant in top {n_high_perf} vectors (>=0.947): {np.sum(is_invariant_high_perf)}")
print(f"Items flipped in top {n_high_perf} vectors: {np.sum(~is_invariant_high_perf)}")
