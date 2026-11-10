import pandas as pd
import json

with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    items = [json.loads(line) for line in f]
test_items = {idx+1: it for idx, it in enumerate(items)}

evt132 = pd.read_csv('results/canonical_ledger_rebuild_20261108/evaluated_vectors_git_pinned/EVT-132_23ff472_11aabc76.csv')
p132 = evt132.set_index('ID')['condition'].to_dict()

events = [
    ('EVT-134', 'results/canonical_ledger_rebuild_20261108/evaluated_vectors_git_pinned/EVT-134_b50a5e3_7ae780e9.csv', 19892, -4),
    ('EVT-135', 'results/canonical_ledger_rebuild_20261108/evaluated_vectors_git_pinned/EVT-135_3fa6c87_909a979d.csv', 19896, 0),
    ('EVT-136', 'results/canonical_ledger_rebuild_20261108/evaluated_vectors_git_pinned/EVT-136_c421bef_ab5ee340.csv', 19895, -1),
    ('EVT-137', 'results/canonical_ledger_rebuild_20261108/evaluated_vectors_git_pinned/EVT-137_4ae1b03_6c26e32d.csv', 19892, -4),
    ('EVT-138', 'results/canonical_ledger_rebuild_20261108/evaluated_vectors_git_pinned/EVT-138_6155654_11900c11.csv', 19893, -3),
]

for eid, path, hits, delta in events:
    df = pd.read_csv(path)
    p = df.set_index('ID')['condition'].to_dict()
    diffs = {k: (p132[k], p[k]) for k in p if p[k] != p132[k]}
    print(f"\n{eid} | Hits: {hits} | Delta vs EVT-132: {delta:+d} | Flips: {len(diffs)}")
    for k, (c_old, c_new) in diffs.items():
        it = test_items[k]
        print(f"  ID {k:5d} ({c_old} -> {c_new}): ${it.get('price', 0):8.1f} | {it.get('title', '')[:50]}")
