import json, os
import pandas as pd

with open('results/canonical_ledger_rebuild_20261110/canonical_events_catalog.json') as f:
    cat = json.load(f)

events_by_id = {e['event_id']: e for e in cat}

# Let's inspect failed probes vs prior anchor
failed_probes = [
    ('EVT-132', 'EVT-134', -4, 'top 5 surgical'),
    ('EVT-132', 'EVT-137', -4, 'top 15 attack'),
    ('EVT-132', 'EVT-139', -6, 'slot 1 top 20 new->used'),
    ('EVT-132', 'EVT-140', -5, 'slot 2 top 18 used->new')
]

with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    test_rows = [json.loads(line) for line in f]

for e_anc_id, e_prb_id, d_hits, desc in failed_probes:
    e_anc = events_by_id[e_anc_id]
    e_prb = events_by_id[e_prb_id]
    
    df_anc = pd.read_csv(e_anc['vector_path'])
    df_prb = pd.read_csv(e_prb['vector_path'])
    
    col_a = 'condition' if 'condition' in df_anc.columns else df_anc.columns[1]
    col_p = 'condition' if 'condition' in df_prb.columns else df_prb.columns[1]
    
    diff_idx = [i for i, (ca, cp) in enumerate(zip(df_anc[col_a], df_prb[col_p])) if ca != cp]
    print(f"\n=======================================================")
    print(f"FAILED PROBE: {e_prb_id} vs {e_anc_id} ({desc}) | ΔHits: {d_hits:+d} | Total Flips: {len(diff_idx)}")
    print(f"=======================================================")
    
    for idx in diff_idx[:10]:
        item_id = idx + 1
        r = test_rows[idx]
        ca = df_anc[col_a].iloc[idx]
        cp = df_prb[col_p].iloc[idx]
        title = str(r.get('title') or '').strip()
        cat = str(r.get('category_id') or '')
        price = float(r.get('price') or 0.0)
        print(f"ID {item_id:5d} | {ca:4s} -> {cp:4s} | ${price:7.1f} | {cat} | {title}")
