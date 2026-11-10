import json, os
import pandas as pd

with open('results/canonical_ledger_rebuild_20261110/canonical_events_catalog.json') as f:
    cat = json.load(f)

# Ladder of breakthrough anchors
ladder = [
    'EVT-041', # 0.94633
    'EVT-060', # 0.94647
    'EVT-063', # 0.94652
    'EVT-084', # 0.94661
    'EVT-118', # 0.94671
    'EVT-120', # 0.94709
    'EVT-127', # 0.94714
    'EVT-128', # 0.94719
    'EVT-129', # 0.94738
    'EVT-132', # 0.94752
    'EVT-143'  # 0.94761
]

events_by_id = {e['event_id']: e for e in cat}

for i in range(len(ladder) - 1):
    e_prev = events_by_id[ladder[i]]
    e_next = events_by_id[ladder[i+1]]
    p_prev = e_prev.get('vector_path')
    p_next = e_next.get('vector_path')
    
    if p_prev and p_next and os.path.exists(p_prev) and os.path.exists(p_next):
        df_prev = pd.read_csv(p_prev)
        df_next = pd.read_csv(p_next)
        col_cond_p = 'condition' if 'condition' in df_prev.columns else df_prev.columns[1]
        col_cond_n = 'condition' if 'condition' in df_next.columns else df_next.columns[1]
        col_id_p = 'ID' if 'ID' in df_prev.columns else 'id'
        
        diff = df_prev[df_prev[col_cond_p] != df_next[col_cond_n]]
        d_score = e_next['public_score'] - e_prev['public_score']
        d_hits = e_next['public_hits'] - e_prev['public_hits']
        
        n2u = len(df_prev[(df_prev[col_cond_p] == 'new') & (df_next[col_cond_n] == 'used')])
        u2n = len(df_prev[(df_prev[col_cond_p] == 'used') & (df_next[col_cond_n] == 'new')])
        
        print(f"{ladder[i]} ({e_prev['public_score']:.5f}) -> {ladder[i+1]} ({e_next['public_score']:.5f}) | ΔHits: {d_hits:+2d} | Flips: {len(diff):3d} (new->used: {n2u}, used->new: {u2n})")
    else:
        print(f"File check failed: {ladder[i]} -> {p_prev} | {ladder[i+1]} -> {p_next}")
