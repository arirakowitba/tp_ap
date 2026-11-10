"""Detailed analysis of the 161 elite disputed items.
"""
import json, os
import numpy as np
import pandas as pd

with open('results/canonical_ledger_rebuild_20261110/canonical_events_catalog.json') as f:
    cat = json.load(f)

elite_events = [e for e in cat if e.get('public_score') is not None and e.get('public_score') >= 0.94700 and e.get('vector_path') and os.path.exists(e.get('vector_path'))]
elite_events.sort(key=lambda x: x['public_score'])
print(f"Loaded {len(elite_events)} elite vectors.")

with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    test_rows = [json.loads(line) for line in f]
n_test = len(test_rows)

quad_df = pd.read_csv('results/sparse_text_seller_group_cv_20261110/test_probabilities_quad_titan.csv')
p_quad = quad_df['p_used_quad_titan'].values
p_tree = quad_df['p_used_tri_titan_trees'].values
p_txt = quad_df['p_used_sparse_text'].values

anchor_df = pd.read_csv('submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv')
col_cond = 'condition' if 'condition' in anchor_df.columns else anchor_df.columns[1]
anchor_labels = anchor_df[col_cond].values

matrix_elite = np.zeros((len(elite_events), n_test), dtype=np.int8)
for i, ev in enumerate(elite_events):
    df_v = pd.read_csv(ev['vector_path'])
    c_col = 'condition' if 'condition' in df_v.columns else df_v.columns[1]
    matrix_elite[i, :] = (df_v[c_col].values == 'used').astype(np.int8)

used_elite = np.sum(matrix_elite, axis=0)
disputed_idx = np.where((used_elite > 0) & (used_elite < len(elite_events)))[0]
print(f"Total disputed items in elite: {len(disputed_idx)}")

disputed_records = []
for idx in disputed_idx:
    item_id = idx + 1
    r = test_rows[idx]
    title = str(r.get('title') or '').strip()
    cat = str(r.get('category_id') or '')
    price = float(r.get('price') or 0.0)
    anc = anchor_labels[idx]
    u_count = used_elite[idx]
    n_count = len(elite_events) - u_count
    
    disputed_records.append({
        'ID': item_id,
        'title': title,
        'category_id': cat,
        'price': price,
        'anchor_evt143': anc,
        'elite_used_votes': int(u_count),
        'elite_new_votes': int(n_count),
        'p_quad_used': float(p_quad[idx]),
        'p_tree_used': float(p_tree[idx]),
        'p_text_used': float(p_txt[idx]),
        'agreement_anchor_quad': (anc == 'used' and p_quad[idx] >= 0.5) or (anc == 'new' and p_quad[idx] < 0.5),
        'margin_quad': float(abs(p_quad[idx] - 0.5))
    })

df_disp = pd.DataFrame(disputed_records)
print(f"Quad-Titan agrees with EVT-143 on {df_disp['agreement_anchor_quad'].sum()} of {len(df_disp)} disputed items ({df_disp['agreement_anchor_quad'].mean()*100:.1f}%)")

disagree_disp = df_disp[~df_disp['agreement_anchor_quad']].sort_values(by='margin_quad', ascending=False)
print(f"\nDisagreements between EVT-143 and Quad-Titan among Elite Disputed: {len(disagree_disp)}")
print("\nTop disagreements among elite disputed items (where Quad-Titan strongly challenges EVT-143):")
for idx, r in disagree_disp.head(20).iterrows():
    print(f"ID {r['ID']:5d} | Anchor: {r['anchor_evt143']:4s} | Votes: U={r['elite_used_votes']:2d}, N={r['elite_new_votes']:2d} | Quad: p_used={r['p_quad_used']:.3f} (tree={r['p_tree_used']:.3f}, txt={r['p_text_used']:.3f}) | ${r['price']:7.1f} | {r['title']}")
