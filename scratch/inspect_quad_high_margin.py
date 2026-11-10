import pandas as pd
import numpy as np
import json

anchor_df = pd.read_csv('submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv')
quad_df = pd.read_csv('results/sparse_text_seller_group_cv_20261110/test_probabilities_quad_titan.csv')
quad_df['ID'] = 1 + np.arange(len(quad_df))
merged = pd.merge(anchor_df, quad_df, on='ID')
dis = merged[merged['condition'] != merged['pred_label_quad_titan']].copy()
dis['margin'] = np.abs(dis['p_used_quad_titan'] - 0.5)

high_dis = dis[dis['margin'] > 0.40].sort_values(by='margin', ascending=False)
print(f'Total high margin (>0.40) disagreements: {len(high_dis)}')

with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    test_rows = [json.loads(line) for line in f]

for idx, row in high_dis.head(25).iterrows():
    item_id = int(row['ID'])
    r = test_rows[item_id - 1]
    title = r.get('title')
    cat = r.get('category_id')
    price = r.get('price')
    print(f"ID {item_id:5d} | Anchor: {row['condition']:4s} -> Quad: {row['pred_label_quad_titan']:4s} (p_used={row['p_used_quad_titan']:.4f}, p_tree={row['p_used_tri_titan_trees']:.4f}, p_txt={row['p_used_sparse_text']:.4f}) | price={price} | {title}")
