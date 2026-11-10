import pandas as pd
import numpy as np

quad_df = pd.read_csv('results/sparse_text_seller_group_cv_20261110/test_probabilities_quad_titan.csv')
anchor_df = pd.read_csv('submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv')
post_df = pd.read_csv('results/item_posterior_confidences_and_patterns_30k.csv')

anc = anchor_df['condition'].values
p_quad = quad_df['p_used_quad_titan'].values
p_tree = quad_df['p_used_tri_titan_trees'].values
p_txt = quad_df['p_used_sparse_text'].values

pred_quad = np.where(p_quad >= 0.5, 'used', 'new')
margin_quad = np.abs(p_quad - 0.5)

dis_mask = (pred_quad != anc)
print(f"Total test items where Quad-Titan differs from EVT-143: {dis_mask.sum()}")

df_dis = pd.DataFrame({
    'ID': np.arange(1, len(anc) + 1)[dis_mask],
    'title': post_df.loc[dis_mask, 'title'].values,
    'price': post_df.loc[dis_mask, 'price'].values,
    'anchor': anc[dis_mask],
    'quad_label': pred_quad[dis_mask],
    'p_quad': p_quad[dis_mask],
    'margin_quad': margin_quad[dis_mask],
    'p_tree': p_tree[dis_mask],
    'p_txt': p_txt[dis_mask],
    'post_conf': post_df.loc[dis_mask, 'posterior_confidence'].values,
    'matched_pat': post_df.loc[dis_mask, 'matched_pattern'].values,
    'invariance': post_df.loc[dis_mask, 'invariance_tier'].values,
})

print("\n--- MARGIN DISTRIBUTION OF QUAD-TITAN DISAGREEMENTS ---")
for m_thresh in [0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00]:
    cnt = (df_dis['margin_quad'] >= m_thresh).sum()
    print(f"Margin >= {m_thresh:.2f}: {cnt:4d} items")

print("\n--- TOP 25 DISAGREEMENTS BY QUAD-TITAN MARGIN ---")
df_dis_sorted = df_dis.sort_values(by='margin_quad', ascending=False)
for idx, r in df_dis_sorted.head(25).reset_index().iterrows():
    print(f"Rank {idx+1:2d} | ID {r['ID']:5d} | {r['anchor']:4s} -> {r['quad_label']:4s} | Margin={r['margin_quad']:.3f} (p_quad={r['p_quad']:.3f}, p_tree={r['p_tree']:.3f}, p_txt={r['p_txt']:.3f}) | Pat={r['matched_pat']:<18} | ${r['price']:7.1f} | {r['title'][:45]}")
