import pandas as pd
import numpy as np

post_df = pd.read_csv('results/item_posterior_confidences_and_patterns_30k.csv')
cert_df = pd.read_csv('results/certified_quad_titan_flips_20261110.csv')
cert_ids = set(cert_df['ID'])

dis_df = post_df[post_df['disagrees_with_anchor']].copy()
dis_df['in_certified_42'] = dis_df['ID'].isin(cert_ids)

# Check Quad titan margin
quad_df = pd.read_csv('results/sparse_text_seller_group_cv_20261110/test_probabilities_quad_titan.csv')
dis_df['p_tree'] = quad_df.loc[dis_df['ID'] - 1, 'p_used_tri_titan_trees'].values
dis_df['p_txt'] = quad_df.loc[dis_df['ID'] - 1, 'p_used_sparse_text'].values
dis_df['tree_dir'] = np.where(dis_df['p_tree'] >= 0.5, 'used', 'new')
dis_df['txt_dir'] = np.where(dis_df['p_txt'] >= 0.5, 'used', 'new')
dis_df['models_concordant'] = (dis_df['tree_dir'] == dis_df['txt_dir'])

# Flips that are concordant between tree and text, or have verified lexical pattern
dis_df['safe'] = dis_df['in_certified_42'] | (dis_df['models_concordant'] & (dis_df['posterior_confidence'] >= 0.85)) | (dis_df['matched_pattern'] != 'none')

print(f"Total disagreements: {len(dis_df)}")
print(f"Safe candidates: {dis_df['safe'].sum()}")

# Order by: certified first (by margin), then safe by posterior confidence
dis_df['rank_score'] = dis_df['in_certified_42'].astype(int) * 10.0 + dis_df['posterior_confidence']
dis_sorted = dis_df.sort_values(by='rank_score', ascending=False).reset_index(drop=True)

print("\n--- CANDIDATE TIERS ---")
tier1 = dis_sorted.iloc[:8]
tier2 = dis_sorted.iloc[:20]
tier3 = dis_sorted[dis_sorted['in_certified_42']] # exactly 42
tier4 = dis_sorted[dis_sorted['safe']].iloc[:75]
tier5 = dis_sorted[dis_sorted['safe']].iloc[:110]

for name, t in [("Tier 1 (Top 8 Surgical)", tier1),
                ("Tier 2 (Top 20 Surgical)", tier2),
                ("Tier 3 (Certified 42)", tier3),
                ("Tier 4 (Expanded 75 Safe)", tier4),
                ("Tier 5 (Podium Breakthrough 110)", tier5)]:
    n_flips = len(t)
    u2n = (t['anchor_evt143'] == 'used').sum()
    n2u = (t['anchor_evt143'] == 'new').sum()
    mean_conf = t['posterior_confidence'].mean()
    # Expected hits on public: n_flips * (20998 / 30000) * (2 * mean_conf - 1)
    exp_public_hits = n_flips * (20998.0 / 30000.0) * (2 * mean_conf - 1.0)
    exp_score = (19898 + exp_public_hits) / 20998.0
    print(f"{name:<35}: Flips={n_flips:3d} (U->N: {u2n:2d}, N->U: {n2u:2d}) | MeanConf={mean_conf*100:5.2f}% | ExpNetHits={exp_public_hits:+5.1f} | ExpScore={exp_score:.5f}")
