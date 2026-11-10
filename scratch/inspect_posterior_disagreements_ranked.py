import pandas as pd
import json

post_df = pd.read_csv('results/item_posterior_confidences_and_patterns_30k.csv')
cert_df = pd.read_csv('results/certified_quad_titan_flips_20261110.csv')
cert_ids = set(cert_df['ID'])

dis_df = post_df[post_df['disagrees_with_anchor']].copy()
dis_df['is_in_certified_42'] = dis_df['ID'].isin(cert_ids)

# Sort by posterior confidence descending
dis_df = dis_df.sort_values(by='posterior_confidence', ascending=False)

print(f"Total posterior disagreements vs EVT-143: {len(dis_df)}")
print(f"Disagreements in certified 42: {dis_df['is_in_certified_42'].sum()}/42")

print("\n--- DISTRIBUTION BY POSTERIOR CONFIDENCE ---")
conf_brackets = [(0.99, 1.00), (0.95, 0.99), (0.90, 0.95), (0.80, 0.90), (0.70, 0.80), (0.60, 0.70), (0.50, 0.60)]
for low, high in conf_brackets:
    cnt = ((dis_df['posterior_confidence'] >= low) & (dis_df['posterior_confidence'] < high)).sum()
    in_cert = ((dis_df['posterior_confidence'] >= low) & (dis_df['posterior_confidence'] < high) & dis_df['is_in_certified_42']).sum()
    print(f"Conf [{low*100:5.1f}%, {high*100:5.1f}%): {cnt:4d} items | in certified 42: {in_cert:3d}")

print("\n--- TOP 30 POSTERIOR DISAGREEMENTS OVERALL ---")
for idx, r in dis_df.head(30).reset_index().iterrows():
    cert_tag = " [CERT-42]" if r['is_in_certified_42'] else " [BAYES]"
    print(f"Rank {idx+1:2d} | ID {r['ID']:5d} | {r['anchor_evt143']:4s} -> {r['posterior_label']:4s} | PostConf={r['posterior_confidence']*100:5.2f}% | PriorQuad={r['prior_p_used_quad']:.3f} | Pat={r['matched_pattern']:<18}{cert_tag} | ${r['price']:7.1f} | {r['title'][:45]}")
