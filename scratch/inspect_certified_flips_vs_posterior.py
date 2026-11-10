import pandas as pd

cert_df = pd.read_csv('results/certified_quad_titan_flips_20261110.csv')
post_df = pd.read_csv('results/item_posterior_confidences_and_patterns_30k.csv')

print(f"Certified flips count: {len(cert_df)}")
print("Columns in cert_df:", cert_df.columns.tolist())

merged = cert_df.merge(post_df[['ID', 'posterior_p_used', 'posterior_label', 'posterior_confidence', 'matched_pattern']], on='ID')
print("\nTop 20 Certified Flips merged with Posterior Confidences:")
for idx, r in merged.head(20).iterrows():
    print(f"ID {r['ID']:5d} | {r['anchor_evt143_condition']:4s} -> {r['certified_condition']:4s} | PostConf={r['posterior_confidence']*100:5.2f}% | PriorQuad={r['p_used_quad_titan']:.3f} | Pat={r['matched_pattern']:<18} | ${r['price']:7.1f} | {r['title'][:50]}")

agree = (merged['certified_condition'] == merged['posterior_label']).sum()
print(f"\nAgreement between Certified Flips and Bayesian Posterior: {agree}/{len(merged)} ({agree/len(merged)*100:.1f}%)")
