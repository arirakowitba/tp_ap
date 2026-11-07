import pandas as pd
df = pd.read_csv('results/unified_multimodal_confidence_matrix.csv').set_index('ID')
unprobed = df[
    (df['curr_s4'] == 'new') &
    (~df['is_toxic']) &
    (df['bayes_tests'] == 0) &
    (df['p_multimodal'] > 0.70) &
    (df['p_blend_103vec'] > 0.80) &
    (df['price'] <= 5000)
].sort_values(by=['p_multimodal', 'p_blend_103vec'], ascending=[False, False])

print(f"Unprobed high confidence items (P_multi > 0.70, P_blend > 0.80): {len(unprobed)}")
for idx, r in unprobed.iloc[:25].iterrows():
    print(f"ID={idx:5d} P_multi={r['p_multimodal']:.3f} Pb={r['p_blend_103vec']:.3f} SellerUsed={r['seller_train_used']} ${r['price']:.0f} | {r['title'][:55]}")
