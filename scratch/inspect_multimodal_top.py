import pandas as pd
df = pd.read_csv('results/unified_multimodal_confidence_matrix.csv').set_index('ID')
cands = df[
    (df['curr_s4'] == 'new') &
    (~df['is_toxic']) &
    (df['p_multimodal'] > 0.70) &
    (df['p_blend_103vec'] > 0.70)
].sort_values(by=['bayes_tests', 'p_multimodal', 'bayes_gain'], ascending=[False, False, False])

print(f"Top safe candidates total: {len(cands)}")
print(f"Tested >= 10: {len(cands[cands['bayes_tests'] >= 10])}")
print(f"Tested 5 to 9: {len(cands[cands['bayes_tests'].between(5, 9)])}")
print(f"Tested 3 to 4: {len(cands[cands['bayes_tests'].between(3, 4)])}")
print(f"Tested 1 to 2: {len(cands[cands['bayes_tests'].between(1, 2)])}")
print(f"Tested 0: {len(cands[cands['bayes_tests'] == 0])}")

print("\n" + "=" * 120)
print(f"{'#':2s} | {'ID':5s} | {'Tests':5s} | {'P_multi':7s} | {'P_blend':7s} | {'Gain':6s} | {'Price':6s} | {'Title'}")
print("=" * 120)
for i, (idx, r) in enumerate(cands.iloc[:75].iterrows(), 1):
    print(f"{i:2d} | {idx:5d} | {r['bayes_tests']:5.0f} | {r['p_multimodal']:7.3f} | {r['p_blend_103vec']:7.3f} | {r['bayes_gain']:+6.3f} | ${r['price']:5.0f} | {r['title'][:55]}")
