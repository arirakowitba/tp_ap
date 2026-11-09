import pandas as pd

p = 'submissions/portfolio_20261109_podium_attack/submission_20261109_01_surgical_rank5_capture_top5.csv'
anchor = 'submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv'
df_p = pd.read_csv(p)
df_a = pd.read_csv(anchor)
diff = (df_p['condition'] != df_a['condition'])
print(f'Total diffs: {diff.sum()}')
comp = pd.DataFrame({
    'ID': df_p.loc[diff, 'ID'],
    'anchor': df_a.loc[diff, 'condition'],
    'target': df_p.loc[diff, 'condition']
})
matrix = pd.read_csv('results/unified_multimodal_confidence_matrix.csv').set_index('ID')
for idx, r in comp.iterrows():
    row_m = matrix.loc[r['ID']]
    print(f"ID {r['ID']:5d} | {r['anchor']} -> {r['target']} | p_multi={row_m['p_multimodal']:.3f} | tests={row_m['bayes_tests']:.0f} | title: {row_m['title']}")
