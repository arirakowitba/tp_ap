import pandas as pd
import numpy as np

c3 = pd.read_csv('submissions/today_20261105_portfolio_09480/submission_20261105_03_conservative_multimodal_top60.csv')
c4b = pd.read_csv('submissions/today_20261105_portfolio_09480/submission_20261105_04b_aggressive_top260.csv')

p_d1d2 = 'submissions/candidates_20261104/candidate_01_d1d2_plus_upper10.csv'
df_ref = pd.read_csv(p_d1d2)
UPPER10 = [4559, 4997, 6072, 6360, 6816, 8661, 14353, 14395, 28301, 28453]
d1d2 = df_ref.copy()
d1d2.loc[d1d2['ID'].isin(UPPER10), 'condition'] = 'used'

df_cb = pd.read_csv('results/full_native_nb_70k_20261105/test_probabilities.csv')
df_lgb = pd.read_csv('results/full_lightgbm_enriched_70k_20261105/lgb_enriched_test_probabilities.csv')
p_sub = np.load('results/frente_c_subword_and_diagnostics_20261104/test_probs_subword_rich_text.npy')

cb_pred = np.where(df_cb['p_native_text'] >= 0.5, 'used', 'new')
nb_pred = np.where(df_cb['p_nested_nb'] >= 0.5, 'used', 'new')
lgb_pred = np.where(df_lgb['p_lgb_enriched'] >= 0.5, 'used', 'new')
sub_pred = np.where(p_sub >= 0.5, 'used', 'new')

unanimous_3 = (cb_pred == nb_pred) & (nb_pred == lgb_pred)
unanimous_4 = (unanimous_3) & (sub_pred == cb_pred)

# Diff mask between C3 and C4b (the 200 items in B)
diff_mask = (c3['condition'] != c4b['condition'])
diff_ids = c3[diff_mask]['ID'].values
idx_b = [x - 1 for x in diff_ids]

b_df = pd.DataFrame({
    'ID': diff_ids,
    'd1d2': d1d2.loc[idx_b, 'condition'].values,
    'c4b_pred': c4b.loc[diff_mask, 'condition'].values,
    'p_native': df_cb.loc[idx_b, 'p_native_text'].values,
    'p_nested': df_cb.loc[idx_b, 'p_nested_nb'].values,
    'p_lgb': df_lgb.loc[idx_b, 'p_lgb_enriched'].values,
    'p_sub': p_sub[idx_b],
    'unanimous_3': unanimous_3[idx_b],
    'unanimous_4': unanimous_4[idx_b]
})

b_df['p_ens'] = 0.40 * b_df['p_native'] + 0.35 * b_df['p_nested'] + 0.15 * b_df['p_lgb'] + 0.10 * b_df['p_sub']
b_df['margin'] = (b_df['p_ens'] - 0.50).abs()
b_df = b_df.sort_values(by='margin', ascending=False).reset_index(drop=True)

print('=== ANALYSIS OF THE 200 ITEMS IN B (NET DELTA: -12 HITS) ===')
print('Total items:', len(b_df))
n_u3 = b_df['unanimous_3'].sum()
n_u4 = b_df['unanimous_4'].sum()
print(f'Unanimous 3-model: {n_u3} ({n_u3/len(b_df):.1%})')
print(f'Unanimous 4-model: {n_u4} ({n_u4/len(b_df):.1%})')

print('\nBy Direction:')
print(b_df['c4b_pred'].value_counts())

print('\nUnanimous 3 by Direction:')
print(pd.crosstab(b_df['c4b_pred'], b_df['unanimous_3']))

print('\nMargin by Unanimity:')
print('Unanimous 3 margin mean:', b_df[b_df['unanimous_3']]['margin'].mean(), 'min:', b_df[b_df['unanimous_3']]['margin'].min())
print('Discordant margin mean:', b_df[~b_df['unanimous_3']]['margin'].mean(), 'min:', b_df[~b_df['unanimous_3']]['margin'].min())

# Check how many are in top 50, next 50, etc.
b_df['rank_in_b'] = range(len(b_df))
print('\nIn Top 50 of B (highest margin):')
top50_b = b_df.head(50)
print('  Unanimous 3:', top50_b['unanimous_3'].sum(), '/ 50')
print('  Unanimous 4:', top50_b['unanimous_4'].sum(), '/ 50')
print('  Margin min:', top50_b['margin'].min(), 'max:', top50_b['margin'].max())
print('  Direction:', top50_b['c4b_pred'].value_counts().to_dict())

print('\nIn Next 50 of B (ranks 51-100 of B):')
next50_b = b_df.iloc[50:100]
print('  Unanimous 3:', next50_b['unanimous_3'].sum(), '/ 50')
print('  Unanimous 4:', next50_b['unanimous_4'].sum(), '/ 50')
print('  Margin min:', next50_b['margin'].min(), 'max:', next50_b['margin'].max())
print('  Direction:', next50_b['c4b_pred'].value_counts().to_dict())

print('\nIn Bottom 100 of B (ranks 101-200 of B):')
bot100_b = b_df.iloc[100:200]
print('  Unanimous 3:', bot100_b['unanimous_3'].sum(), '/ 100')
print('  Unanimous 4:', bot100_b['unanimous_4'].sum(), '/ 100')
print('  Margin min:', bot100_b['margin'].min(), 'max:', bot100_b['margin'].max())
print('  Direction:', bot100_b['c4b_pred'].value_counts().to_dict())
