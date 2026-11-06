import pandas as pd
import numpy as np

c2 = pd.read_csv('submissions/today_20261105_portfolio_09480/submission_20261105_03_conservative_multimodal_top60.csv')
p_d1d2 = 'submissions/candidates_20261104/candidate_01_d1d2_plus_upper10.csv'
df_ref = pd.read_csv(p_d1d2)
UPPER10 = [4559, 4997, 6072, 6360, 6816, 8661, 14353, 14395, 28301, 28453]
d1d2 = df_ref.copy()
d1d2.loc[d1d2['ID'].isin(UPPER10), 'condition'] = 'used'

df_cb = pd.read_csv('results/full_native_nb_70k_20261105/test_probabilities.csv')
df_lgb = pd.read_csv('results/full_lightgbm_enriched_70k_20261105/lgb_enriched_test_probabilities.csv')
p_sub = np.load('results/frente_c_subword_and_diagnostics_20261104/test_probs_subword_rich_text.npy')

df_id_ev = pd.read_csv('results/forensic_id_solver_20261105/id_level_target_evidence_all_probes.csv')
neg_ids = set(df_id_ev[df_id_ev['bounded_effect_delta'] < -0.35]['item_id'])

# Probability ensemble
p_ens = 0.40 * df_cb['p_native_text'] + 0.35 * df_cb['p_nested_nb'] + 0.15 * df_lgb['p_lgb_enriched'] + 0.10 * p_sub
margin = np.abs(p_ens - 0.50)

# Check which items are already flipped in Sub 2
c2_flips = set(c2[c2['condition'] != d1d2['condition']]['ID'])

cb_nat_pred = np.where(df_cb['p_native_text'] >= 0.5, 'used', 'new')
cb_nb_pred = np.where(df_cb['p_nested_nb'] >= 0.5, 'used', 'new')
lgb_pred = np.where(df_lgb['p_lgb_enriched'] >= 0.5, 'used', 'new')
sub_pred = np.where(p_sub >= 0.5, 'used', 'new')

candidates = []
for idx in range(30000):
    item_id = idx + 1
    d_lbl = d1d2.loc[idx, 'condition']
    if item_id in neg_ids or item_id in UPPER10 or item_id in c2_flips:
        continue
    # Agreement checks
    cb_agree = (cb_nat_pred[idx] == cb_nb_pred[idx])
    lgb_agree = (cb_agree and cb_nat_pred[idx] == lgb_pred[idx])
    all4_agree = (lgb_agree and cb_nat_pred[idx] == sub_pred[idx])
    
    # Model proposal
    m_lbl = 'used' if p_ens[idx] >= 0.5 else 'new'
    if m_lbl != d_lbl:
        candidates.append({
            'ID': item_id,
            'd1d2': d_lbl,
            'model_pred': m_lbl,
            'p_ens': p_ens[idx],
            'margin': margin[idx],
            'cb_agree': cb_agree,
            'lgb_agree': lgb_agree,
            'all4_agree': all4_agree,
            'p_nat': df_cb.loc[idx, 'p_native_text'],
            'p_nb': df_cb.loc[idx, 'p_nested_nb'],
            'p_lgb': df_lgb.loc[idx, 'p_lgb_enriched'],
            'p_sub': p_sub[idx]
        })

df_cand = pd.DataFrame(candidates).sort_values(by='margin', ascending=False).reset_index(drop=True)
print('Total candidate disagreements beyond Sub 2:', len(df_cand))
n_cb = df_cand['cb_agree'].sum()
n_lgb = df_cand['lgb_agree'].sum()
n_all4 = df_cand['all4_agree'].sum()
print(f'Candidates where CB Native and CB NB AGREE: {n_cb} ({n_cb/len(df_cand):.1%})')
print(f'Candidates where All 3 Trees AGREE: {n_lgb} ({n_lgb/len(df_cand):.1%})')
print(f'Candidates where All 4 Models AGREE: {n_all4} ({n_all4/len(df_cand):.1%})')

print('\nTop 50 candidates by margin:')
top50 = df_cand.head(50)
print('  Margin min:', top50['margin'].min(), 'max:', top50['margin'].max(), 'mean:', top50['margin'].mean())
print('  CB agree:', top50['cb_agree'].sum(), '/ 50')
print('  All 3 agree:', top50['lgb_agree'].sum(), '/ 50')
print('  All 4 agree:', top50['all4_agree'].sum(), '/ 50')

print('\nTop 100 candidates by margin:')
top100 = df_cand.head(100)
print('  Margin min:', top100['margin'].min(), 'max:', top100['margin'].max(), 'mean:', top100['margin'].mean())
print('  CB agree:', top100['cb_agree'].sum(), '/ 100')
print('  All 3 agree:', top100['lgb_agree'].sum(), '/ 100')
print('  All 4 agree:', top100['all4_agree'].sum(), '/ 100')

df_cand.to_csv('scratch/candidates_ranked_beyond_sub2.csv', index=False)
