import pandas as pd
import numpy as np

c3 = pd.read_csv('submissions/today_20261105_portfolio_09480/submission_20261105_03_conservative_multimodal_top60.csv')
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

df_id_ev = pd.read_csv('results/forensic_id_solver_20261105/id_level_target_evidence_all_probes.csv')
neg_ids = set(df_id_ev[df_id_ev['bounded_effect_delta'] < -0.35]['item_id'])

# Ensemble probability
p_ens = 0.40 * df_cb['p_native_text'] + 0.35 * df_cb['p_nested_nb'] + 0.15 * df_lgb['p_lgb_enriched'] + 0.10 * p_sub
margin = np.abs(p_ens - 0.50)

# Find clean disagreements where models are unanimous
c3_flips = set(c3[c3['condition'] != d1d2['condition']]['ID'])

unan3_dis_new = []
for idx in range(30000):
    item_id = idx + 1
    d_lbl = d1d2.loc[idx, 'condition']
    if item_id in neg_ids or item_id in UPPER10 or item_id in c3_flips:
        continue
    if unanimous_3[idx] and cb_pred[idx] != d_lbl:
        unan3_dis_new.append({
            'item_id': item_id,
            'd1d2_lbl': d_lbl,
            'model_lbl': cb_pred[idx],
            'margin': margin[idx],
            'is_4_agree': unanimous_4[idx],
            'p_ens': p_ens[idx],
            'p_native': df_cb.loc[idx, 'p_native_text'],
            'p_nested': df_cb.loc[idx, 'p_nested_nb'],
            'p_lgb': df_lgb.loc[idx, 'p_lgb_enriched'],
            'p_sub': p_sub[idx]
        })

df_u3_new = pd.DataFrame(unan3_dis_new).sort_values(by='margin', ascending=False).reset_index(drop=True)
print(f'Total UNANIMOUS 3-model clean disagreements beyond C3: {len(df_u3_new)}')
print(f'Total UNANIMOUS 4-model clean disagreements beyond C3: {df_u3_new["is_4_agree"].sum()}')
print('\nMargin distribution:')
print(df_u3_new['margin'].describe())
print('\nDirections:')
print(df_u3_new['model_lbl'].value_counts())
print('\n4-model agree margin distribution:')
print(df_u3_new[df_u3_new['is_4_agree']]['margin'].describe())
print(df_u3_new[df_u3_new['is_4_agree']]['model_lbl'].value_counts())
