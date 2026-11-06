import pandas as pd
import numpy as np
from pathlib import Path

# Load optimal blend probabilities (OOF 0.93551)
p_blend = pd.read_csv('results/full_f117_lgbm_20261105/test_probabilities_optimal_blend_oof09355.csv')
p_opt = p_blend['p_blend_opt'].values

# Load individual model probabilities
df_cont = pd.read_csv('results/continuous_training_20261105/continuous_test_probabilities_latest.csv')
p_cb = df_cont['p_catboost_deep'].values
p_sub = df_cont['p_subword_rich'].values

df_f117 = pd.read_csv('results/full_f117_lgbm_20261105/test_probabilities_f117_lgbm.csv')
p_lgb = df_f117['p_f117_lgbm'].values

# Load Sub 2 anchor (0.94709 = 19887 hits)
sub2 = pd.read_csv('submissions/today_20261105_portfolio_09480/submission_20261105_03_conservative_multimodal_top60.csv')
sub2_cond = sub2['condition'].values

# Load protected anchor IDs
df_ev = pd.read_csv('results/forensic_id_solver_20261105/id_level_target_evidence_all_90probes.csv')
neg_ids = set(df_ev[df_ev['bounded_effect_delta'] < -0.35]['item_id'])
UPPER10 = [4559, 4997, 6072, 6360, 6816, 8661, 14353, 14395, 28301, 28453]

# Model predictions
opt_pred = np.where(p_opt >= 0.5, 'used', 'new')
cb_pred = np.where(p_cb >= 0.5, 'used', 'new')
lgb_pred = np.where(p_lgb >= 0.5, 'used', 'new')
sub_pred = np.where(p_sub >= 0.5, 'used', 'new')

# Strict consensus: All models agree with the optimal blend!
strict_consensus = (cb_pred == lgb_pred) & (lgb_pred == opt_pred)
four_model_consensus = (strict_consensus) & (sub_pred == opt_pred)

print(f'Total test rows with Strict Consensus (CB, LGB, Blend): {strict_consensus.sum()} / 30000 ({strict_consensus.mean():.1%})')
print(f'Total test rows with 4-Model Consensus (CB, LGB, Sub, Blend): {four_model_consensus.sum()} / 30000 ({four_model_consensus.mean():.1%})')

# Disagreements with Sub 2 with strict consensus
strict_dis = []
for idx in range(30000):
    item_id = idx + 1
    if item_id in neg_ids or item_id in UPPER10:
        continue
    if strict_consensus[idx] and opt_pred[idx] != sub2_cond[idx]:
        margin_opt = abs(p_opt[idx] - 0.5)
        margin_cb = abs(p_cb[idx] - 0.5)
        margin_lgb = abs(p_lgb[idx] - 0.5)
        min_margin = min(margin_opt, margin_cb, margin_lgb)
        strict_dis.append({
            'item_id': item_id,
            'sub2_lbl': sub2_cond[idx],
            'model_lbl': opt_pred[idx],
            'margin_opt': margin_opt,
            'min_margin': min_margin,
            'is_4m': four_model_consensus[idx],
            'p_opt': p_opt[idx],
            'p_cb': p_cb[idx],
            'p_lgb': p_lgb[idx]
        })

df_strict = pd.DataFrame(strict_dis).sort_values(by='min_margin', ascending=False).reset_index(drop=True)
n_4m = df_strict['is_4m'].sum()
print(f'\nTotal Clean Disagreements with Sub 2 under STRICT CONSENSUS: {len(df_strict)}')
print(f'Where all 4 models agree: {n_4m}')
print('\nMargin quantiles of strict consensus disagreements:')
print(df_strict[['min_margin', 'margin_opt']].describe())

print('\nCutoffs by min_margin:')
print('Min margin >= 0.30:', (df_strict['min_margin'] >= 0.30).sum())
print('Min margin >= 0.25:', (df_strict['min_margin'] >= 0.25).sum())
print('Min margin >= 0.20:', (df_strict['min_margin'] >= 0.20).sum())
print('Min margin >= 0.15:', (df_strict['min_margin'] >= 0.15).sum())
print('Min margin >= 0.10:', (df_strict['min_margin'] >= 0.10).sum())
df_strict.to_csv('scratch/strict_consensus_disagreements.csv', index=False)
