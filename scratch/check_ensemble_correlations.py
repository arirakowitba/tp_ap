import numpy as np
import pandas as pd

# Load Deep XGBoost Depth 9
df_xgb = pd.read_csv('results/deep_xgboost_depth9_20261110/test_probabilities.csv')
p_xgb_d9 = (1.0 - df_xgb['prob_used']).values # prob of new

# Load Deep LightGBM 127
lgb_npz = np.load('results/deep_lgbm_leaves127_20261110/lgbm_leaves127_predictions.npz')
p_lgb_127 = 1.0 - lgb_npz['test_lgb']

# Load Transductive Cycle 2
trans_npz = np.load('results/transductive_cycle2_20261110/transductive_cycle2_predictions.npz')
p_trans = trans_npz['trans_blend_test']
# Check orientation of trans_blend_test: mean prob
print(f"Mean probs: XGB_D9={p_xgb_d9.mean():.4f}, LGB_127={p_lgb_127.mean():.4f}, Trans={p_trans.mean():.4f}")

# Load Multiseed Bagging
bag_npz = np.load('results/multiseed_bagging_20261110/multiseed_bagged_predictions.npz')
p_bag = bag_npz['final_bagged_test']
print(f"Bagging mean prob: {p_bag.mean():.4f}")

# If trans or bag is oriented towards 'new' or 'used', let's check correlations:
df_corr = pd.DataFrame({
    'xgb_d9': p_xgb_d9,
    'lgb_127': p_lgb_127,
    'trans': p_trans,
    'bag': p_bag
})
print("\nCorrelation matrix between models:")
print(df_corr.corr())
