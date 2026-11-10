import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd

print("Loading Deep XGBoost depth 9 test probabilities...")
df_xgb = pd.read_csv('results/deep_xgboost_depth9_20261110/test_probabilities.csv')

print("Loading Deep LGBM leaves 127 test probabilities...")
lgb_npz = np.load('results/deep_lgbm_leaves127_20261110/lgbm_leaves127_predictions.npz')
# Remember: lgbm was inverted (class 0 = new or 1 = new? Let's check alignment)
test_lgb = lgb_npz['test_lgb']

print(f"XGB shape: {df_xgb.shape}, LGB shape: {test_lgb.shape}")

# Let's inspect probabilities for our key items:
key_ids = [1477, 3777, 4256, 11421, 11779, 13224, 19857, 29386, 6627, 14438, 15441, 22299, 27888]

print(f"{'ID':5s} | {'XGB P(new)':10s} | {'LGB P(new)':10s} | {'Blend P(new)':12s}")
print("-" * 50)
for iid in key_ids:
    idx = iid - 1
    p_xgb = 1.0 - df_xgb.iloc[idx]['prob_used']
    # If lgb is inverted:
    p_lgb = 1.0 - test_lgb[idx]
    blend = 0.70 * p_xgb + 0.30 * p_lgb
    print(f"{iid:5d} | {p_xgb:10.4f} | {p_lgb:10.4f} | {blend:12.4f}")
