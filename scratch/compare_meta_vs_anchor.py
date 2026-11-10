import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scratch.analyze_all_vectors import anchor

# Load models
df_xgb = pd.read_csv('results/deep_xgboost_depth9_20261110/test_probabilities.csv')
p_xgb = 1.0 - df_xgb['prob_used'].values

lgb_npz = np.load('results/deep_lgbm_leaves127_20261110/lgbm_leaves127_predictions.npz')
p_lgb = 1.0 - lgb_npz['test_lgb']

bag_npz = np.load('results/multiseed_bagging_20261110/multiseed_bagged_predictions.npz')
p_bag = bag_npz['final_bagged_test']

p_meta = 0.65 * p_xgb + 0.30 * p_lgb + 0.05 * p_bag

# Compare with EVT-143
anchor_vec = np.array([1 if anchor['vec'][i] == 'new' else 0 for i in range(1, 30001)], dtype=np.int8)

meta_pred = (p_meta >= 0.5).astype(np.int8)

agreements = (meta_pred == anchor_vec).sum()
disagreements = 30000 - agreements
print(f"Meta-Blend (92.15% OOF) vs EVT-143 (94.761% Anchor):")
print(f"Agreements: {agreements} / 30,000 ({agreements/30000*100:.2f}%)")
print(f"Disagreements: {disagreements} / 30,000 ({disagreements/30000*100:.2f}%)")

# Let's inspect confidence of disagreements:
# Confidence = abs(p_meta - 0.5)
conf_dis = np.abs(p_meta - 0.5)[meta_pred != anchor_vec]
print(f"Disagreements confidence distribution:")
print(f"  Confidence >= 0.40 (P > 0.90 or P < 0.10): {(conf_dis >= 0.40).sum()}")
print(f"  Confidence >= 0.35 (P > 0.85 or P < 0.15): {(conf_dis >= 0.35).sum()}")
print(f"  Confidence >= 0.30 (P > 0.80 or P < 0.20): {(conf_dis >= 0.30).sum()}")
print(f"  Confidence >= 0.20 (P > 0.70 or P < 0.30): {(conf_dis >= 0.20).sum()}")
print(f"  Confidence < 0.10  (Marginal 0.40-0.60):    {(conf_dis < 0.10).sum()}")
