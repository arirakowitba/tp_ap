import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression, RidgeClassifier

# Load ground truth y_train
import json
with open('data/ap_2026_q_2/raw/train_data.jsonlines', encoding='utf-8') as f:
    y_train = np.array([1 if json.loads(line)['condition'] == 'new' else 0 for line in f], dtype=np.int8)

# Load OOFs
xgb_oof_df = pd.read_csv('results/deep_xgboost_depth9_20261110/oof_predictions.csv')
oof_xgb = (1.0 - xgb_oof_df['prob_used']).values

lgb_npz = np.load('results/deep_lgbm_leaves127_20261110/lgbm_leaves127_predictions.npz')
oof_lgb = 1.0 - lgb_npz['oof_lgb']

trans_npz = np.load('results/transductive_cycle2_20261110/transductive_cycle2_predictions.npz')
oof_trans = trans_npz['trans_blend_oof']

bag_npz = np.load('results/multiseed_bagging_20261110/multiseed_bagged_predictions.npz')
oof_bag = bag_npz['final_bagged_oof']

print(f"OOF Metrics on 70,000 Train Set:")
for name, arr in [('XGB Depth 9', oof_xgb), ('LGB 127 Leaves', oof_lgb), ('Transductive Cycle 2', oof_trans), ('Multiseed Bagging', oof_bag)]:
    acc = accuracy_score(y_train, (arr >= 0.5).astype(int))
    auc = roc_auc_score(y_train, arr)
    ll = log_loss(y_train, np.clip(arr, 1e-6, 1 - 1e-6))
    print(f" - {name:22s} | Acc: {acc:.5f} ({int(acc*70000)}/70000) | AUC: {auc:.5f} | LogLoss: {ll:.5f}")

# Stacking Design Matrix (70,000 x 4)
X_oof = np.column_stack([oof_xgb, oof_lgb, oof_trans, oof_bag])

# Simple convex blend optimization:
best_acc = 0
best_weights = None
for w_xgb in np.linspace(0.4, 0.8, 9):
    for w_lgb in np.linspace(0.1, 0.4, 7):
        for w_trans in np.linspace(0.0, 0.2, 5):
            w_bag = 1.0 - (w_xgb + w_lgb + w_trans)
            if w_bag < 0:
                continue
            blend = w_xgb * oof_xgb + w_lgb * oof_lgb + w_trans * oof_trans + w_bag * oof_bag
            acc = accuracy_score(y_train, (blend >= 0.5).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_weights = (w_xgb, w_lgb, w_trans, w_bag)

print(f"\nBest Convex Blend Weights (XGB, LGB, Trans, Bag): {best_weights}")
best_blend_oof = best_weights[0] * oof_xgb + best_weights[1] * oof_lgb + best_weights[2] * oof_trans + best_weights[3] * oof_bag
print(f"Best Blend OOF Acc: {best_acc:.5f} ({int(best_acc*70000)}/70000) | AUC: {roc_auc_score(y_train, best_blend_oof):.5f}")
