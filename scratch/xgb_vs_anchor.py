import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scratch.analyze_all_vectors import anchor

df_xgb = pd.read_csv('results/deep_xgboost_depth9_20261110/test_probabilities.csv')

disagreements = []
for idx in range(30000):
    iid = idx + 1
    p_used = df_xgb.iloc[idx]['prob_used']
    p_new = 1.0 - p_used
    xgb_pred = 'new' if p_new >= 0.5 else 'used'
    anch = anchor['vec'][iid]
    if xgb_pred != anch:
        confidence = abs(p_new - 0.5)
        disagreements.append({
            'id': iid,
            'anchor': anch,
            'xgb_pred': xgb_pred,
            'p_new': p_new,
            'confidence': confidence
        })

print(f"Total items where XGB Depth 9 disagrees with EVT-143: {len(disagreements)} / 30000 ({len(disagreements)/30000*100:.2f}%)")
df_dis = pd.DataFrame(disagreements).sort_values('confidence', ascending=False)
print("\nTop 15 highest-confidence disagreements (XGB strongly disagrees with EVT-143):")
print(df_dis.head(15))
