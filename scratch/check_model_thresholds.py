import pandas as pd
import numpy as np

p = 'results/transductive_deep_models/transductive_test_probabilities.csv'
df_probs = pd.read_csv(p).set_index('ID')
anchor = pd.read_csv('submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv').set_index('ID')['condition']

p_blend = df_probs['p_transductive_blend']

print("Differences vs anchor (0.94752):")
for th in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
    pred_lbl = np.where(p_blend >= th, 'used', 'new')
    diff = (pred_lbl != anchor).sum()
    u2n = ((anchor == 'used') & (pred_lbl == 'new')).sum()
    n2u = ((anchor == 'new') & (pred_lbl == 'used')).sum()
    print(f"Threshold {th:.2f}: total diff={diff:4d} | used->new={u2n:4d} | new->used={n2u:4d}")
