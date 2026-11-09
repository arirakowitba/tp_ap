import numpy as np
import pandas as pd

anchor_path = 'submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv'
df_anc = pd.read_csv(anchor_path).rename(columns={'ID': 'id', 'condition': 'target'})
df_anc['id'] = df_anc['id'].astype(int)
anc_map = dict(zip(df_anc['id'], df_anc['target']))

df_lgb = pd.read_csv('results/high_capacity_lgbm_5fold/test_probabilities.csv')
print("Loaded LightGBM test predictions:", len(df_lgb))
print(df_lgb.head())

# Threshold analysis
probs = df_lgb['prob_used'].to_numpy()
print("\nProbabilities summary:")
print(f"Min: {probs.min():.4f}, Max: {probs.max():.4f}, Mean: {probs.mean():.4f}, Median: {np.median(probs):.4f}")

# How many diffs vs anchor at various thresholds
for t in [0.45, 0.48, 0.50, 0.51, 0.52, 0.55, 0.60]:
    pred = np.where(probs >= t, 'used', 'new')
    diff_u2n = sum((pred == 'new') & (df_anc['target'] == 'used'))
    diff_n2u = sum((pred == 'used') & (df_anc['target'] == 'new'))
    print(f"Threshold t={t:.2f}: Total used={sum(pred=='used')}, Flips vs Anchor: used->new = {diff_u2n}, new->used = {diff_n2u}")

# Let's inspect the highest-confidence new->used flips from LightGBM!
diff_mask = (df_anc['target'] == 'new') & (probs > 0.50)
df_diff = df_lgb[diff_mask].copy()
df_diff['prob_used'] = probs[diff_mask]
df_diff = df_diff.sort_values('prob_used', ascending=False)
print(f"\nTotal new->used candidate flips (anchor is new, LightGBM says used): {len(df_diff)}")
print("\nTop 20 highest-confidence new->used candidates:")
print(df_diff[['ID', 'prob_used']].head(20).to_string(index=False))
