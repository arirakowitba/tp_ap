import pandas as pd

df_fusion = pd.read_csv('results/cross_referenced_bayesian_fusion/joint_information_fusion_all30k.csv')
s4 = pd.read_csv('submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv').set_index('ID')['condition']

# Safe candidates
cands = df_fusion[
    (df_fusion["joint_value"] > 0) &
    (df_fusion["is_safe"])
].sort_values(by="joint_value", ascending=False).copy()

# How many are still 'new' in s4?
cands['s4_cond'] = cands['item_id'].map(s4)
cands_new_in_s4 = cands[cands['s4_cond'] == 'new'].copy()

print(f"Total safe cands with joint_value > 0: {len(cands)}")
print(f"Candidates that are STILL 'new' in anchor s4: {len(cands_new_in_s4)}")
print("\nTop 20 candidates STILL 'new' in s4:")
cols = ['item_id', 'joint_value', 'p_posterior', 'times_tested', 'price', 'title']
print(cands_new_in_s4[cols].head(20).to_string())
