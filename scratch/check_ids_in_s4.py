import pandas as pd

anchor = pd.read_csv('submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv').set_index('ID')['condition']
df_probs = pd.read_csv('results/transductive_deep_models/transductive_test_probabilities.csv').set_index('ID')

# Check top items from df_c
test_ids = [10753, 27016, 859, 5585, 12590, 4483, 5606, 9604, 18146, 6308, 16515, 23119, 28010, 15176, 25449, 1015, 24735, 11994, 7936, 29278, 20259, 17645, 28819]

for idx in test_ids:
    print(f"ID {idx:5d} in anchor s4: {anchor.loc[idx]}")
