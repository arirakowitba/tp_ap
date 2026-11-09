import pandas as pd
from pathlib import Path

p_fusion = Path('results/cross_referenced_bayesian_fusion/joint_information_fusion_all30k.csv')
if p_fusion.exists():
    df_fusion = pd.read_csv(p_fusion)
    print(f"Fusion rows: {len(df_fusion)}")
    print("Columns:", df_fusion.columns.tolist())
    
    # Check top positive joint value items
    cands = df_fusion[
        (df_fusion["anchor_condition"] == "new") &
        (df_fusion["joint_value"] > 0) &
        (df_fusion["is_safe"])
    ].sort_values(by="joint_value", ascending=False)
    print(f"Total safe cands: {len(cands)}")
    print(cands[['item_id', 'joint_value', 'p_posterior', 'times_tested', 'title']].head(20).to_string())
else:
    print("Fusion file not found!")
