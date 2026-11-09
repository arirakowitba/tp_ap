"""Audit every item in Slot 1 in detail."""

import pandas as pd

df_evt = pd.read_csv("submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv")
df_s1 = pd.read_csv("submissions/portfolio_20261109_podium_ladder/submission_slot1_top20_rank6_probe.csv")
df_test = pd.read_json("data/ap_2026_q_2/raw/test_data.jsonlines", lines=True)
fusion = pd.read_csv("results/cross_referenced_bayesian_fusion/joint_information_fusion_all30k.csv")

diff_idx = df_s1[df_s1["condition"] != df_evt["condition"]]["ID"].values - 1

print(f"Total flips in Slot 1: {len(diff_idx)}")
print("=" * 110)
print(f"{'#':2s} | {'Item ID':15s} | {'Row':5s} | {'Price':8s} | {'Cat':10s} | {'JV':6s} | {'P_mod':5s} | {'Title'}")
print("=" * 110)

for idx, i in enumerate(diff_idx, 1):
    row_test = df_test.iloc[i]
    row_fus = fusion.iloc[i]
    print(f"{idx:2d} | {row_test['id']:15s} | {i+1:5d} | ${row_test['price']:7.1f} | {row_test['category_id']:10s} | {row_fus['joint_value']:.4f} | {row_fus['p_model_mean']:.3f} | {row_test['title']}")
print("=" * 110)
