"""Analyze diff between 0.94719 and 0.94738."""
import pandas as pd

s_orig = pd.read_csv("submissions/submission_20261106_05_climax_consumer_unanimous_top15.csv")
s1 = pd.read_csv("submissions/portfolio_20261107_target_09490/submission_20261107_01_purified_probe_top25.csv")

diff = s_orig[s_orig["condition"] != s1["condition"]]
print(f"Diff count between 0.94719 and 0.94738: {len(diff)}")
print(diff.head(25))
