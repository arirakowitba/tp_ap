"""Analyze exact differences across all submissions in portfolio_20261107."""
import pandas as pd
from pathlib import Path

dir_20261107 = Path("submissions/portfolio_20261107_target_09490")

files = [
    ("s1", "submission_20261107_01_purified_probe_top25.csv", 0.94738, 19893),
    ("s2", "submission_20261107_02_calibrated_expansion_top45.csv", 0.94738, 19893),
    ("s3", "submission_20261107_03_podium_attack_top65.csv", 0.94709, 19887),
    ("s4", "submission_20261107_04_podium_breakthrough_top85.csv", 0.94752, 19896),
    ("s5", "submission_20261107_05_podium_climax_top105.csv", 0.94738, 19893),
]

dfs = {tag: pd.read_csv(dir_20261107 / fn).set_index("ID")["condition"] for tag, fn, sc, h in files}

print("=== DIFFERENCES MATRIX ===")
for i in range(len(files)):
    for j in range(i+1, len(files)):
        t1, fn1, sc1, h1 = files[i]
        t2, fn2, sc2, h2 = files[j]
        diff_mask = (dfs[t1] != dfs[t2])
        diff_ids = diff_mask[diff_mask].index.tolist()
        delta_h = h2 - h1
        print(f"{t1} ({sc1:.5f}) vs {t2} ({sc2:.5f}) | Flips: {len(diff_ids):3d} | Delta Hits: {delta_h:+2d} | Diff IDs: {diff_ids[:10]}")

# Specifically check s3 vs s4 and s4 vs s5
print("\n=== FLIPS BETWEEN S3 (0.94709) AND S4 (0.94752) [Delta = +9 hits] ===")
diff_s3_s4 = dfs["s3"][dfs["s3"] != dfs["s4"]]
for fid, c_s4 in diff_s3_s4.items():
    c_s3 = dfs["s3"][fid]
    print(f"ID {fid:5d}: s3={c_s3:4s} -> s4={c_s4:4s}")

print("\n=== FLIPS BETWEEN S4 (0.94752) AND S5 (0.94738) [Delta = -3 hits] ===")
diff_s4_s5 = dfs["s4"][dfs["s4"] != dfs["s5"]]
for fid, c_s5 in diff_s4_s5.items():
    c_s4 = dfs["s4"][fid]
    print(f"ID {fid:5d}: s4={c_s4:4s} -> s5={c_s5:4s}")
