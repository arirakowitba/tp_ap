import json
from pathlib import Path
import pandas as pd

ROOT = Path(".")
p_sub2 = ROOT / "submissions" / "today_20261105_portfolio_09480" / "submission_20261105_03_conservative_multimodal_top60.csv"
p_sub3 = ROOT / "submissions" / "submission_20261106_03_surgical_purified_top10.csv"
p_sub4 = ROOT / "submissions" / "submission_20261106_04_pure_unanimous_consumer_top7.csv"
p_sub5 = ROOT / "submissions" / "submission_20261106_05_climax_consumer_unanimous_top15.csv"

df2 = pd.read_csv(p_sub2).set_index("ID")["condition"]
df3 = pd.read_csv(p_sub3).set_index("ID")["condition"]
df4 = pd.read_csv(p_sub4).set_index("ID")["condition"]
df5 = pd.read_csv(p_sub5).set_index("ID")["condition"]

with open(ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines", encoding="utf-8") as f:
    test_rows = [json.loads(line) for line in f]

def inspect_flips(name, df_cand, df_ref):
    diff = df_cand[df_cand != df_ref]
    print(f"\n=== {name} vs REF (Flips: {len(diff)}) ===")
    for item_id, new_val in diff.items():
        idx = item_id - 1
        it = test_rows[idx]
        old_val = df_ref[item_id]
        print(f"ID {item_id:5d}: {old_val} -> {new_val} | cat={it.get('category_id')} | ${it.get('price')} | {it.get('title')[:60]}")

inspect_flips("Sub 3 (0.94709)", df3, df2)
inspect_flips("Sub 4 (0.94714)", df4, df2)
inspect_flips("Sub 5 (0.94719)", df5, df2)
