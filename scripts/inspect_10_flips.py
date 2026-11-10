"""Inspect 10 false used items in EVT-132."""
import json
import pandas as pd

with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    items = [json.loads(line) for line in f]

flips = [435, 3777, 6627, 10745, 14438, 15441, 18487, 19857, 28004, 29306]

df_evt = pd.read_csv("submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv")

for fid in flips:
    item = items[fid - 1]
    title = item.get("title", "")
    price = item.get("price", 0)
    cat = item.get("category_id", "")
    cond_evt = df_evt.loc[fid - 1, "condition"]
    print(f"ID {fid:5d} | EVT: {cond_evt:4s} | ${price:9.1f} | {cat:8s} | {title}")
