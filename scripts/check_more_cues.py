import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(".")

with open(ROOT / "data/ap_2026_q_2/raw/test_data.jsonlines", encoding="utf-8") as f:
    test_raw = [json.loads(line) for line in f]

evt = pd.read_csv(ROOT / "submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv")
y_evt = evt["condition"].values
s1 = pd.read_csv(ROOT / "submissions/SUBMISSION_1_TODAY_RECOMMENDED.csv")
diff_s1 = set(i for i in range(30000) if s1.loc[i, "condition"] != y_evt[i])

# Check items with "sellado"
print("=== All items with 'sellado' in EVT=used ===")
for i, r in enumerate(test_raw):
    if i in diff_s1:
        continue
    t = (r.get("title") or "").lower()
    p = float(r.get("price") or 0.0)
    if "sellado" in t and y_evt[i] == "used":
        print(f"ID {i+1}: ${p:.1f} | {r.get('title')}")

# Check items with "a reparar"
print("\n=== All items with 'a reparar' in EVT=new ===")
for i, r in enumerate(test_raw):
    if i in diff_s1:
        continue
    t = (r.get("title") or "").lower()
    p = float(r.get("price") or 0.0)
    if "a reparar" in t and y_evt[i] == "new":
        print(f"ID {i+1}: ${p:.1f} | {r.get('title')}")
