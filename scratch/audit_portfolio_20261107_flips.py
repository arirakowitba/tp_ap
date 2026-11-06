import json
from pathlib import Path
import pandas as pd

ROOT = Path(".")
p_anc = ROOT / "submissions" / "submission_20261106_05_climax_consumer_unanimous_top15.csv"
p_s5 = ROOT / "submissions" / "portfolio_20261107_target_09490" / "submission_20261107_05_podium_climax_top80.csv"
p_s1 = ROOT / "submissions" / "portfolio_20261107_target_09490" / "submission_20261107_01_calibrated_probe_top35.csv"

df_anc = pd.read_csv(p_anc).set_index("ID")["condition"]
df_s5 = pd.read_csv(p_s5).set_index("ID")["condition"]
df_s1 = pd.read_csv(p_s1).set_index("ID")["condition"]

diff5 = df_s5[df_s5 != df_anc]
diff1 = df_s1[df_s1 != df_anc]
print(f"Total flips in Candidate 01 (Slot 1): {len(diff1)}")
print(f"Total flips in Candidate 05 (Slot 5): {len(diff5)}")

with open(ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines", encoding="utf-8") as f:
    test_rows = [json.loads(line) for line in f]

flips = []
for item_id, new_val in diff5.items():
    idx = item_id - 1
    it = test_rows[idx]
    old_val = df_anc[item_id]
    flips.append({
        "id": item_id,
        "old": old_val,
        "new": new_val,
        "dir": f"{old_val} -> {new_val}",
        "cat": it.get("category_id", ""),
        "price": it.get("price", 0),
        "title": it.get("title", "")[:60]
    })
df_f = pd.DataFrame(flips)
print("\nDirection breakdown in Candidate 05:")
print(df_f["dir"].value_counts())

print("\nCategory breakdown in Candidate 05:")
print(df_f["cat"].value_counts().head(10))

print("\nAre there any real estate items (MLA1468)?", (df_f["cat"] == "MLA1468").sum())
print("Are there any vehicle items (MLA1743, MLA1744)?", df_f["cat"].isin(["MLA1743", "MLA1744"]).sum())

print("\n--- First 35 flips (Slot 1) ---")
for _, r in df_f.head(35).iterrows():
    print(f"ID {r['id']:5d}: {r['dir']} | {r['cat']} | ${r['price']:.0f} | {r['title']}")

print("\n--- Remaining 45 flips (Slots 2-5) ---")
for _, r in df_f.iloc[35:].iterrows():
    print(f"ID {r['id']:5d}: {r['dir']} | {r['cat']} | ${r['price']:.0f} | {r['title']}")
