import pandas as pd
from pathlib import Path

ROOT = Path(".")
df = pd.read_csv(ROOT / "results" / "new_to_used_candidates_deep_audit.csv")

tier2 = df[df["unanimous"] & (~df["has_strong_kw"]) & (df["p_blend"] > 0.80)].sort_values(by="p_blend", ascending=False)
print(f"Total Tier 2 items with p_blend > 0.80: {len(tier2)}")

for idx, r in tier2.head(35).iterrows():
    print(f"ID {int(r['item_id']):5d}: p={r['p_blend']:.3f} (lgb={r['p_lgb']:.2f}, cb={r['p_cb']:.2f}) | ${r['price']:.0f} | {r['category_id']} | {r['title']}")
