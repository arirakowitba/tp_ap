import pandas as pd
from pathlib import Path

ROOT = Path(".")
df = pd.read_csv(ROOT / "results" / "new_to_used_candidates_deep_audit.csv")

tier1 = df[df["unanimous"] & df["has_strong_kw"]].sort_values(by="p_blend", ascending=False)
print(f"Total Tier 1 items: {len(tier1)}")

for idx, r in tier1.iterrows():
    print(f"ID {int(r['item_id']):5d}: p={r['p_blend']:.3f} (lgb={r['p_lgb']:.2f}, cb={r['p_cb']:.2f}) | kw=[{r['kw_hits']}] | ${r['price']:.0f} | {r['category_id']} | {r['title']}")
