import pandas as pd
from pathlib import Path

ROOT = Path(".")
df = pd.read_csv(ROOT / "results" / "new_to_used_candidates_deep_audit.csv")

tested = df[df["times_tested"] > 0].sort_values(by="b_gain", ascending=False)
print(f"Total tested 'new -> used' candidates: {len(tested)}")
print(f"Tested with positive Bayesian gain (> +0.20): {(tested['b_gain'] > 0.20).sum()}")

print("\nTop 20 Tested candidates with highest Bayesian gain:")
for idx, r in tested.head(20).iterrows():
    print(f"ID {int(r['item_id']):5d}: gain={r['b_gain']:+.3f} | tested={r['times_tested']} | p={r['p_blend']:.3f} | ${r['price']:.0f} | {r['category_id']} | {r['title']}")
