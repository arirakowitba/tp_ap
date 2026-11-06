import pandas as pd
from pathlib import Path

ROOT = Path(".")
p = ROOT / "results" / "bayesian_solver_98vec" / "bayesian_transductive_item_effects_98vec.csv"
df = pd.read_csv(p)

print(f"Total rows in bayesian effects: {len(df)}")
print(f"Safe rows: {df['is_safe'].sum()}")

safe_pos = df[(df["expected_gain_vs_anchor"] > 0.15) & (df["is_safe"])].copy()
print(f"Safe pos (>0.15): {len(safe_pos)}")

safe_pos["dir"] = safe_pos["anchor_condition"] + " -> " + safe_pos["flipped_condition"]
print("\nDirection breakdown in safe pos:")
print(safe_pos["dir"].value_counts())

print("\nCategory breakdown in safe pos:")
print(safe_pos["category_id"].value_counts().head(10))

print("\nTop 40 safe opportunities:")
for _, r in safe_pos.head(40).iterrows():
    print(f"ID {int(r['item_id']):5d}: {r['dir']} | gain={r['expected_gain_vs_anchor']:+.3f} | p_used={r['model_p_used']:.3f} | tested={r['times_tested']} | ${r['price']:.0f} | {r['category_id']} | {r['title']}")
