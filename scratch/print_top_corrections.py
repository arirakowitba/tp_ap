import json

with open("results/item_label_confidences_summary.json", encoding="utf-8") as f:
    summary = json.load(f)

print(f"{'Rank':<4} | {'ID':<6} | {'Correction':<14} | {'P(new)':<8} | {'Price':<9} | {'Title':<42} | {'Justification'}")
print("-" * 125)

for c in summary["top_50_highest_confidence_corrections"][:30]:
    direction = f"{c['anchor_label']} -> {c['recommended_label']}"
    print(f"{c['rank']:<4d} | {c['item_id']:<6d} | {direction:<14s} | {c['p_new_probe']:<8.4f} | ${c['price']:<8.1f} | {c['title'][:40]:<42s} | {c['justification'][:50]}")
