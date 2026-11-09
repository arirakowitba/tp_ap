import json

with open("submissions/portfolio_20261109_podium_ladder/portfolio_manifest.json") as f:
    m = json.load(f)

for s in m:
    print(f"Slot {s['slot']}: {s['filename']:43s} | Flips: {s['flips_vs_anchor']:3d} | Exp: {s['expected_score']:.5f} | SHA: {s['sha256_lf']}")
