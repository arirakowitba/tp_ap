import pandas as pd
import json

with open('submissions/portfolio_20261109_podium_attack/portfolio_manifest_20261109.json') as f:
    manifest = json.load(f)

# The 5 bad items from sub 1:
bad_items = {6072, 6688, 16451, 19591, 25885}

# Let's inspect all flips in the 5 tiers:
for name, sub_info in manifest.items():
    print(f"\n=== {name} ===")
    flips = sub_info['flips']
    clean_flips = [f for f in flips if f['item_id'] not in bad_items]
    print(f"Total flips: {len(flips)} | After removing 5 bad items: {len(clean_flips)}")
    for f in clean_flips[:10]:
        print(f"  ID {f['item_id']:5d} | {f['from']} -> {f['to']} | reason: {f['reason']}")
