import json
import pandas as pd

# Load test data lines (1-indexed)
with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    test_lines = [json.loads(line) for line in f]
test_data = {i + 1: d for i, d in enumerate(test_lines)}

octeto_ids = [1477, 3777, 4256, 11421, 11779, 13224, 19857, 29386]

# Also load confidences from probes
df_conf = pd.read_csv('results/item_label_confidences_from_probes.csv')

print(f"{'ID':5s} | {'Cat':10s} | {'Price':8s} | {'Flip':13s} | {'Probe P(new)':12s} | {'Title'}")
print("-" * 80)
for iid in octeto_ids:
    d = test_data[iid]
    c_row = df_conf[df_conf['item_id'] == iid]
    p_new = c_row['p_new_probe'].values[0] if len(c_row) > 0 else -1
    flip = "used -> new" if iid in [1477, 3777, 13224, 19857] else "new -> used"
    print(f"{iid:5d} | {d.get('category_id',''):10s} | {str(d.get('price',''))[:8]:8s} | {flip:13s} | {p_new:12.4f} | {d.get('title','')}")
