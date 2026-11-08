import json
from pathlib import Path
import pandas as pd

p_test = Path('data/ap_2026_q_2/raw/test_data.jsonlines')
p_s4_yesterday = Path('submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv')
p_probs = Path('results/transductive_deep_models_103vec/transductive_test_probabilities_103vec.csv')

s4 = pd.read_csv(p_s4_yesterday).set_index('ID')['condition']
probs = pd.read_csv(p_probs).set_index('ID')

target_ids = [345, 11421, 9822, 11994, 26109, 268, 16358, 23224, 10881, 3780, 18399, 10003, 23119]

with open(p_test, encoding='utf-8') as f:
    test_rows = [json.loads(line) for line in f]

print("=== INSPECTING THE 13 ULTRA-PURE CANDIDATES FOR SLOT 2 ===")
for i in target_ids:
    r = test_rows[i-1]
    p_blend = probs.loc[i, 'p_blend_103vec'] if i in probs.index else -1
    print(f"ID={i:5d} | S4={s4[i]} | P_blend={p_blend:.3f} | Price=${float(r.get('price', 0) or 0):5.0f} | LT={r.get('listing_type_id')} | IQ={r.get('initial_quantity')} | {r.get('title')}")
