import pandas as pd, json
from pathlib import Path

p_s4 = 'submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv'
p_probs = 'results/transductive_deep_models_103vec/transductive_test_probabilities_103vec.csv'
p_test = 'data/ap_2026_q_2/raw/test_data.jsonlines'

s4 = pd.read_csv(p_s4).set_index('ID')['condition']
df = pd.read_csv(p_probs)

with open(p_test, encoding='utf-8') as f:
    test_rows = [json.loads(line) for line in f]

s4_new_p_high = df[(df['ID'].map(s4) == 'new') & (df['p_blend_103vec'] > 0.90)].sort_values(by='p_blend_103vec', ascending=False)
s4_used_p_low = df[(df['ID'].map(s4) == 'used') & (df['p_blend_103vec'] < 0.10)].sort_values(by='p_blend_103vec', ascending=True)

print("=== S4 PREDICTS NEW, BUT P_BLEND > 0.90 (TOP 25) ===")
for idx, r in s4_new_p_high.iloc[:25].iterrows():
    i = int(r['ID'])
    it = test_rows[i-1]
    p = float(it.get('price', 0) or 0)
    print(f"ID={i:5d} | Pb={r['p_blend_103vec']:.3f} | ${p:5.0f} | LT={it.get('listing_type_id')} | IQ={it.get('initial_quantity')} | {it.get('title')[:55]}")

print("\n=== S4 PREDICTS USED, BUT P_BLEND < 0.10 (TOP 25) ===")
for idx, r in s4_used_p_low.iloc[:25].iterrows():
    i = int(r['ID'])
    it = test_rows[i-1]
    p = float(it.get('price', 0) or 0)
    print(f"ID={i:5d} | Pb={r['p_blend_103vec']:.3f} | ${p:5.0f} | LT={it.get('listing_type_id')} | IQ={it.get('initial_quantity')} | {it.get('title')[:55]}")
