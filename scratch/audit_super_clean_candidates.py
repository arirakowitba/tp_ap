from pathlib import Path
import pandas as pd, json, re

ROOT = Path('.')
p_bayes = ROOT / 'results' / 'bayesian_solver_103vec' / 'bayesian_transductive_item_effects_103vec.csv'
p_probs = ROOT / 'results' / 'transductive_deep_models_103vec' / 'transductive_test_probabilities_103vec.csv'
p_test = ROOT / 'data' / 'ap_2026_q_2' / 'raw' / 'test_data.jsonlines'

df_bayes = pd.read_csv(p_bayes).set_index('item_id')
df_probs = pd.read_csv(p_probs).set_index('ID')

with open(p_test, encoding='utf-8') as f:
    test_rows = [json.loads(line) for line in f]

test_meta = []
for i, r in enumerate(test_rows, 1):
    test_meta.append({
        'ID': i,
        'title_raw': r.get('title', ''),
        'price_raw': float(r.get('price', 0) or 0),
        'cat_raw': r.get('category_id', '')
    })
df_meta = pd.DataFrame(test_meta).set_index('ID')

df = df_bayes.join(df_probs).join(df_meta)

# Filter:
# 1. Currently 'new' in Slot 4
# 2. is_safe is True
# 3. expected_gain_vs_s4 > 0.15
# 4. p_blend_103vec > 0.70
# 5. times_tested >= 3
cands = df[
    (df['anchor_condition_s4'] == 'new') &
    (df['is_safe']) &
    (df['expected_gain_vs_s4'] > 0.15) &
    (df['p_blend_103vec'] > 0.70) &
    (df['times_tested'] >= 3)
].copy()

# Negative patterns
patterns = [
    r'\b0\s*km\b', r'\bcero\s*km\b', r'\bnuevo\b', r'\bnueva\b', r'\bnuevos\b', r'\bnuevas\b',
    r'\bsellad[oas]*\b', r'\bcerrad[oas]*\b', r'\bblister\b', r'\bestrenar\b',
    r'\ben caja\b', r'\bde fabrica\b', r'\bsin uso\b', r'\bposter\b', r'\blote de\b'
]
def is_bad(t):
    s = str(t).lower()
    for p in patterns:
        if re.search(p, s):
            return True
    return False

cands['is_bad'] = cands['title_raw'].apply(is_bad)
clean = cands[~cands['is_bad']].sort_values(by=['times_tested', 'expected_gain_vs_s4'], ascending=[False, False])

print(f"Total candidates meeting strict criteria (times_tested >= 3): {len(clean)}")
print(f"Candidates with times_tested >= 5: {len(clean[clean['times_tested'] >= 5])}")
print(f"Candidates with times_tested 3-4: {len(clean[clean['times_tested'].between(3, 4)])}")

print("\n" + "=" * 110)
print(f"{'#':2s} | {'ID':5s} | {'Tests':5s} | {'Gain':6s} | {'P_blend':7s} | {'Price':7s} | {'Title'}")
print("=" * 110)
for i, (idx, r) in enumerate(clean.iterrows(), 1):
    print(f"{i:2d} | {idx:5d} | {r['times_tested']:5.0f} | {r['expected_gain_vs_s4']:+6.3f} | {r['p_blend_103vec']:7.3f} | ${r['price_raw']:6.0f} | {r['title_raw'][:60]}")
