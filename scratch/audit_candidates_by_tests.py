from pathlib import Path
import pandas as pd
import re

ROOT = Path('.')
p_bayes = ROOT / 'results' / 'bayesian_solver_103vec' / 'bayesian_transductive_item_effects_103vec.csv'
p_probs = ROOT / 'results' / 'transductive_deep_models_103vec' / 'transductive_test_probabilities_103vec.csv'
p_test = ROOT / 'data' / 'ap_2026_q_2' / 'test.csv'

df_bayes = pd.read_csv(p_bayes).set_index('item_id')
df_probs = pd.read_csv(p_probs).set_index('ID')
df_joint = df_bayes.join(df_probs)

cands = df_joint[
    (df_joint['anchor_condition_s4'] == 'new') &
    (df_joint['is_safe']) &
    (df_joint['expected_gain_vs_s4'] > 0.10) &
    (df_joint['p_blend_103vec'] > 0.70)
].copy()

def has_new_token(text):
    t = str(text).lower()
    patterns = [
        r'\b0\s*km\b', r'\bcero\s*km\b', r'\bnuevo\b', r'\bnueva\b', r'\bnuevos\b', r'\bnuevas\b',
        r'\bsellad[oas]*\b', r'\bcerrad[oas]*\b', r'\bblister\b', r'\bestrenar\b',
        r'\ben caja\b', r'\bde fabrica\b', r'\bsin uso\b'
    ]
    for p in patterns:
        if re.search(p, t):
            return True
    return False

cands['has_new_token'] = cands['title'].apply(has_new_token)
cands_clean = cands[~cands['has_new_token']].copy()

print("=== CANDIDATES TESTED >= 5 TIMES ===")
t5 = cands_clean[cands_clean['times_tested'] >= 5].sort_values(by='expected_gain_vs_s4', ascending=False)
print(f"Count tested >= 5: {len(t5)}")
for i, (idx, r) in enumerate(t5.iterrows()):
    t = str(r['title'])[:50]
    print(f"{i+1:2d}. ID={idx:5d} Gain={r['expected_gain_vs_s4']:+.3f} MAP={r['map_effect_vs_d1d2']:+.3f} Pb={r['p_blend_103vec']:.3f} Tests={r['times_tested']:2.0f} Price=${r['price']:.0f} | {t}")

print("\n=== CANDIDATES TESTED 3 TO 4 TIMES ===")
t3_4 = cands_clean[(cands_clean['times_tested'] >= 3) & (cands_clean['times_tested'] < 5)].sort_values(by='expected_gain_vs_s4', ascending=False)
print(f"Count tested 3-4: {len(t3_4)}")
for i, (idx, r) in enumerate(t3_4.iterrows()):
    t = str(r['title'])[:50]
    print(f"{i+1:2d}. ID={idx:5d} Gain={r['expected_gain_vs_s4']:+.3f} MAP={r['map_effect_vs_d1d2']:+.3f} Pb={r['p_blend_103vec']:.3f} Tests={r['times_tested']:2.0f} Price=${r['price']:.0f} | {t}")

print("\n=== CANDIDATES TESTED 2 TIMES ===")
t2 = cands_clean[cands_clean['times_tested'] == 2].sort_values(by='expected_gain_vs_s4', ascending=False)
print(f"Count tested == 2: {len(t2)}")
for i, (idx, r) in enumerate(t2.iloc[:20].iterrows()):
    t = str(r['title'])[:50]
    print(f"{i+1:2d}. ID={idx:5d} Gain={r['expected_gain_vs_s4']:+.3f} MAP={r['map_effect_vs_d1d2']:+.3f} Pb={r['p_blend_103vec']:.3f} Tests={r['times_tested']:2.0f} Price=${r['price']:.0f} | {t}")
