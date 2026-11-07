from pathlib import Path
import pandas as pd

ROOT = Path('.')
p_bayes = ROOT / 'results' / 'bayesian_solver_103vec' / 'bayesian_transductive_item_effects_103vec.csv'
p_probs = ROOT / 'results' / 'transductive_deep_models_103vec' / 'transductive_test_probabilities_103vec.csv'

df_bayes = pd.read_csv(p_bayes).set_index('item_id')
df_probs = pd.read_csv(p_probs).set_index('ID')
df_joint = df_bayes.join(df_probs)

gold = df_joint[
    (df_joint['anchor_condition_s4'] == 'new') &
    (df_joint['is_safe']) &
    (df_joint['expected_gain_vs_s4'] > 0.20) &
    (df_joint['p_blend_103vec'] > 0.75)
].sort_values(by='expected_gain_vs_s4', ascending=False)

print(f"Total gold consensus candidates: {len(gold)}")
print("-" * 100)
for i, (idx, r) in enumerate(gold.iloc[:60].iterrows()):
    title = str(r['title'])[:50]
    print(f"{i+1:2d}. ID={idx:5d} Gain={r['expected_gain_vs_s4']:+.3f} MAP={r['map_effect_vs_d1d2']:+.3f} Pb={r['p_blend_103vec']:.3f} Tests={r['times_tested']:2.0f} Price=${r['price']:.0f} | {title}")
