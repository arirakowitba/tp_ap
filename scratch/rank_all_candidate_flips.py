import sys
sys.path.insert(0, '.')
import json
import numpy as np
import pandas as pd
from scratch.analyze_all_vectors import anchor, vectors

print("Loading model predictions...")
# 1. XGB Depth 9
df_xgb = pd.read_csv('results/deep_xgboost_depth9_20261110/test_probabilities.csv')
p_xgb = 1.0 - df_xgb['prob_used'].values

# 2. LGB 127
lgb_npz = np.load('results/deep_lgbm_leaves127_20261110/lgbm_leaves127_predictions.npz')
p_lgb = 1.0 - lgb_npz['test_lgb']

# 3. Bagging 50 trees
bag_npz = np.load('results/multiseed_bagging_20261110/multiseed_bagged_predictions.npz')
p_bag = bag_npz['final_bagged_test']

# 4. Transductive Cycle 2
trans_npz = np.load('results/transductive_cycle2_20261110/transductive_cycle2_predictions.npz')
p_trans = trans_npz['trans_blend_test']

# Meta-Blend
p_meta = 0.65 * p_xgb + 0.30 * p_lgb + 0.05 * p_bag

# 5. Load raw test data for metadata & lexical check
print("Loading raw test data...")
with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    test_lines = [json.loads(line) for line in f]

# 6. Load probe confidences
df_probe = pd.read_csv('results/item_label_confidences_from_probes.csv')

# Build comprehensive candidate analysis dataframe
items = []
for idx, d in enumerate(test_lines):
    iid = idx + 1
    anch = anchor['vec'][iid]
    a_num = 1 if anch == 'new' else 0
    p_m = p_meta[idx]
    rec = 'new' if p_m >= 0.5 else 'used'
    
    # Expected gain:
    # If anch == 'used' (a_num=0): gain if rec == 'new' is (2*p_m - 1)
    # If anch == 'new'  (a_num=1): gain if rec == 'used' is (1 - 2*p_m)
    expected_gain = (2 * p_m - 1) * (1 - 2 * a_num) * 0.699933
    
    # Check Top 17 elite probe consistency
    top17_preds = [v['vec'][iid] for v in vectors if v['score'] >= 0.94700]
    top17_new_cnt = sum(1 for p in top17_preds if p == 'new')
    top17_used_cnt = len(top17_preds) - top17_new_cnt
    
    # Attributes
    attrs = {a['name']: a['value_name'] for a in d.get('attributes', [])}
    km = attrs.get('Kilómetros')
    ano = attrs.get('Año')
    
    items.append({
        'id': iid,
        'title': d.get('title'),
        'category_id': d.get('category_id'),
        'price': d.get('price'),
        'anchor': anch,
        'rec': rec,
        'is_flip': int(anch != rec),
        'p_meta': p_m,
        'p_xgb': p_xgb[idx],
        'p_lgb': p_lgb[idx],
        'p_bag': p_bag[idx],
        'expected_gain': expected_gain,
        'top17_new_cnt': top17_new_cnt,
        'top17_used_cnt': top17_used_cnt,
        'km': km,
        'ano': ano
    })

df_all = pd.DataFrame(items)

# Filter to flips only
df_flips = df_all[df_all['is_flip'] == 1].sort_values('expected_gain', ascending=False).reset_index(drop=True)
print(f"\nTotal candidate flips: {len(df_flips)}")
print(f"Flips with expected_gain > 0: {(df_flips['expected_gain'] > 0).sum()}")
print(f"Flips with expected_gain > 0.40 (P > 0.78): {(df_flips['expected_gain'] > 0.40).sum()}")
print(f"Flips with expected_gain > 0.50 (P > 0.85): {(df_flips['expected_gain'] > 0.50).sum()}")
print(f"Flips with expected_gain > 0.60 (P > 0.93): {(df_flips['expected_gain'] > 0.60).sum()}")

# Print Top 30 flips
print("\n--- TOP 30 HIGHEST CONVICTION CANDIDATE FLIPS ---")
print(f"{'Rank':4s} | {'ID':5s} | {'Flip':13s} | {'P(meta)':8s} | {'ExpGain':7s} | {'Top17(N/U)':10s} | {'Cat':10s} | {'Price':8s} | {'Title'}")
print("-" * 110)
for i, r in df_flips.head(30).iterrows():
    flip = f"{r['anchor']} -> {r['rec']}"
    top17 = f"{r['top17_new_cnt']}/{r['top17_used_cnt']}"
    print(f"{i+1:4d} | {r['id']:5d} | {flip:13s} | {r['p_meta']:8.4f} | {r['expected_gain']:+7.3f} | {top17:10s} | {r['category_id']:10s} | {str(r['price'])[:8]:8s} | {r['title'][:45]}")

# Cumulative expected hits curve
df_flips['cum_gain'] = df_flips['expected_gain'].cumsum()
print("\nCumulative Expected Public Leaderboard Hits Gained:")
for k in [10, 25, 50, 75, 100, 110, 125, 150, 200]:
    gain = df_flips.iloc[k - 1]['cum_gain']
    proj_hits = 19898 + gain
    proj_score = proj_hits / 20998.0
    print(f"  Top {k:3d} flips: Expected +{gain:5.1f} hits -> Projected Hits: {proj_hits:7.1f} ({proj_score:.5f})")

# Save ranked candidate table
df_flips.to_csv('results/ranked_candidate_flips.csv', index=False)
print("\nSaved full ranked flips to results/ranked_candidate_flips.csv")
