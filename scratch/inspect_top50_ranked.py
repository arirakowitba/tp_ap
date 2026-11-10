import json
import pandas as pd

df = pd.read_csv('results/ranked_candidate_flips.csv')

print(f"Total flips in ranked list: {len(df)}")
top50 = df.head(50)

print(f"{'Rank':4s} | {'ID':5s} | {'Flip':13s} | {'P_meta':6s} | {'ExpG':6s} | {'Cat':10s} | {'Price':8s} | {'Title'}")
print("-" * 105)
for i, r in top50.iterrows():
    print(f"{i+1:4d} | {r['id']:5d} | {r['anchor']} -> {r['rec']:4s} | {r['p_meta']:.3f} | {r['expected_gain']:+5.2f} | {r['category_id']:10s} | {str(r['price'])[:8]:8s} | {r['title']}")
