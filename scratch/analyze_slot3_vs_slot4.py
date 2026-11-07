import pandas as pd, json

p_s3 = 'submissions/portfolio_20261107_target_09490/submission_20261107_03_podium_attack_top65.csv'
p_s4 = 'submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv'
p_test = 'data/ap_2026_q_2/raw/test_data.jsonlines'

s3 = pd.read_csv(p_s3).set_index('ID')['condition']
s4 = pd.read_csv(p_s4).set_index('ID')['condition']

with open(p_test, encoding='utf-8') as f:
    test_rows = [json.loads(line) for line in f]

# Items in S3 that were 'used' but in S4 were 'new' (the bad items in S3)
bad_in_s3 = s3[(s3 == 'used') & (s4 == 'new')].index
print(f"Items flipped in S3 that were reverted to 'new' in S4: {len(bad_in_s3)}")
for item_id in bad_in_s3:
    r = test_rows[item_id - 1]
    print(f"ID={item_id:5d} Price=${float(r.get('price', 0) or 0):6.0f} Cat={r.get('category_id')} | {r.get('title')}")
