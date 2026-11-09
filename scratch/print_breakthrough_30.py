import json
import pandas as pd

s7_2 = pd.read_csv('submissions/portfolio_20261107_target_09490/submission_20261107_02_calibrated_expansion_top45.csv').set_index('ID')['condition']
s7_4 = pd.read_csv('submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv').set_index('ID')['condition']

with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    test_dict = {i+1: json.loads(line) for i, line in enumerate(f)}

diff_ids = [815, 1067, 1346, 1823, 3031, 4997, 6308, 6900, 7936, 8227, 8650, 8661, 8674, 8997, 10013, 14353, 16190, 16515, 17645, 22237, 22499, 22905, 22953, 24735, 27016, 28301, 28431, 28453, 29533, 29859]

records = []
for idx in diff_ids:
    it = test_dict[idx]
    records.append({
        'ID': idx,
        's7_2': s7_2.loc[idx],
        's7_4': s7_4.loc[idx],
        'title': it.get('title'),
        'price': it.get('price')
    })

df = pd.DataFrame(records)
print(df.to_string())
