import pandas as pd

def read_sub(p):
    df = pd.read_csv(p)
    col_id = [c for c in df.columns if c.lower() in ('id', 'item_id')][0]
    col_target = [c for c in df.columns if c.lower() in ('condition', 'target')][0]
    df = df.rename(columns={col_id: 'ID', col_target: 'condition'})
    df['ID'] = df['ID'].astype(str)
    return df.set_index('ID')

anchor_path = 'submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv'
df_anchor = read_sub(anchor_path)

subs = [
    ('sub1', 'submissions/submission_20261109_01_surgical_rank5_capture_top5.csv', 0.94733, 19892),
    ('sub2', 'submissions/submission_20261109_02_pure_used_capture_top6.csv', 0.94752, 19896),
    ('sub3', 'submissions/submission_20261109_03_pure_used_expansion_top12.csv', 0.94747, 19895),
    ('sub4', 'submissions/submission_20261109_04_podium_attack_top15.csv', 0.94733, 19892),
    ('sub5', 'submissions/submission_20261109_05_podium_climax_top14.csv', 0.94738, 19893),
]

all_diff_ids = set()
sub_diffs = {}

for name, path, score, hits in subs:
    df = read_sub(path)
    diff = df[df['condition'] != df_anchor['condition']]
    delta_hits = hits - 19896
    print(f"=== {name} (score={score}, hits={hits}, delta={delta_hits}) ===")
    print(f"Total flipped vs anchor: {len(diff)}")
    items = []
    for item_id, row in diff.iterrows():
        anc = df_anchor.loc[item_id, 'condition']
        cur = row['condition']
        direction = f"{anc}->{cur}"
        print(f"  ID {item_id}: {direction}")
        items.append((item_id, anc, cur))
        all_diff_ids.add(item_id)
    sub_diffs[name] = items

print(f"\nTotal unique items flipped across all 5 today: {len(all_diff_ids)}")
