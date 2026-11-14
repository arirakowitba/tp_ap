import pandas as pd
import numpy as np

s4 = pd.read_csv('submissions/SUBMISSION_4_TODAY_RECOMMENDED.csv')
t30 = pd.read_csv('submissions/claude_20261114/claude_blendfix_posterior_top30_sobre_S4.csv')
t30_u58 = pd.read_csv('submissions/claude_20261114/claude_T30_mas_U58_blendfix_noprobado.csv')
b10 = pd.read_csv('results/round141_claude_baseline/full70k/submission_b10_full.csv')

id_col = 'ID' if 'ID' in s4.columns else 'id'
pred_col_s4 = 'condition' if 'condition' in s4.columns else s4.columns[1]
pred_col_t30 = 'condition' if 'condition' in t30.columns else t30.columns[1]
pred_col_u58 = 'condition' if 'condition' in t30_u58.columns else t30_u58.columns[1]
pred_col_b10 = 'condition' if 'condition' in b10.columns else b10.columns[1]

# Sort by ID
s4 = s4.sort_values(id_col).reset_index(drop=True)
t30 = t30.sort_values(id_col).reset_index(drop=True)
t30_u58 = t30_u58.sort_values(id_col).reset_index(drop=True)
b10 = b10.sort_values(id_col).reset_index(drop=True)

df = pd.DataFrame({
    'id': s4[id_col],
    's4': s4[pred_col_s4],
    't30': t30[pred_col_t30],
    't30_u58': t30_u58[pred_col_u58],
    'b10': b10[pred_col_b10]
})

f_t30 = df['t30'] != df['s4']
f_u58 = df['t30_u58'] != df['t30']

u58_rows = df[f_u58].copy()

match_b10 = (u58_rows['s4'] == u58_rows['b10'])
diff_b10 = ~match_b10
print(f"On U58 (58 rows) vs Claude's submission_b10_full.csv:")
print(f"  s4 == b10 (concordant): {match_b10.sum()}")
print(f"  s4 != b10 (discordant): {diff_b10.sum()}")

rev_to_b10 = (u58_rows.loc[diff_b10, 't30_u58'] == u58_rows.loc[diff_b10, 'b10'])
print(f"In the {diff_b10.sum()} discordant rows, does t30_u58 revert to b10? {rev_to_b10.sum()} / {diff_b10.sum()}")

flip_from_b10 = (u58_rows.loc[match_b10, 't30_u58'] != u58_rows.loc[match_b10, 'b10'])
print(f"In the {match_b10.sum()} concordant rows, does t30_u58 flip away from b10? {flip_from_b10.sum()} / {match_b10.sum()}")
