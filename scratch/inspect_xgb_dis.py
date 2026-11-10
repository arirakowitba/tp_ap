import sys
sys.path.insert(0, '.')
import json
import pandas as pd
from scratch.xgb_vs_anchor import df_dis

with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    test_lines = [json.loads(line) for line in f]

print("Inspecting Top 20 XGB disagreements with EVT-143:")
for idx, r in df_dis.head(20).iterrows():
    iid = int(r['id'])
    d = test_lines[iid - 1]
    print(f"ID {iid:5d} | Anchor: {r['anchor']:4s} -> XGB: {r['xgb_pred']:4s} (P_new={r['p_new']:.4f}) | Cat: {d.get('category_id'):10s} | Price: {str(d.get('price')):8s} | Title: {d.get('title')}")
