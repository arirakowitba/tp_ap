"""Diagnose Slot 1 Negative Feedback: Why did EVT-132 say NEW?"""
import json
import pandas as pd
import numpy as np

# Load the 20 items of Slot 1
df_pure = pd.read_csv("results/clean_pure_candidates_16models.csv").head(20)

with open("data/ap_2026_q_2/raw/train_data.jsonlines", "r", encoding="utf-8") as f:
    train_items = [json.loads(line) for line in f]

print("=== EXAMINING TRAINING DATA ANALOGUES FOR SLOT 1 ITEMS ===")
for idx, r in df_pure.iterrows():
    title = r["title"].lower()
    cat = r["category_id"]
    price = r["price"]
    
    # Search similar in train
    matches = []
    for t in train_items:
        t_title = t.get("title", "").lower()
        if t.get("category_id") == cat:
            # check word overlap
            w1 = set(title.split())
            w2 = set(t_title.split())
            overlap = len(w1.intersection(w2))
            if overlap >= 2:
                matches.append((t.get("condition"), t.get("price"), t_title))
    
    n_used = sum(1 for m in matches if m[0] == "used")
    n_new = sum(1 for m in matches if m[0] == "new")
    total = len(matches)
    pct_new = (n_new / total * 100) if total > 0 else 0
    print(f"\nItem #{idx+1}: {r['title'][:45]} | Price: {price} | Cat: {cat}")
    print(f"  Category & Word matches in Train: Total {total} | Used: {n_used} | New: {n_new} ({pct_new:.1f}% NEW)")
    if matches:
        print(f"  Sample train: {matches[0][0]} | ${matches[0][1]} | {matches[0][2][:40]}")
