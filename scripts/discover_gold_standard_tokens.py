import json
import re
from collections import defaultdict
import pandas as pd

# 1. Load train data
print("Loading train data...")
with open('data/ap_2026_q_2/raw/train_data.jsonlines', encoding='utf-8') as f:
    train_items = [json.loads(line) for line in f]

# 2. Tokenize into n-grams (1 to 3 words)
token_counts_used = defaultdict(int)
token_counts_new = defaultdict(int)

def get_ngrams(text):
    # clean text
    words = re.findall(r'[a-záéíóúñ0-9]+', text.lower())
    res = set()
    for w in words:
        if len(w) >= 3:
            res.add(w)
    for i in range(len(words)-1):
        res.add(words[i] + ' ' + words[i+1])
    for i in range(len(words)-2):
        res.add(words[i] + ' ' + words[i+1] + ' ' + words[i+2])
    return res

print("Extracting n-grams from 70,000 train items...")
for it in train_items:
    cond = it['condition']
    grams = get_ngrams(it['title'])
    if cond == 'used':
        for g in grams:
            token_counts_used[g] += 1
    else:
        for g in grams:
            token_counts_new[g] += 1

all_tokens = set(token_counts_used.keys()).union(set(token_counts_new.keys()))

pure_used_tokens = []
pure_new_tokens = []

for tok in all_tokens:
    u = token_counts_used[tok]
    n = token_counts_new[tok]
    tot = u + n
    if tot >= 20: # strong support
        prec_u = u / tot
        prec_n = n / tot
        if prec_u >= 0.975:
            pure_used_tokens.append((tok, tot, u, n, prec_u))
        elif prec_n >= 0.975:
            pure_new_tokens.append((tok, tot, u, n, prec_n))

pure_used_tokens.sort(key=lambda x: (x[4], x[1]), reverse=True)
pure_new_tokens.sort(key=lambda x: (x[4], x[1]), reverse=True)

print(f"Discovered {len(pure_used_tokens)} high-precision USED cues (>=97.5% precision, support >= 20)")
for tok, tot, u, n, prec in pure_used_tokens[:20]:
    print(f"  USED: '{tok:25s}' | Support: {tot:4d} | Prec: {prec*100:5.1f}% (Used: {u}, New: {n})")

print(f"\nDiscovered {len(pure_new_tokens)} high-precision NEW cues (>=97.5% precision, support >= 20)")
for tok, tot, u, n, prec in pure_new_tokens[:20]:
    print(f"  NEW : '{tok:25s}' | Support: {tot:4d} | Prec: {prec*100:5.1f}% (New: {n}, Used: {u})")

# 3. Check EVT-143 predictions vs these golden tokens
anchor_df = pd.read_csv('submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv')
anchor_p = anchor_df.set_index('ID')['condition'].to_dict()

with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    items = [json.loads(line) for line in f]
test_items = {idx+1: it for idx, it in enumerate(items)}

# Check test items
mismatches_used = [] # Token says USED (>=97.5%), but EVT-143 says NEW
mismatches_new = []  # Token says NEW (>=97.5%), but EVT-143 says USED

used_cues_dict = {tok: (tot, prec) for tok, tot, u, n, prec in pure_used_tokens}
new_cues_dict = {tok: (tot, prec) for tok, tot, u, n, prec in pure_new_tokens}

for cid, it in test_items.items():
    cur = anchor_p[cid]
    grams = get_ngrams(it['title'])
    
    # check if any pure used token matches
    matched_u = [g for g in grams if g in used_cues_dict]
    matched_n = [g for g in grams if g in new_cues_dict]
    
    if matched_u and cur == 'new' and not matched_n:
        best_u = max(matched_u, key=lambda g: used_cues_dict[g][0]) # highest support
        mismatches_used.append((cid, best_u, used_cues_dict[best_u], it.get('price', 0), it.get('title', '')))
    
    if matched_n and cur == 'used' and not matched_u:
        best_n = max(matched_n, key=lambda g: new_cues_dict[g][0])
        mismatches_new.append((cid, best_n, new_cues_dict[best_n], it.get('price', 0), it.get('title', '')))

print(f"\n=======================================================")
print(f"TEST ITEMS WHERE EVT-143 SAYS 'NEW' BUT HAS >=97.5% PURE USED TOKEN: {len(mismatches_used)}")
for cid, tok, (supp, prec), price, title in mismatches_used[:20]:
    print(f"ID {cid:5d} | Cue: '{tok}' (supp={supp}, prec={prec*100:.1f}%) | ${price:8.1f} | {title}")

print(f"\n=======================================================")
print(f"TEST ITEMS WHERE EVT-143 SAYS 'USED' BUT HAS >=97.5% PURE NEW TOKEN: {len(mismatches_new)}")
for cid, tok, (supp, prec), price, title in mismatches_new[:20]:
    print(f"ID {cid:5d} | Cue: '{tok}' (supp={supp}, prec={prec*100:.1f}%) | ${price:8.1f} | {title}")
