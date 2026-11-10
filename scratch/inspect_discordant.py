import json

target_ids = {'MLA3743291532', 'MLA1769820953', 'MLA8696135233', 'MLA4002658026', 'MLA4988814155'}
with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        if r.get('id') in target_ids:
            title = r.get('title')
            warr = r.get('warranty')
            price = r.get('price')
            attrs = [a.get('value_name') or a.get('name') for a in (r.get('attributes') or [])]
            print(f"ID: {r.get('id')} | Title: '{title}' | Price: {price} | Warr: '{warr}' | Attrs: {attrs[:4]}")
