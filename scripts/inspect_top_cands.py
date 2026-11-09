import json

test_path = 'data/ap_2026_q_2/raw/test_data.jsonlines'
ids_to_check = {28010, 12726, 22770, 10356, 2204, 6822, 12738, 19025, 21937, 8010, 25763, 26665, 25371, 4060, 3417}

with open(test_path, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f, 1):
        if idx in ids_to_check:
            row = json.loads(line)
            title = row.get("title", "")
            warranty = row.get("warranty", "")
            price = row.get("price", "")
            tags = row.get("tags", [])
            print(f"ID {idx}: Title=\"{title}\"")
            print(f"   Warranty: \"{warranty}\" | Price: {price} | Tags: {tags}")
            attrs = [f"{a.get('name')}: {a.get('value_name')}" for a in row.get('attributes', []) if isinstance(a, dict)]
            if attrs:
                print(f"   Attributes ({len(attrs)}): {attrs[:4]}")
            print("-" * 60)
