import json
from collections import Counter

print("Loading train data...")
with open('data/ap_2026_q_2/raw/train_data.jsonlines', encoding='utf-8') as f:
    train_data = [json.loads(line) for line in f]

def check_in_train(label, filter_fn):
    matches = [d for d in train_data if filter_fn(d)]
    conds = Counter(d['condition'] for d in matches)
    total = len(matches)
    pct_new = (conds['new'] / total * 100) if total > 0 else 0
    print(f"[{label}] Count: {total} | new: {conds['new']} ({pct_new:.1f}%), used: {conds['used']} ({100-pct_new:.1f}%)")

print("\n--- CHECKING CANDIDATES IN TRAIN ---")
check_in_train("Moneda / Numismatica (MLA44260)", lambda d: d['category_id'] == 'MLA44260')
check_in_train("Moneda de 1900-1950", lambda d: 'moneda' in d['title'].lower() and any(yr in d['title'] for yr in ['1905', '1910', '1920', '1930', '1940']))
check_in_train("0 km sin rodar", lambda d: '0 km sin rodar' in d['title'].lower() or '0km sin rodar' in d['title'].lower())
check_in_train("excelente estado", lambda d: 'excelente estado' in d['title'].lower())
check_in_train("buen estado", lambda d: 'buen estado' in d['title'].lower())
check_in_train("estampillas", lambda d: 'estampilla' in d['title'].lower())
check_in_train("Alquiler departamentos (MLA1473)", lambda d: d['category_id'] == 'MLA1473')
check_in_train("Alquiler Soler / Palermo", lambda d: 'alquiler' in d['title'].lower() and 'departamento' in d['title'].lower())
check_in_train("Chatitas en cuero ecologico", lambda d: 'chatita' in d['title'].lower() or 'chatitas' in d['title'].lower())
check_in_train("La ciberpolitica / Nuevos ciudadanos (MLA6009 Libros)", lambda d: d['category_id'] == 'MLA6009' and 'libro' in d.get('domain_id', '').lower())
