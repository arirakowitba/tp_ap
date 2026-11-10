import json
import re

print("Loading train data...")
titles = []
conditions = []

with open("data/ap_2026_q_2/raw/train_data.jsonlines", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        titles.append(item.get("title", "").lower())
        conditions.append(item.get("condition"))

total_train = len(titles)
print(f"Loaded {total_train} train items.")

cues_to_test = [
    # USED cues
    ("para repuesto", r"\bpara repuesto[s]?\b"),
    ("para despiece", r"\bpara despiece\b"),
    ("para reparar", r"\b(?:para|a)\s+reparar\b"),
    ("no funciona", r"\bno\s+funciona\b"),
    ("no enciende", r"\bno\s+(?:enciende|prende)\b"),
    ("buen estado", r"\bbuen\s+estado\b"),
    ("muy buen estado", r"\bmuy\s+buen\s+estado\b"),
    ("excelente estado", r"\bexcelente\s+estado\b"),
    ("optimo estado", r"\b(?:optimo|óptimo)\s+estado\b"),
    ("impecable estado", r"\bimpecable\s+estado\b"),
    ("como nuevo", r"\bcomo\s+nuev[oa]s?\b"),
    ("detalles de uso", r"\b(?:detalles|marcas|signos)\s+de\s+uso\b"),
    ("poco uso", r"\b(?:poco|escaso|minimo|mínimo)\s+uso\b"),
    ("segunda mano", r"\bsegunda\s+mano\b"),
    ("usado", r"\busad[oa]s?\b"),
    ("antiguo", r"\bantigu[oa]s?\b"),
    ("vintage", r"\b(?:vintage|vitage|retro)\b"),
    ("de coleccion", r"\bde\s+colecci[oó]n\b"),
    ("le falta", r"\ble\s+falta[n]?\b"),
    ("con detalles", r"\bcon\s+detalles\b"),
    ("desgaste", r"\bdesgaste\b"),
    ("permuto", r"\bpermuto\b"),

    # NEW cues
    ("caja sellada", r"\b(?:en\s+)?caja\s+sellada\b"),
    ("cerrado de fabrica", r"\bcerrad[oa]\s+de\s+f[aá]brica\b"),
    ("sellado de fabrica", r"\bsellad[oa]\s+de\s+f[aá]brica\b"),
    ("caja cerrada", r"\b(?:en\s+)?caja\s+cerrada\b"),
    ("blister cerrado", r"\b(?:en\s+)?blister\s+cerrado\b"),
    ("a estrenar", r"\ba\s+estrenar\b"),
    ("sin uso", r"\bsin\s+uso\b"),
    ("nuevo sin uso", r"\bnuev[oa]\s+sin\s+uso\b"),
    ("0 km", r"\b0\s*km\b|\bcero\s+kil[oó]metro\b"),
    ("0 km sin rodar", r"\b0\s*km\s+sin\s+rodar\b"),
    ("garantia oficial", r"\bgarant[ií]a\s+oficial\b"),
    ("con garantia escrita", r"\bcon\s+garant[ií]a\s+escrita\b"),
    ("factura a o b", r"\bfactura\s+[ab]\b|\bhacemos\s+factura\b"),
    ("nuevo original", r"\bnuev[oa]\s+original\b|\boriginal\s+nuev[oa]\b"),
    ("distribuidor oficial", r"\bdistribuidor\s+oficial\b"),
    ("somos fabricantes", r"\bsomos\s+fabricantes\b"),
    ("por mayor", r"\b(?:x|por)\s+mayor\b"),
    ("a pedido", r"\ba\s+pedido\b"),
]

print(f"\n{'Cue Name':<25} | {'Train Count':<11} | {'Used Count':<10} | {'New Count':<10} | {'Pct Used':<9} | {'Pct New':<9}")
print("-" * 85)

for name, pat in cues_to_test:
    rx = re.compile(pat)
    used_cnt = 0
    new_cnt = 0
    for t, c in zip(titles, conditions):
        if rx.search(t):
            if c == 'used':
                used_cnt += 1
            else:
                new_cnt += 1
    total = used_cnt + new_cnt
    if total > 0:
        pct_used = (used_cnt / total) * 100
        pct_new = (new_cnt / total) * 100
        print(f"{name:<25} | {total:<11d} | {used_cnt:<10d} | {new_cnt:<10d} | {pct_used:7.2f}% | {pct_new:7.2f}%")
