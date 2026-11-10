import json
import re

titles = []
conditions = []

with open("data/ap_2026_q_2/raw/train_data.jsonlines", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        titles.append(item.get("title", "").lower())
        conditions.append(item.get("condition"))

more_cues = [
    # USED candidates
    ("para despiece", r"\bpara despiece\b"),
    ("despiece", r"\bdespiece\b"),
    ("para repuestos", r"\bpara repuestos\b"),
    ("para partes", r"\bpara partes\b"),
    ("no anda", r"\bno anda\b"),
    ("a reparar", r"\ba reparar\b"),
    ("roto / rota", r"\brot[oa]s?\b"),
    ("impecable", r"\bimpecable\b"),
    ("impecables", r"\bimpecables\b"),
    ("estado impecable", r"\bestado impecable\b"),
    ("perfecto estado", r"\bperfecto estado\b"),
    ("muy lindo estado", r"\bmuy lindo estado\b"),
    ("buen estado general", r"\bbuen estado general\b"),
    ("uso particular", r"\buso particular\b"),
    ("solo entendidos", r"\bsolo entendidos\b"),
    ("funciona bien", r"\bfunciona bien\b"),
    ("funciona perfecto", r"\bfunciona perfecto\b"),
    ("completo en caja", r"\bcompleto en caja\b"),
    ("marcas de uso", r"\bmarcas de uso\b"),
    ("detalles esteticos", r"\bdetalles est[eé]ticos\b"),

    # NEW candidates
    ("nuevo en caja", r"\bnuev[oa] en caja\b"),
    ("nuevo en blister", r"\bnuev[oa] en blister\b"),
    ("nuevo cerrado", r"\bnuev[oa] cerrad[oa]\b"),
    ("nuevo sellado", r"\bnuev[oa] sellad[oa]\b"),
    ("sellado en caja", r"\bsellad[oa] en caja\b"),
    ("caja sellada original", r"\bcaja sellada original\b"),
    ("a estrenar en caja", r"\ba estrenar en caja\b"),
    ("0 km de fabrica", r"\b0\s*km de f[aá]brica\b"),
    ("sin rodar", r"\bsin rodar\b"),
    ("0 km sin rodar", r"\b0\s*km sin rodar\b"),
    ("factura a y b", r"\bfactura a y b\b"),
    ("local a la calle", r"\blocal a la calle\b"),
    ("locales a la calle", r"\blocales a la calle\b"),
    ("venta por mayor", r"\bventa por mayor\b"),
    ("directo de fabrica", r"\bdirecto de f[aá]brica\b"),
    ("fabricantes", r"\bfabricantes\b"),
    ("envio gratis", r"\benv[ií]o gratis\b"),
]

print(f"\n{'Cue Name':<25} | {'Train Count':<11} | {'Used Count':<10} | {'New Count':<10} | {'Pct Used':<9} | {'Pct New':<9}")
print("-" * 85)

for name, pat in more_cues:
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
