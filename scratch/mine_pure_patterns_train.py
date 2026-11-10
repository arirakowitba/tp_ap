"""Comprehensive empirical pattern audit on the 70,000 train dataset.

Tests dozens of high-precision lexical and structural cues:
- Positive cues for 'new'
- Negative cues for 'used'
Measures exact count, precision, and false positive examples.
"""

import json, re
from collections import defaultdict

with open('data/ap_2026_q_2/raw/train_data.jsonlines', encoding='utf-8') as f:
    train_rows = [json.loads(line) for line in f]

print(f"Loaded {len(train_rows)} train items.\n")

# Candidate regex dictionary
CANDIDATES = {
    # Cues for USED
    "USED_para_repuesto": (r"\bpara\s+repuestos?\b|\bpara\s+desarme\b|\bdesguace\b", "used"),
    "USED_no_funciona": (r"\bno\s+funciona\b|\bcon\s+fallas?\b|\bfunciona\s+mal\b", "used"),
    "USED_roto_averiado": (r"\b(averiad[oa]s?|rot[oa]s?)\b(?!\s*vaso)", "used"),
    "USED_usado_word": (r"\b(usad[oa]s?|segunda\s+mano)\b(?!\s*no\b)", "used"),
    "USED_un_solo_uso": (r"\b(un|1)\s+solo\s+uso\b|\bpoco\s+uso\b|\bpoqu[ií]simo\s+uso\b", "used"),
    "USED_como_nuevo": (r"\bcomo\s+nuev[oa]s?\b|\bigual\s+a\s+nuev[oa]\b", "used"),
    "USED_impecable_vitrina": (r"\bde\s+vitrina\b|\bimpecable\s+estado\b", "used"),
    "USED_marcas_de_uso": (r"\bmarcas?\s+de\s+uso\b|\brayones?\b|\bdesgaste\b|\bdetalles?\s+de\s+uso\b", "used"),
    "USED_reacondicionado": (r"\b(reacondicionad[oa]s?|reacond)\b", "used"),
    "USED_antiguedad_estricta": (r"\b(antiguedad|antiguas?|antiguos?|de\s+epoca)\b", "used"),
    "USED_juguete_vintage": (r"\b(vintage|retro)\b.*\b(juguete|muñec[oa]|autito|figura)\b|\b(juguete|muñec[oa]|autito|figura)\b.*\b(vintage|retro)\b", "used"),
    "USED_para_reparar": (r"\bpara\s+reparar\b|\ba\s+reparar\b", "used"),

    # Cues for NEW
    "NEW_caja_sellada": (r"\bcaja\s+sellada\b|\bcerrad[oa]\s+de\s+f[aá]brica\b|\bprecintad[oa]s?\b|\ben\s+blister\b", "new"),
    "NEW_garantia_oficial": (r"\bgarant[ií]a\s+oficial\b|\bcon\s+garant[ií]a\s+de\s+f[aá]brica\b", "new"),
    "NEW_factura_ab": (r"\bfactura\s+[ab]\b|\bhacemos\s+factura\s+[ab]\b|\bemitimos\s+factura\b", "new"),
    "NEW_digital_ebook": (r"\b(libro\s+digital|ebook|e-book|formato\s+pdf|en\s+pdf)\b", "new"),
    "NEW_descarga_digital": (r"\bdescarga\s+digital\b|\benv[ií]o\s+digital\b|\bentrega\s+inmediata\s+digital\b", "new"),
    "NEW_cero_km": (r"\b0\s*km\b|\bcero\s*km\b", "new"),
    "NEW_distribuidor_oficial": (r"\bdistribuidor\s+oficial\b|\blocal\s+a\s+la\s+calle\b", "new"),
    "NEW_cuotas_sin_interes": (r"\bcuotas\s+sin\s+inter[eé]s\b", "new"),
}

results = []
for name, (pat, target) in CANDIDATES.items():
    rx = re.compile(pat, re.I)
    matches = [r for r in train_rows if rx.search(r['title'])]
    tot = len(matches)
    if tot == 0:
        continue
    correct = sum(1 for r in matches if r['condition'] == target)
    precision = correct / tot
    results.append({
        'name': name,
        'target': target,
        'pattern': pat,
        'total_train_matches': tot,
        'correct': correct,
        'errors': tot - correct,
        'precision': precision
    })

results.sort(key=lambda x: (x['precision'], x['total_train_matches']), reverse=True)

print(f"{'Pattern Name':<28} | Target | Matches | Errors | Precision")
print("-" * 65)
for r in results:
    print(f"{r['name']:<28} | {r['target'].upper():<6} | {r['total_train_matches']:7d} | {r['errors']:6d} | {r['precision']*100:6.2f}%")
