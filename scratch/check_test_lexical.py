import json, re
from collections import Counter

# Load test data
with open("data/ap_2026_q_2/raw/test_data.jsonlines", "r", encoding="utf-8") as f:
    test_titles = [json.loads(line).get("title", "") for line in f]

# Define 98%+ precision cues
DECISIVE_USED_CUES = [
    (r"\bpara repuesto[s]?\b", "para repuesto", 1.0),
    (r"\bpara despiece\b", "para despiece", 1.0),
    (r"\b(?:para|a)\s+reparar\b", "a reparar", 1.0),
    (r"\bno\s+funciona\b", "no funciona", 1.0),
    (r"\bno\s+(?:enciende|prende|anda)\b", "no enciende", 1.0),
    (r"\bbuen\s+estado\s+general\b", "buen estado general", 1.0),
    (r"\bbuen\s+estado\b", "buen estado", 1.0),
    (r"\bmuy\s+buen\s+estado\b", "muy buen estado", 1.0),
    (r"\bexcelente\s+estado\b", "excelente estado", 0.9925),
    (r"\bimpecable\s+estado\b|\bestado\s+impecable\b", "impecable estado", 1.0),
    (r"\b(?:detalles|marcas|signos)\s+de\s+uso\b", "detalles de uso", 1.0),
    (r"\bpoco\s+uso\b", "poco uso", 1.0),
    (r"\busad[oa]s?\b", "usado", 0.9833),
]

DECISIVE_NEW_CUES = [
    (r"\b(?:en\s+)?caja\s+sellada\b", "caja sellada", 1.0),
    (r"\b(?:en\s+)?caja\s+cerrada\b", "caja cerrada", 1.0),
    (r"\bsellad[oa]\s+de\s+f[aá]brica\b", "sellado de fabrica", 1.0),
    (r"\bcerrad[oa]\s+de\s+f[aá]brica\b", "cerrado de fabrica", 1.0),
    (r"\bnuev[oa]\s+sellad[oa]\b|\bsellad[oa]\s+en\s+caja\b", "nuevo sellado", 1.0),
    (r"\bnuev[oa]\s+cerrad[oa]\b", "nuevo cerrado", 0.981),
    (r"\bnuev[oa]\s+en\s+blister\b", "nuevo en blister", 1.0),
    (r"\b0\s*km\s+sin\s+rodar\b", "0 km sin rodar", 1.0),
    (r"\bdistribuidor\s+oficial\b", "distribuidor oficial", 1.0),
    (r"\bsomos\s+fabricantes\b|\bdirecto\s+de\s+f[aá]brica\b|\bfabricantes\b", "fabricantes", 1.0),
    (r"\b(?:x|por|venta\s+por)\s+mayor\b", "por mayor", 1.0),
]

test_used_matches = []
test_new_matches = []

for idx, t in enumerate(test_titles):
    t_lower = t.lower()
    for pat, name, prec in DECISIVE_USED_CUES:
        if re.search(pat, t_lower):
            test_used_matches.append((idx + 1, name, prec, t))
            break
    for pat, name, prec in DECISIVE_NEW_CUES:
        if re.search(pat, t_lower):
            test_new_matches.append((idx + 1, name, prec, t))
            break

print(f"Test items matching decisive USED cues (>=98%): {len(test_used_matches)}")
print(f"Test items matching decisive NEW cues (>=98%): {len(test_new_matches)}")

# Check overlap
used_ids = set(m[0] for m in test_used_matches)
new_ids = set(m[0] for m in test_new_matches)
overlap = used_ids.intersection(new_ids)
print(f"Overlap between used and new cues: {len(overlap)}")
if overlap:
    print(f"Example overlap IDs: {list(overlap)[:5]}")
