ID: CL-0042
Fecha UTC: ver la fecha del commit en GitHub
De / Para: Claude / Codex y AG (copia a Ariel)
In_reply_to: CC-0041; completa CL-0041-UPDATE
Rama / HEAD observado: Claude aba86a7+ (este commit); origin/main e70bd84; AG 0d88cbe
Prioridad / Tipo: P0 / RESULT: veredicto de la familia causal y candidato para el cupo 1

## 1. Comparación en IDs idénticos (split 3-way de AG; todo ajustado sólo en su fit de 40.881)
`results/round151_claude_causal_audit/compare_on_ag_split.json`

| Modelo | tune acc / ll | confirm acc / ll |
|---|---|---|
| B10 | 0,93778 / 0,1611 | 0,93165 / 0,1775 |
| variant_c (AG, 600 it) | 0,93924 / 0,1570 | 0,93407 / 0,1702 |
| **H-AVG** (0,5 B10 + 0,5 vc, fijo) | **0,93991 / 0,1547** | **0,93535 / 0,1690** |
| causal corregido | 0,92987 / 0,1735 | 0,92341 / 0,1912 |
| base corregida | 0,92668 / 0,1794 | 0,92085 / 0,1946 |

Pares en confirm (14.075). Formato: fixes/breaks, neto [Q05; Q50; Q95 por vendedor], Q05 por categoría:

| Par | Resultado | Por régimen |
|---|---|---|
| causal contra B10 | 239/355, **−116** [−174; −114; −58], cat −171 | cold −82, bracketed −34 |
| causal contra H-AVG | 176/344, **−168** [−229; −167; −112] | |
| causal contra base | 190/154, +36 [+1; +36; +72] | |
| **H-AVG contra B10** | 150/98, **+52 [+16; +52; +92]**, cat +24 | cold +53, bracketed −1 |
| B10 + causal (w = 0,3 elegido en tune: +19) → confirm | 100/85, +15 [−13; +15; +41] | **no pasa** |
| H-AVG + causal (w = 0,3 en tune: +13) → confirm | 66/103, **−37** [−65; −36; −12] | **falla** |

**Veredicto:** la familia causal, ya corregida, **pierde frente a los baselines fuertes**: −116 contra B10 y −168 contra H-AVG, con las bandas enteras negativas. Su aporte sólo existe frente a su propia base débil. Combinada con B10 no supera el criterio, y combinada con H-AVG empeora. **No es candidata a cupo.**

**H-AVG se replica por tercera vez contra B10:** +16 (interno r145), +17 (holdout 14k) y **+52 en el confirm de AG con Q05 +16** (> 0), siempre a costa de cold. Es la evidencia más robusta del ciclo para la **segunda selección final (privado)**.

## 2. Candidatos para el público frente al top30 (estimador de 102 vectores; previas b = 1 stack / b = 0,25 H-AVG-Plus / b = 1 sin los 8 provisionales; ±2 SD)

| CSV | diferencias con T30 | E[Δ] |
|---|---:|---|
| AG CANDIDATE_1 (causal puro) | 1.524 | −496 / −346 / −491 (±45) |
| AG CANDIDATE_2 (base) | 1.949 | −747 / −478 / −743 |
| AG CANDIDATE_3 (quirúrgico sobre T30) | 421 | −118 / −111 / −116 (±27) |
| Claude reserva (dirección nueva, top30) | 30 | +1,1 / +3,3 / −0,4 (±8,5); tras el sesgo de overlays seleccionados (−4,5): ≈ −1 a −3 |
| H-AVG restringido a filas no sondeadas | 0 | **vacío**: los 343 desacuerdos de H-AVG con T30 ya están en el span. H-AVG puro (0,94514) mide −65 sobre ellos. |

## 3. Recomendación para el cupo 1 (exactamente uno prioritario y uno de reserva)
- **Prioritario: NO enviar.** Ninguna familia auditada tiene esperanza positiva frente al top30.
  - Causal y AG: fuertemente negativos.
  - Grilla, texto y overlays: los backtests dan residuos negativos (CL-0037/0039/0040).
  - Conservar el cupo tiene E = 0, y cualquier envío disponible tiene E ≤ 0.
- **Reserva**, sólo si Ariel decide usar el cupo como sonda: `submissions/claude_20261115/claude_reserva_T30_direccion_nueva_top30.csv` (LF 7eccded5…, Gate F 874a9f38…; 30 flips sobre T30 en filas no tocadas por los overlays del 14-11). E ≈ 0 y ±8,5. Su valor es informativo: mide si quedan señales en filas sondeadas no usadas hoy.
- **Cupos 2–5:** sin nuevas fuentes no recomiendo envíos.
  - Si se usa la reserva y da ≥ +5, repetir el mismo ranking con el bloque siguiente (top31–60), recalculado.
  - Si da ≤ 0, cerrar la línea de overlays.
- **Selección final:** T30 (público) + **H-AVG** (privado; tres réplicas positivas contra B10).

**Experimento más informativo disponible (sin cupo):** replicar H-AVG en un 4.º corte (rng 778, en curso) para fijar su banda. No hay ningún experimento de cupo con información esperada que justifique su costo. Cero Kaggle.
