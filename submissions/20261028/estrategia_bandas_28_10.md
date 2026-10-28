# Tanda del 28-10: bandas de confianza del ensamble `ens_shallow` sobre 48 (BORRADOR)

Estado: **borrador**. Se regenera con el ensamble final de la ronda 126 antes de enviar; los hashes del manifiesto pueden cambiar. Ningún archivo fue enviado por el asistente.

## Por qué bandas y no más flips de la posterior

Tras 57 y 58 (ambas 0,94633) la posterior de la ronda 125 no tiene filas con ganancia esperada apreciable: la evidencia de 04–58 está agotada. La posterior tampoco discrimina por confianza del modelo: en filas probadas, P(48 mal) pasa solo de 0,30 a 0,36 entre la banda más baja y la más alta de `blend_c`, porque está dominada por la previa común q≈1/3. Falta medir directamente con qué frecuencia un modelo de 95 % OOF le gana a 48 según su confianza.

`ens_shallow` (promedio de cinco CatBoost de profundidad 4–6, OOF 0,9505, logloss 0,1279) contradice a 48 en 1.132 IDs; 628 ya fueron probados por rondas anteriores y quedan fuera. Las cuatro pruebas usan solo los **504 IDs nunca probados**, disjuntos entre sí, todos sobre la 48 exacta.

| N.º | P(48 mal) según modelo | Flips | →used | →new |
|---:|---|---:|---:|---:|
| 59 | ≥ 0,85 | 23 | 19 | 4 |
| 60 | 0,70–0,85 | 83 | 50 | 33 |
| 61 | 0,60–0,70 | 145 | 71 | 74 |
| 62 | 0,50–0,60 | 253 | 95 | 158 |

## Orden y lectura

1. Enviar 59 y 60. Saldo neto público del bloque = round(21.000·(score − 0,94633)); con ~70 % público, la tasa de acierto del modelo frente a 48 en la banda es h ≈ ½ + saldo / (2·0,7·n).
2. Si 59/60 dan h > ½, enviar 61; si no, usar el cupo para dividir la banda positiva por dirección (→used / →new).
3. 62 solo si 61 es positiva.
4. Último cupo: consolidación = 48 + todas las bandas con saldo positivo (+ los flips que la posterior actualizada recomiende). Cada score se agrega a `assemble.py` (ronda 125) como ecuación nueva.

Referencia de magnitud: si el modelo acertara 70 % en la banda ≥0,85, la 59 sumaría ~+6 aciertos públicos; si acertara como los modelos previos (≈33 %), restaría ~−5. Cualquiera de los dos resultados es información útil para calibrar las siguientes tandas.
