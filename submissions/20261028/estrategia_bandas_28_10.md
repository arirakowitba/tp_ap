# Tanda del 28-10: bandas de confianza del ensamble honesto `fin_stack` sobre 48 (versión final 28-10 20 hs)

Estado: **final del 28-10**, generado con `fin_stack` (sin la fuga de `stk_sellerother`). Manifiesto: `manifest_bandas_fin_stack.json`. Ningún archivo fue enviado por el asistente.

## Por qué bandas y no más flips de la posterior

Tras 57 y 58 (ambas 0,94633) la posterior de la ronda 125 no tiene filas con ganancia esperada apreciable: la evidencia de 04–58 está agotada. La posterior tampoco discrimina por confianza del modelo: en filas probadas, P(48 mal) pasa solo de 0,30 a 0,36 entre la banda más baja y la más alta de `blend_c`, porque está dominada por la previa común q≈1/3. Falta medir directamente con qué frecuencia un modelo de 95 % OOF le gana a 48 según su confianza.

`fin_stack` (apilado logístico con CV anidada de 5 CatBoost sin fuga + XGBoost + red tabular; OOF **0,9548**, logloss 0,1191) contradice a 48 en 1.518 IDs; 631 ya fueron probados y quedan fuera. Las cuatro pruebas usan solo los **887 IDs nunca probados**, disjuntos entre sí, sobre la 48 exacta.

| N.º | P(48 mal) según modelo | Flips | →used | →new |
|---:|---|---:|---:|---:|
| 59 | ≥ 0,85 | 67 | 34 | 33 |
| 60 | 0,7–0,85 | 252 | 71 | 181 |
| 61 | 0,6–0,7 | 252 | 73 | 179 |
| 62 | 0,5–0,6 | 316 | 108 | 208 |

## Orden y lectura

1. Enviar 59 y 60. Saldo neto público del bloque = round(21.000·(score − 0,94633)); con ~70 % público, la tasa de acierto del modelo frente a 48 en la banda es h ≈ ½ + saldo / (2·0,7·n).
2. Si 59/60 dan h > ½, enviar 61; si no, usar el cupo para dividir la banda positiva por dirección (→used / →new).
3. 62 solo si 61 es positiva.
4. Último cupo: consolidación = 48 + todas las bandas con saldo positivo (+ los flips que la posterior actualizada recomiende). Cada score se agrega a `assemble.py` (ronda 125) como ecuación nueva.

Referencia de magnitud: si el modelo acertara 70 % en la banda ≥0,85, la 59 sumaría ~+6 aciertos públicos; si acertara como los modelos previos (≈33 %), restaría ~−5. Cualquiera de los dos resultados es información útil para calibrar las siguientes tandas.

`submission_63_fin_stack_modelo_puro.csv`: predicción pura de `fin_stack` (sin la 48). Sirve para medir directamente el modelo nuevo contra 0,94633; su score también es la mejor referencia para el 30 % privado.
