# Ronda de entrenamiento train-only: texto OOF + CatBoost profundo (1 de noviembre de 2026)

Datos: ZIP original `ap-2026-q-2 (1).zip`, SHA256 `3e0b032195766e681a411a1d1ea06fe906940dd025a7317adfc5302791bbe886`; 70.000 train y 30.000 test; `used=1`. Ninguna etiqueta test, score público ni otro dataset etiquetado usado en entrenamiento.

Código: `scripts/train_text_condition_20261101.py` y `scripts/train_deep_text_20261101.py`. CatBoost profundidad 6, LR .05, hasta 1250 rondas; early stopping y umbral elegidos sólo en tune. Baseline con variables profundas históricas, descartando agregados calculados globalmente y reconstruyendo frecuencias y precio relativo exclusivamente desde fit. Candidato añade probabilidad de texto TF-IDF de caracteres 3–5 + regresión logística: vectorizador y modelo ajustados en cada fold GroupKFold por vendedor para OOF de fit; modelo de texto completo de fit para tune/confirm. Probabilidades de tune/confirm no participan del ajuste CatBoost. Feature set nuevo y arquitectura elegidos durante desarrollo, por lo que estos confirm no son externos al proceso de selección.

| Corte | n fit/tune/confirm | Acc base confirm | Acc texto confirm | Neto pareado | Vistos / nuevos | Logloss base → texto | IC95% bootstrap vendedor, neto |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261104 | 38512/16046/15442 | .91983 | .92948 | +149 | +9 / +140 | .20300 → .18457 | [+90,+217] |
| 20261105 | 37287/15959/16754 | .92438 | .93267 | +139 | +1 / +138 | .19725 → .17585 | [+82,+203] |

El test tiene 71.623% de vendedores vistos según el repo, contra 34.0% y 31.1% en estos confirm. Una reponderación ilustrativa a 30k, **suponiendo invariancia del efecto por estrato**, da +154 [80,229] y +106 [29,186] aciertos frente a **este baseline CatBoost**, no frente a D1+D2. Los confirm se solapan en 3541 IDs y 2957 vendedores; no sumar resultados ni tratarlos como independientes. La mejora se concentra fuertemente en vendedores nuevos. La comparación aún carece de HistGB+FastText/B10/H-AVG en exactamente estos IDs, y D1+D2 no tiene un clon train validado. Por tanto no se promueve CSV test y no se proyecta score público.

Los pares y reportes JSON, con hash de split y predicción, se guardaron en Library: `libfile_da26305bcb1c8191b46ff741eb67dcdc` (ZIP `paired_deep_confirm_20261101.zip`). SHA256 de pares: 20261104 `328619a17023f5b8d714ed57a6704b1ff3f6ffa2ddd48e8c9e16cebc084d2779`; 20261105 `df6da6695d84802f65dafb262d98ef8924eeee0f8df5dec47630f4401942cce3`.

La primera corrida exploratoria con menos variables dio +210 en 15449 confirm sobre su base débil; la segunda con variables intermedias +162 en 16861, pero no respaldan promoción. La versión con 101 features y TF-IDF inductivo es la evidencia actual. Próximo gate: comparar estos mismos cortes contra baselines fuertes y repetir regla congelada en un split seller-group más cercano a la mezcla test, auditar solapamiento con los 18 holdouts históricos, luego refit 70k sólo si sigue ganando. No Kaggle por agentes.
