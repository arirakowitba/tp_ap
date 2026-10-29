# Ronda 140 — por qué falló `fin_stack` en las cinco pruebas del 28-10

Fecha: 28-10-2026, noche. Autor: Claude (sesión con Hugo). Insumos: las cinco pruebas que envió Ariel (registradas en [round137](round137_adaptive_candidates_20261028.md) y [round138](round138_fin_stack_segment_samples_20261028.md)) y las predicciones guardadas en la ronda 127. No se usaron etiquetas de test ni datos externos etiquetados.

## Reproducción de los bloques

`src/round140_diagnostico/reconstruir_bloques_28_10.py` rearma desde la 48 exacta los IDs de cada prueba y verifica los SHA256 publicados por Ariel: 59 (67), 59 vendedores nuevos (26), 59 vistos→new (27), muestra 150 vistos y muestra 150 nuevos. Los cinco coinciden.

| Prueba | Flips | Score | Saldo aprox. vs 48 |
|---|---:|---:|---:|
| 59 completa | 67 | 0,94576 | −12 |
| 59 vendedores nuevos | 26 | 0,94619 | −3 |
| 59 vistos→new | 27 | 0,94590 | −9 |
| muestra 150 nuevos | 150 | 0,94495 | −29 |
| muestra 150 vistos | 150 | 0,94257 | −79 |

Leaderboard informado por Ariel: Ariel 9.º con 0,94633 (48); 8.º 0,94638, 4.º 0,94747, 1.º 0,95404.

## Hallazgo 1: los pesos negativos del apilado invierten la señal

`fin_stack` = logística sobre logits con pesos [CatBoost ×5: +1,67; XGBoost: −0,57; red: −0,16]. En la muestra de 150 vistos:

| Modelo | P(used) media | Predice new |
|---|---:|---:|
| `fin_stack` | 0,38 | 88 % |
| `fin_cat5` (5 CatBoost sin fuga) | 0,58 | 29 % |
| `b_xgb_noso` | 0,76 | 17 % |
| `c_nn_avg4` | 0,77 | 15 % |
| `v1`, `cat_v1` (ronda 126) | 0,75–0,78 | 13–15 % |

Donde XGBoost y la red coinciden con 48 en "used" con mucha confianza, los pesos negativos empujan el apilado a "new". En OOF ese patrón sumaba (+0,2 puntos), pero en test se invierte: es sobreajuste del nivel 2, compatible con la dependencia entre folds que señaló Ariel (TE global y OOF de base reutilizados por el meta-modelo). **Los pesos negativos en el apilado deben prohibirse.** `fin_stack` y las pruebas 60–63 quedan descartadas.

## Hallazgo 2: aun sin el apilado, los desacuerdos con 48 se pierden

`fin_cat5` sola habría hecho 66 de los 67 flips de la 59, que perdió ≈−12 (acierto del modelo ≈37 % en público). En la muestra de 150 vistos habría hecho solo 56 de los 150 cambios. En CV aleatoria, cuando CatBoost contradice la mayoría del propio vendedor acierta 88 % (1.385 filas); en el público, los flips equivalentes (vistos→new contra vendedores casi 100 % usados en train, mediana de 1 aviso en train) ganan ≈12 %. Los títulos de esos casos no son duplicados del vendedor (similitud mediana 0,35), así que no es un efecto de reavisos.

Lectura: la CV aleatoria con TE/stacks globales sobreestima justamente la capacidad de contradecir el historial del vendedor. Coincide con el patrón histórico (q≈1/3: en desacuerdos de modelos con 48, 48 acierta dos de cada tres).

## Recomendación para el 29-10

1. Conservar 48 como submission seleccionada; no enviar 60–63.
2. Antes de otra prueba de flips, validar con el protocolo de [round139](round139_leaderboard_training_plan_20261028.md): TE, stacks y estadísticas de vendedor recalculadas **dentro** de cada fold externo seller-group, tune/confirm separados, bootstrap por vendedor, y comparación pareada contra una proxy de 48.
3. Si se vuelve a apilar, restringir pesos ≥ 0 (promedio convexo) y elegir pesos en tune, no en el OOF evaluado.
4. Hipótesis a medir en esa validación: penalizar que el modelo contradiga el historial del vendedor visto cuando la evidencia es solo texto.
