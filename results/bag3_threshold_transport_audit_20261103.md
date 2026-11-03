# Auditoría de transporte de umbral y frescura de CatBoost bag3 — 2026-11-03

## Alcance

Se reanalizaron los pares por ID de los cortes 20261124–20261127, sin reentrenar ni elegir reglas sobre confirm. Se separó el efecto probabilístico del bagging del efecto de elegir umbrales distintos para single y bag3 en el tune de cada corte. También se reconstruyeron los conjuntos confirm de 2026, 314159, 20261143 y 20261144 con `split_testlike`.

## Resultados

| Seed confirm | Neto bag3 con umbrales propios de tune | Neto con umbral común .50 | Delta logloss bag3-single |
|---|---:|---:|---:|
| 20261124 | +28 | +16 | -0.000997 |
| 20261125 | +14 | +7 | -0.001721 |
| 20261126 | +6 | +3 | -0.001620 |
| 20261127 | +4 | 0 | -0.000484 |

En 50.349 evaluaciones y 36.001 IDs únicos, el neto deduplicado con los umbrales propios queda entre **+34 y +48** según prioridad de solapamiento. Con umbral común .50 queda entre **+14 y +16**. El logloss mejora en los cuatro cortes, señal de probabilidades más estables, pero la mayor parte de la aparente mejora de accuracy depende del ajuste de umbral.

### Backtest prequential de umbrales

Se aplicó el par de umbrales elegido en el tune del corte anterior al confirm siguiente:

- 20261124 → 20261125: +7.
- 20261125 → 20261126: -5.
- 20261126 → 20261127: +1.
- Total: **+3** sobre 37.806 evaluaciones.

La mejora de accuracy no se transporta de forma material. Bag3 sigue siendo útil para calibración/logloss y diversidad, pero no hay evidencia train-only de una ganancia estable de decenas mediante el umbral.

### Solapamiento de confirms

El confirm 20261144 no es disjunto de los usados antes:

- 2.628 IDs en común con 2026 (21,4% de confirm44).
- 2.658 con 314159 (21,7%).
- 2.600 con 20261143 (21,2%).

Tras todos los cortes ya registrados en el repo, sólo cerca del 2% de un confirm nuevo típico corresponde a IDs nunca evaluados antes. Todo nuevo confirm convencional debe declararse **desarrollo reutilizado**, no validación externa independiente. Esto no invalida el resultado interno de 20261144, pero sí impide usarlo como confirmación totalmente fresca de la arquitectura.

## Decisión

1. Mantener bag3 como base probabilística preferida por logloss, sin atribuirle una mejora de accuracy estable basada en umbrales por corte.
2. En 20261144, reportar además: neto bag3-single a umbral común .50 y con los umbrales transportados desde 20261127 (.510 single, .505 bag3).
3. Evaluar la futura feature OOF prefijada dentro de CatBoost con regla congelada y tratar confirm como desarrollo reutilizado. Exigir backtest prequential por cortes y deduplicación por ID antes de promover.
4. No usar este análisis para generar submission ni gastar cuota.
