# Plan previo: mezcla LightGBM sólo en vendedor nuevo — 2026-11-03 09:35 UTC

Motivo exploratorio: la mezcla fija .5 LightGBM + .5 bag3 tuvo netos new-seller +2/+10/+10 en confirms reutilizados 20261124/25/26, pero netos seen -26/-8/-1. Estos cortes se solapan; la decisión de segmentar surge de sus resultados y requiere **un corte nuevo**.

## Regla congelada antes del nuevo confirm

Split `split_testlike(sellers, 20261127)` sobre las 70k etiquetas originales. Entrenar bag3 de tres CatBoost con pesos 1/3 y el LightGBM de `scripts/ablate_lightgbm_vs_bag3_20261103.py`, con extracción texto OOF y TE desde fit. Elegir iteraciones por tune. Para cada confirm:

- vendedor presente en fit: `p_gate=p_bag3`
- vendedor ausente de fit: `p_gate=.5*p_bag3+.5*p_lgb`
- ambas decisiones: `p>=.5`

No elegir peso, umbral ni condición sobre confirm. Comparar con bag3 solo usando accuracy, logloss, fixes/breaks, neto por segmento y bootstrap por vendedor. Guardar pares por ID y hashes en checkpoint privado. No reentrenar sobre test ni fabricar CSV. Si el corte nuevo no aporta mejora clara, cerrar línea y reasignar esfuerzo a una señal más amplia. Un resultado positivo aislado seguiría siendo hipótesis, no mejora pública verificada.
