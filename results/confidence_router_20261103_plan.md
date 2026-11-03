# Plan congelado: router por confianza bag3 vs LightGBM — 2026-11-03 10:42 UTC

En cuatro confirms de desarrollo (seeds 20261124/25/26/27), LightGBM solo pierde mucho contra bag3, pero cuando bag3 está cerca de .5 y LightGBM discrepa con alta confianza, hubo saldos +10/+1/+9/+6. Estos cortes se solapan; se inspeccionaron varias bandas de margen, por lo que la observación es **selección sobre desarrollo**, no confirmación.

## Validación nueva predeclarada

Usar split `split_testlike(sellers, 20261128)` de los 70k train originales. Entrenar bag3 con seeds 20261128, 20262125, 20263122 y LightGBM de `scripts/ablate_lightgbm_vs_bag3_20261103.py`, ambos con texto word+char OOF y TE desde fit, early stopping sólo tune. Regla fija, sin selección en confirm:

```python
bag = p_bag3 >= .5
lgb = p_lgb >= .5
take_lgb = (bag != lgb) & (abs(p_bag3 - .5) <= .1) & (abs(p_lgb - .5) >= .2)
pred_router = where(take_lgb, lgb, bag)
```

Comparar contra bag3 a umbral .5 en el **mismo confirm**: accuracy, logloss sólo de probabilidades si se especifica una probabilidad combinada, fixes/breaks, seen/new, bootstrap vendedor, pares por ID y hashes. Un saldo positivo aislado con IC cruzando cero no promueve submission. Un resultado negativo cierra el router. No Kaggle ni cuota.
