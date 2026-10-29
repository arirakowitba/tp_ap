# Ronda 145: OOF por vendedor con texto word/char y tabular (29-10-2026)

Sheet original 70k train, corte histórico ronda 13 56k fit/14k valid alineado por row_id e `y=used`. Reproducción de la arquitectura de dos etapas del script histórico con `GROUP=True`: codificaciones supervisadas de categoría/vendedor y profesores de texto TF-IDF word 1–2 y char_wb 2–5, ambos generados en tres folds internos `StratifiedGroupKFold` por vendedor. HistGradientBoosting final de 380 iteraciones con 52 features. Todos los ajustes y vocabularios sólo usan fit. Se compara por ID con CatBoost fuerte de ronda 13, mismo holdout.

| Modelo | Accuracy | Log loss | Seen accuracy | New accuracy |
|---|---:|---:|---:|---:|
| CatBoost fuerte | .944214 | .147403 | .962673 | .902603 |
| HistGB con profesores OOF group | .913143 | .218030 | .927408 | .880986 |

Pares: 175 fixes, 610 breaks, neto −435. El nuevo modelo no alcanza la base fuerte. Para probar complementariedad, se dividieron las 14k filas por vendedor (seed145): tune 6.887, confirm 7.113, cero vendedores compartidos. Pesos del modelo nuevo 0/.025/.05/.10/.20. Accuracy y desempate por log loss en tune seleccionaron **0**; incluso .025 perdió un acierto neto y elevó log loss .142104→.142586. Confirm congelado es idéntico a CatBoost (.941937/.152534). No seleccionar retrospectivamente un peso por el resultado de confirm.

Límite: este corte histórico no es confirm virgen, aunque la selección interna respeta vendedores. No se justifica submission. Código, JSON y pares completos: Library `libfile_17605a75ced88191b8b698d2c706f719`, SHA256 `c287cc55fe25b3787f95d19d3b9d8662c5af0b1c763b7ba2650f9863d447039c`.
