# Ronda 148: NB-SVM OOF en CatBoost del compacto original (29-10-2026)

Motivación: validar la señal de Antigravity en la misma partición histórica de ronda 13 y contrastar con una base más fuerte. Fuente Sheet original íntegro, 70k train, `y=used=1`, fit 56k/valid14k con índices y etiquetas cotejados con NPZ. Se generan 11 claves TE y profesores word/char con folds internos por vendedor dentro de fit; NB-SVM word1–2 de 100k features, log-count ratio **recalculado en cada GroupKFold**. Dos CatBoost idénticos (700 iter, depth6, lr .045, L2 12) sobre esas features; sólo el segundo recibe NB-SVM OOF. Sin scores públicos como etiquetas.

| Modelo, mismos 14k | Accuracy | Log loss | Seen | New |
|---|---:|---:|---:|---:|
| CatBoost compacto | .915286 | .215091 | .930192 | .881683 |
| CatBoost compacto + NB-SVM | .916786 | .211335 | .931120 | .884472 |
| CatBoost fuerte histórico ronda13, referencia | .944214 | .147403 | .962673 | .902603 |

Dual-text: 116 fixes/95 breaks, neto +21. Split adicional por vendedor de los 14k (seed148) sin vendedores compartidos: tune 7.038, +4 neto, LL .222381→.217776; confirm 6.962, 59/42 fixes/breaks, **+17**, LL .207722→.204823, seen +7/new +10; bootstrap vendedor CI95 neto [−3,+37]. El modelo ya estaba entrenado antes de esta división, que es diagnóstico de estabilidad dentro de un holdout histórico, **no confirm virgen**.

Interpretación: señal NB-SVM complementaria en una base .915, pero la arquitectura aún pierde ~384 aciertos frente al CatBoost fuerte .944 en las mismas filas. La mejora no demuestra utilidad sobre la base fuerte ni autoriza submission. Debe integrarse a la arquitectura fuerte con features profundas y OOF de la misma procedencia o probarse sobre una validación nueva equivalente. Además los cortes de Antigravity se solapan (2.337 IDs) y su ganancia frente a CatBoost anterior es +2/+46, ver round147.

Checkpoint código, pares por row_id y métricas: Library `libfile_b13a90325e14819197b642e40d329032`, SHA256 ZIP `20fe7ddb33bbfc5ac6607b2cad73b412793ab1cb28ba6926cecc18a5e223437f`.
