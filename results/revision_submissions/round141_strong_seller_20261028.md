# Segunda validación etiquetada de historial de vendedor (28-10-2026)

Código: [strong_seller_20261028.py](../../src/revision_submissions/strong_seller_20261028.py), importa el protocolo de [seller_diagnostic_20261028.py](../../src/revision_submissions/seller_diagnostic_20261028.py). Checkpoint reproducible con pares confirm, resultados y ambas fuentes: Library `libfile_36286e864588819191a608dcece38410`; SHA256 ZIP `079808a38ff6d1f1814bd6f8e721da3743b16076455c62d389711d16a67e266f`. Usa exclusivamente 70k train original de data.zip y sus etiquetas.

## Arquitectura y control de fuga

Modelo base de dos etapas: TF-IDF word 1–2 título con regresión logística, cuyas probabilidades **OOF por vendedor** se generan en cuatro folds internos del fit. HistGradientBoosting de 220 iteraciones, 22 hojas, recibe esa probabilidad y metadatos numéricos/categóricos codificados sin target; para tune/confirm, el vectorizador y texto final se ajustan sólo con fit. Se excluye seller_id del modelo base, pero la regla usa el prior etiquetado de compañeros en fit, suavizado alpha=3. Regla predeclarada: n_fit>=3, prior a >=.30 de 0.5 y contradicción binaria con base; reemplazar probabilidad con .6 prior + .4 base. El protocolo de splits es el de round139, con vendedores completamente nuevos y filas vistas retenidas. Confirm no interviene en entrenamiento ni selección.

| Seed | Confirm filas | Vistos | Filas activadas | Fix/break vs base | Neto | Acc base→regla, total | Logloss base→regla, total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 12.906 | 3.950 | 107 | 84/22 | +62 | .888424→.893228 | .271320→.266665 |
| 314159 | 12.844 | 3.831 | 125 | 91/33 | +58 | .883214→.887730 | .271453→.265904 |

Tune: 2026 +50 (79/29), 314159 +60 (87/27). Bootstrap por vendedor sobre neto confirm, 2.000 remuestreos: 2026 IC95 [+38,+87], 314159 [+28,+90]. Los confirm se solapan en 2.235 IDs y 1.984 vendedores, por lo que las dos ganancias no se suman como evidencia independiente. En nuevos no se activa la regla ni cambia métrica. Hashes SHA256 pares: seed2026 `99ecd28ab3e023b97ff35d7f1f63c6c69e54a1fbfed155565281371d41c0e7af`, seed314159 `ad3925fe7c9f402a5de207371ed9c5240f92371b5fd10deb4c037a9197d31732`.

## Decisión

La mejora sobre un modelo tabular de dos etapas es real en estos cortes y tiene menor tamaño que la mejora sobre texto solo. **La base sigue siendo débil (~.88) frente a modelos principales (~.95 en sus CV distintas)**. Las métricas no se comparan directamente porque tampoco comparten folds, features ni preparación; este experimento no autoriza un CSV para Kaggle. La asimetría pública de `fin_stack` en vistos→new es compatible con un problema de historial, pero no prueba que esta regla corrija ese stack. La siguiente prueba decisiva es exactamente la misma comparación pareada con probabilidades externas de HistGB+FastText/CatBoost y una proxy etiquetada de 48. Evitar añadir una regla basada en seller a la 48 sólo por estos resultados.
