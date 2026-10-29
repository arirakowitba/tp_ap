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

## Corrección de alineación y contraste decisivo con ronda 13

La afirmación previa de que los índices de ronda 13 no coincidían con el JSON era **errónea**: el chequeo había invertido la clase positiva. El compacto original del Google Sheet `1O0MY-uSIh1SBBNhxWqNOuqVst8la0fBnqfOqz-_7NhE` usa `y=1` para **used**; nuestro JSON auxiliar había codificado `y=1` para **new**. Se exportó el Sheet original, se recorrieron sus 70.000 filas train y se cotejaron `row_id`, `seller_id` y `y=(condition=='used')` contra el JSON original: **0 discrepancias en las tres columnas**. Además, los 14.000 `rows/y/seen` de los tres NPZ de ronda 13 coinciden con el JSON y el fit de `partition.npz`. La ordenación por índice es válida para esta fuente verificada; el código y la evidencia están en el checkpoint indicado abajo.

Se aplicó la misma regla predeclarada de este informe (prior de vendedor fit con alpha=3, n>=3, distancia >=.30, contradicción, mezcla .6/.4) a probabilidades **fuertes** de ronda 13 en el tercer corte 56k/14k, sin usar etiquetas de valid para construir el prior. Resultado:

| Base fuerte | Filas activadas | Fix/break | Neto | Accuracy antes→después | Logloss antes→después | IC95 neto bootstrap vendedor |
|---|---:|---:|---:|---:|---:|---:|
| HistGB | 59 | 4/53 | −49 | .943571→.940071 | .148187→.150658 | [−77,−28] |
| LightGBM | 60 | 5/52 | −47 | .943500→.940143 | .147307→.149700 | [−74,−26] |
| CatBoost | 56 | 3/51 | −48 | .944214→.940786 | .147403→.149819 | [−76,−27] |

**Descartar la regla para los modelos fuertes.** Su aparente mejora sobre bases .79–.89 era dependiente del comparador: al contradecir a modelos ~.94, el historial del vendedor casi siempre rompe un acierto. No generar submission ni sumar esta regla a la 48. Este corte histórico está solapado con otros cortes y no es confirm virgen; la pérdida es suficientemente grande para detener esta hipótesis, no una estimación precisa del test.

Código: `src/revision_submissions/round13_strong_gate_20261028.py`. Pares por ID y métricas en checkpoint Library `libfile_e56b30045cc88191a8921667d2f2366d`. Fuente del compacto: Sheet original, no reconstrucción de ronda 9. El score público no se usó como etiqueta.
