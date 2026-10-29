# Diagnóstico de historial de vendedor frente a texto, sólo train

Fecha local 28-10-2026. Código [seller_diagnostic_20261028.py](../../src/revision_submissions/seller_diagnostic_20261028.py). Checkpoint con código, métricas y predicciones pareadas de confirm: Library `libfile_c572d93a1b2c81918b739386df4e4190`; SHA256 de código `1533a98e22799e2edc5ed107bc91d20d4e502a395a42b27f74b8cf34591ec6de`. Entrada: `data.zip` original, train_data.jsonlines (70.000 etiquetas); test_data no leído.

## Hipótesis y protocolo

La caída pública en vistos→new de `fin_stack` sugiere medir si el historial etiquetado de otros avisos del vendedor contradice al texto. Dos semillas 2026/314159: 10% de vendedores completos por tune y otro 10% por confirm (nuevos); además se retienen 10%+10% de filas de vendedores restantes, que conservan otras filas en fit (vistos). Fit ≈45k, tune ≈12k, confirm ≈13k. Vectorizador TF-IDF word 1–2 y regresión logística de título fit exclusivamente sobre fit. Prior de vendedor suavizado alpha=3 calculado exclusivamente sobre fit. Regla fijada antes de mirar confirm: si vendedor tiene >=3 filas en fit, |prior-.5|>=.30 y contradice clasificación texto, combinar .6 prior + .4 texto. Sin ajuste de la regla con tune. No usar scores públicos como labels.

| Seed | Confirm total | Confirm vistos | Gate | Fix/break vs texto | Δ aciertos todos | Logloss todos antes→después |
|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 12.906 | 3.950 | 218 | 196/20 | +176 | .456308→.442727 |
| 314159 | 12.844 | 3.831 | 273 | 243/29 | +214 | .442644→.427020 |

En tune, 2026: 224/28, +196; 314159: 223/23, +200. Sólo se modifica el segmento visto. Bootstrap de 2.000 remuestreos de vendedores vistos, diferencia neta de aciertos en confirm: seed 2026 IC percentil 95% [+136,+222]; seed 314159 [+166,+267]. Los cortes comparten 2.235 filas confirm y 1.984 vendedores confirm, por tanto **no** son réplicas independientes ni se deben sumar sus saldos. La partición construida tiene ~69–70% de filas de vendedor nuevo en confirm, más que el ~28% declarado para test; ponderar o diseñar un confirm representativo antes de extrapolar accuracy.

## Límite decisivo

La base de título sola obtiene 0,787–0,799 en confirm total: **no compite** con HistGB+FastText, CatBoost, la política 48 ni `fin_stack`. La selección por contradicción con este baseline débil hace que el rendimiento 89–90% del gate sea condicional y no transportable al conjunto donde un modelo fuerte contradice al vendedor. Este ensayo es un control de integridad y mecanismo, **no una candidata a submission**. El historial de vendedor ya se había explorado en otras rondas; la aportación nueva aquí es la comparación estrictamente pareada y la advertencia sobre baseline. Próximo paso: repetir el mecanismo exactamente sobre probabilidades OOF de los modelos fuertes con fit/tune/confirm externos disjuntos, incluyendo sellers nuevos y vistos, y sólo promover si gana frente a ellos. Auditar especialmente TE y las predicciones de compañeros que pueden depender de la etiqueta propia.

## Procedencia y reproducibilidad

CSV pareados incluyen ID train, etiqueta real, seller, probabilidad texto, prior calculado en fit, conteo en fit, gate y predicción combinada. Hashes SHA256 de pares 2026: `fcb72a8b52fc8934015791ce264af37cd2b8732688bcd058f6c06f7a60428544`; 314159: `e5604f9d2a5dbf939b0a49c9e01cb576bab43c0a5351be612878be8d36de145f`. Result JSON SHA256 `ec04df501dd45087953531ed526210cc1b56701dd5696b3aaf980ca26a9e1dc2`. El código y pares están completos en el checkpoint. Ninguna predicción de test ni submission se generó.
