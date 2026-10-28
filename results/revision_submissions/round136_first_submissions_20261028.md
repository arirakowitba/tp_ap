# Revisión de las primeras submissions del 28-10

HEAD Claude revisado: `37e6106a350a75ccfbcfdf7708583c05aeea3950` (un commit tras `612303c`). Agrega `fin_stack`, bandas 59–62 y modelo puro 63. La base 48 declarada tiene SHA256 `3f8d08e0e61854c2805a02140b1f7428016d63deae4bd7d6c820c7478f6c9722`.

## Verificación de los primeros CSV

| Prueba | Flips nuevos sobre 48 | Dirección | SHA256 CSV | Library |
|---|---:|---|---|---|
| 59 | 67 | 34→used, 33→new | `b34723587e1ef09c897a4af5768a2ea025c0dd313c04bd6494d5f0289899f373` | `libfile_4fa30a298c2481919e6417536c5d5492` |
| 60 | 252 | 71→used, 181→new | `1f6f5d3d1a034f77369ae3cce27c0bc36a8e495d0f241fc2bdeb3d6c2fe942cf` | `libfile_9c36a438bcc08191916911e73f385960` |

Ambos CSV tienen 30.000 IDs únicos 1–30000 y encabezado `ID,condition`. Su diferencia entre sí es 319 filas = 67+252, consistente con bloques disjuntos. Los hashes coinciden con el manifiesto de Claude. No se enviaron a Kaggle.

## Interpretación y orden

Enviar primero 59: prueba de 67 desacuerdos de máxima confianza que no se habían probado. Recibir su score exacto. La 60 mide una segunda banda disjunta; si la 59 cae, conviene dividir la 60 por dirección antes de gastar otro cupo, pues el efecto usado/nuevo puede ser asimétrico. Si la 59 mejora, enviar 60 para estimar decaimiento por confianza. No enviar 61/62/63 automáticamente.

Advertencia: `fin_stack` reporta .9548/.1191 en CV aleatoria, pero el código `final_stack_coordinador.py` apila predicciones base OOF mediante otra StratifiedKFold con la misma semilla. Los modelos base OOF de las filas de entrenamiento del meta modelo pueden haber usado etiquetas del fold que el meta modelo evalúa; además la TE global del nivel base ya fue auditada. Esta cifra no es validación externa del stack ni calibración de `P(48 mal)`. Por ello las bandas son pruebas exploratorias, sin garantía de ganancia. Los scores públicos son agregados y jamás se convierten en etiquetas por ID.

Siguiente validación de modelo: reconstruir TE y modelos base dentro de cada fold externo, congelar pesos/hiperparámetros en tune, medir pares contra HistGB+FastText y CatBoost en confirm seller-group seen/new con bootstrap por vendedor y log loss. Hasta entonces no promover 63 como modelo puro.
