# Ronda 45: red densa con FastText y datos estructurados

Hipótesis previa: interacciones no lineales de vectores FastText originales (100.000 documentos, entrenamiento sin etiquetas) y 81 variables numéricas estructuradas pueden complementar HistGB + FastText para vendedores nuevos.

Protocolo: dos particiones seller-group guardadas de ronda 39. En cada una, 41–42 mil filas fit, y tune y confirm de vendedores disjuntos. MLP (128,64), alpha=10, 35 épocas como máximo; imputación, escalado y clipping ajustados solamente en fit. Umbral de decisión .475, congelado de la base. Pesos de mezcla predeclarados 0, .1 y .25. Se selecciona el mejor acierto neto en tune y se evalúa confirm sin reajustar. No se utilizan etiquetas de test ni scores públicos.

| Partición | Tune 10%: fixes/breaks/net | Confirm 10%: fixes/breaks/net | IC 95% net confirm, bootstrap por vendedor | Peso seleccionado en tune | Confirm seleccionado |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 19/21/−2 | 23/19/+4 | [−8,+16] | 0 | 0 |
| 2 | 24/30/−6 | 15/32/−17 | [−31,−3] | 0 | 0 |

El MLP solo perdió 855/999 aciertos en tune/confirm de la primera partición y 1028/1086 en la segunda, respecto a HistGB + FastText. Mezclarlo al 10% elevó el log loss de tune de .204014 a .205723 y de .203370 a .207912. El peso seleccionado fue cero en las dos particiones, así que la prueba no produce nueva candidata.

Segmentos por dirección de los cambios con mezcla 10% (fixes/breaks): `new→used` 14/15 y 13/13 en la primera tune/confirm, 22/24 y 8/22 en la segunda; `used→new` 5/6 y 10/6 en la primera, 2/6 y 7/10 en la segunda. Todos esos conjuntos son de vendedores ausentes del fit, y tune y confirm no comparten vendedor. La caída principal del segundo confirm se concentra en `new→used`, por lo que tampoco aparece una regla direccional transferible.

Límite: esta red no recibe las predicciones OOF de los maestros de texto ni las codificaciones de categoría de HistGB. La prueba descarta esta arquitectura concreta como complemento directo; no descarta redes que incorporen esas señales sin fuga. Ambos cortes están basados en el mismo train y no equivalen a evaluación pública. La base emparejada tampoco reproduce la política exacta de la submission 33.

Reproducción: ejecutar `python3 work20/round45_mlp_fasttext_structured.py` desde el directorio raíz extraído con numpy, pandas y scikit-learn; `recovered/Analisis rpedictivo/data.zip` es el train original y `round39_fasttext_100k.npy` se construyó sobre los 100.000 textos originales mediante el código incluido. `round45_mlp_pairs_*.npz` preserva índices, etiquetas, vendedores y probabilidades pareadas. No se preparó ni envió submission.
