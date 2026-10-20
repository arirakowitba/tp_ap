# Entrenamiento y feedback posterior: 20/10/2026

**Estado:** la submission 16, con 64 flips sobre la 15, ya está en Library. Esta ronda entrenó modelos nuevos antes de recibir su score. La imagen que comunica el score de la 15 fue recibida, pero el valor numérico de su OCR es ambiguo: falta transcribirlo literalmente para convertirlo a aciertos públicos. No se supone ninguna etiqueta individual de test.

## Cambios evaluados

Se probó una regresión logística con TF-IDF de palabras y caracteres, categorías y variables numéricas, y un LightGBM con proyecciones SVD de texto. Con variables básicas, el primer fold dio respectivamente **0,90632** y **0,89393** de accuracy. Al incorporar las 340 variables tabulares y de texto ya preparadas, el mismo fold dio **0,91026** y **0,91485**. Esos cuatro resultados fueron inferiores a la familia CatBoost; se detuvieron antes de consumir los otros dos folds.

Luego se entrenaron dos CatBoost con señales nuevas de condición en **título, atributos y garantía**: menciones de nuevo/usado, reparación, empaque y contradicciones entre campos. Se reconstruyeron dentro de cada fold los target encodings y el modelo de texto supervisado. Los tres folds agrupados conservan las mismas particiones del ensamble previo para poder comparar por fila; estas particiones ya se usaron otras veces, así que no son una evaluación ciega nueva.

| Modelo o combinación | Aciertos OOF / 70.000 | Accuracy | Log loss |
|---|---:|---:|---:|
| Ensamble anterior, referencia | 65.317 | 0,933100 | **0,167032** |
| CatBoost simétrica con señales nuevas | 65.216 | 0,931657 | 0,170840 |
| CatBoost depthwise con señales nuevas | **65.357** | 0,933671 | 0,168262 |
| 40 % anterior + 60 % depthwise nueva | **65.359** | 0,933700 | **0,167026** |
| 25 % anterior + 75 % depthwise nueva | 65.362 | **0,933743** | 0,167353 |

La depthwise nueva ganó 305 filas que fallaba la referencia y perdió 265 que la referencia acertaba: **+40 netas sobre 570 desacuerdos**. El ensamble 40/60 suma 42 aciertos frente a la referencia y mantiene prácticamente el mismo log loss. El peso se comparó en estos mismos datos reutilizados; la diferencia es pequeña, susceptible a selección de variante, y **no prueba una mejora en test oculto**.

## Qué nos dicen las submissions anteriores

El público usa, según lo informado, 70 % de las 30.000 filas. Bajo asignación aleatoria de filas a público, con una previa Beta(1,1) y marginalizando cuántos ID del lote son públicos, los saldos observados tienen esta lectura **por cohorte**:

| Cohorte | Saldo público observado | Media posterior de acierto de los flips | Intervalo posterior 90 % | Probabilidad alternativa media del nuevo modelo |
|---|---:|---:|---:|---:|
| P, 32 casos | +4 | 0,584 | 0,417–0,745 | 0,638 |
| Q, 144 casos | −19 | 0,407 | 0,328–0,487 | **0,612** |

**Diagnóstico central:** el nuevo CatBoost mejora la accuracy de entrenamiento, pero mantiene un exceso de confianza en Q: apoya el cambio de 127 de sus 144 casos pese al saldo público agregado negativo. Sus probabilidades no son una calibración confiable para seleccionar grandes lotes parecidos a Q. Para los 16 casos Q4 que integran la 15, da apenas 0,539 de probabilidad alternativa media; el usuario informa que la 15 funcionó muy bien. Ninguno de estos datos identifica qué filas concretas de Q o de la 15 son públicas o verdaderas.

Después de excluir la cohorte 16, quedan **1.448 candidatos** nuevos bajo los filtros actuales. La nueva depthwise apoya flips en 39, y solamente 6 superan 0,55. Se creó `future_candidate_ranking_not_calibrated.csv` para inspeccionar familias y desacuerdos, **no** para aplicar sus etiquetas automáticamente. Abrir lotes mucho mayores exige mejor evidencia independiente, no solo subir el umbral de cobertura.

## Decisión para las pruebas 17–20

Mantener 15 como control y esperar el score de la 16. Evaluar el aporte real de su cohorte de 64; si es positivo, buscar una extensión nueva y disjunta respaldada por validación y por el patrón del lote. Si es negativo, usar la tabla de desacuerdos para cambiar el tratamiento o entrenar una representación complementaria antes de la 17. Reservar la 20 para combinar cohortes positivas una vez conocidos los cuatro resultados. La nueva profundidad de árboles sirve para ampliar señales de selección; **no altera el CSV de la 16 ya preparada**.

Archivos reproducibles: `comparison.json` y `audit_cv_public.json` conservan métricas y supuestos; los `.npz` guardan predicciones OOF y de test; los tres `.cbm` de depthwise contienen sus modelos por fold. Los scripts `train_text_cue_catboost_20261020.py` y `analyze_models_after_probe15_20261020.py` documentan preparación, entrenamiento y análisis. La matriz de candidatos contiene puntuaciones **no calibradas**.
