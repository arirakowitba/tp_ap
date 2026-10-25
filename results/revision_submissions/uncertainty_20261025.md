# Incertidumbre por fila y aprendizaje secuencial (25 oct 2026)

Archivo completo de 30.000 filas: [AP_2026_Q2_row_distributions_20261025.csv](https://chatgpt.com/library?file_id=libfile_e7888e22317481919baa739834321c58&account_id=personal). El script `estimate_row_uncertainty.py` reproduce la tabla si se colocan los NPZ del checkpoint 13 y 30 junto a los CSV reconstituidos. Los TSV y bases del repositorio conservan los cambios de 37 submissions exactas.

## Qué significa cada distribución

Por ID, `p_used_model_raw` es el blend HistGB/CatBoost entrenado con train. La distribución predictiva de la clase es Bernoulli(`p_used_train_binned`). Para cuantificar incertidumbre de calibración, se usan las 14.000 etiquetas del tercer corte y 20 celdas: vendedor visto/nuevo × decil del blend Hist/Cat. En cada celda, la tasa de used tiene posterior Beta(`beta_alpha`, `beta_beta`), con prior de concentración 10 y media del centro del decil; columnas low/high muestran percentiles 5/95 **de la tasa de la celda**, no una garantía de cobertura individual ni independencia del tercer corte. Las 37 predicciones históricas por fila son votos *correlacionados*: `history_used_votes_of37` describe cambios, no 37 observaciones de etiquetas. `public_row_impact_identifiable=no` para todas.

`expected_flip_gain_vs33` es 2p−1 para cambiar a used o 1−2p para cambiar a new, aplicado a la decisión 33. Esta cifra exige que las probabilidades sean fiables *condicionadas en el desacuerdo con 33*, algo refutado por resultados recientes. Sirve como diagnóstico y no como ranking listo para submission.

| Submission | Flips | Ganancia esperada por modelo (30k) | Delta score público frente a 33 |
|---:|---:|---:|---:|
| 35 | 13 | +4.75 | −0.00005 |
| 36 | 18 | −1.99 | −0.00043 |
| 37 | 146 | +116.55 | −0.00124 |
| 38 | 201 | +150.75 | −0.00191 |
| 39 | 1228 | +433.24 | −0.01638 |
| 40 | 4 | −0.66 | −0.00019 |
| 41 | 6 | +4.73 | pendiente |

La contradicción 37–39 es grande. Causas posibles: selección sesgada hacia errores propios del modelo, probabilidad de 33 no representada, shift, reuse del corte y dependencia por vendedor. No se adjudica causalidad ni una etiqueta a ningún ID. El score público observa solo una fracción desconocida del test y no es comparable directamente con la suma esperada de 30k; aun así, el signo y la magnitud invalidan usar esa suma de manera ingenua.

## Actualización tras la siguiente prueba

1. Incorporar score, CSV, hash y delta exacto por ID al inventario. Comparar solo bloques disjuntos o anidados; registrar el saldo agregado con supuesto explícito sobre tamaño público y precisión de score.
2. Mantener las Beta de calibración basadas en etiquetas de train. Actualizarlas con nuevas etiquetas independientes, nunca con diferencias de leaderboard por fila.
3. Evaluar en tune/confirm una regla para los desacuerdos con 33: medir fixes y breaks por dirección, vendedor, decil y dependencia de título. Conservar un confirm sin seleccionar.
4. Para candidatos futuros, exigir ganancia positiva consistente en esos segmentos y limitar el tamaño del bloque. Usar resultados públicos previos para descartar familias amplias que fallaron, sin optimizar filas según supuestas etiquetas inferidas.
5. La 41 mide seis nuevos vendedores del bloque 37. Esperar su score antes de definir otra prueba sobre ese mismo bloque.

La meta pública de 96% requerirá mejoras genuinas de generalización; las probabilidades actuales no justifican proyectarla.
