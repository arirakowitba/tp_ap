# Tercer corte y CatBoost pareado — ponderación discordante

Se congeló la regla de peso 1.75 descrita en la ronda previa y se evaluó en el corte seller-group 46077, con los mismos 145 insumos. Tune: +5/14123 vs HistGB+FastText, log loss .222172→.221611. Confirm: −4/14648, log loss .215678→.215075, bootstrap vendedor 95% del neto [−30,+22]. La secuencia de netos confirm ahora es +22,+9,−4; no hay ganancia estable de accuracy.

Se recuperaron predicciones de CatBoost entrenado en el mismo fit de cada corte y se alinearon por índice/etiqueta/probabilidad FastText. Están preservadas solo para los desacuerdos HistGB/FastText en confirm, no para holdout completo. En esos subconjuntos la variante obtuvo +3/254, +6/293 y +14/273 frente a CatBoost (umbral .5). Esto es un análisis segmentado posterior: no demuestra que supere CatBoost globalmente ni a la política exacta de 48. No promover submission.

Checkpoint de pares, script y reporte: Library [pending]. Los tres cortes reutilizan train históricamente. No se usaron scores Kaggle como etiquetas.
