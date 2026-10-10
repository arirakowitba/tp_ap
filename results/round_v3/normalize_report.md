# Normalización de texto de dominio

Se comparó el texto original con reglas fijas para acentos, abreviaturas (0 km, s/uso, p/reparar), errores y sinónimos. Las reglas no usan el target. Las representaciones TF-IDF y categorías se ajustan solo sobre cada conjunto de entrenamiento. La probabilidad SVM usada por LightGBM se genera con tres folds agrupados OOF.

- Datos íntegros: 70.000 etiquetas y 30.000 filas de test.
- Holdout agrupado: semilla 42, 56.000 entrenamiento y 14.000 validación.
- SVM original: 90,8786%; normalizado: 90,7786%.
- LightGBM con SVM normalizado y ocho flags: mejor 94,0357%.
- LightGBM con SVM original y únicamente ocho flags: 93,9714%.

No se obtuvo una mejora individual frente al mejor experimento anterior (94,0357% en este holdout). Las probabilidades se conservan para evaluar complementariedad; estos resultados son de selección, no una nueva puntuación Kaggle ni una confirmación independiente. No se generó ni promovió una submission propia.

Reproducción: `python src/normalize_domain_round.py --data-dir ../data --deadline <epoch>`; requiere `results/round_v3/cache_base/matrices.pkl` para las pruebas LightGBM. La ablation adicional se ejecuta agregando `--flags-only`. Las matrices cacheadas no se versionan; el CSV de predicciones identifica las filas.
