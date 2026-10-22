# Segunda partición para comprobar robustez

Ejecutar `python -u tp_ap/scripts/validate_fresh_round2.py` desde la raíz de trabajo. Los NPZ por etapa permiten reanudar sin repetir ajustes completados.

- Split nuevo determinístico: seed20261023, fit49.000/tune7.000/confirm14.000; subpartición tune seed20261024.
- CatBoost650árboles, depth7, lr0.065, l2=20, texto nativo BoW+NaiveBayes diccionario2500. Dos ajustes independientes49k/56k, sin early stopping ni evaluación de confirmación durante entrenamiento.
- NB-logratio sobre TF-IDF word/char, C4 fijo. Vocabularios y ratios únicamente del entrenamiento correspondiente.
- Comparaciones predeclaradas: baseline, mezcla85%baseline+15%NB y metaLR C0.1 con dos logits e interacciones seller_seen/quantity<=1/listing_free.
- MetaLR aprende solo sobre7k predicciones tune de bases49k. Confirm usa basesreentrenadas56k y meta sin modificaciones.
- No utiliza predicciones históricas, labels recuperados del test, feedback de flips ni labels de confirm para elegir hiperparámetros.
- Una segunda partición no es holdout históricamente virgen: toda la base70k se utilizó en experimentos anteriores. Sirve como prueba de estabilidad del procedimiento fijado, no para estimar un score Kaggle exacto.
- `features.pkl` y `input_arrays.npz` son caches locales de datos; no deben tratarse como nuevas fuentes de etiquetas.
- Primer lanzamiento falló por paquete CatBoost incompleto; se reparó con catboost1.2.10 no-deps y se relanzó sin cambios de parámetros.
