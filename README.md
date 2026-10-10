# Competencia AP 2026 Q2

## Orden de ejecución
1. notebooks/00_carga_datos.ipynb localiza o descarga JSONL.
2. notebooks/01_eda_limpieza.ipynb inspecciona los datos y genera cache.
3. Ejecuta un notebook de notebooks/modelos para trabajar un modelo.
4. notebooks/99_comparacion_modelos.ipynb muestra mejores accuracy.

Los modelos usan src/ap2026q2_common.py. Transformaciones aprendidas se ajustan dentro de cada split. Holdout aleatorio 80/20, holdout agrupado por familia y CV agrupada son protocolos distintos. Los datos y corridas locales se guardan en data/ap_2026_q_2 y quedan fuera de Git.

En Colab, autentica el repo privado, monta Drive y configura KAGGLE_API_TOKEN como secret. KaggleHub descarga la competencia cuando faltan los JSONL. AP2026Q2_DATA_DIR permite elegir otro directorio. Se necesita el sample_submission.csv oficial para generar archivos: el helper preserva sus encabezados, valida coincidencia exacta de IDs y alinea predicciones por ID. La celda de exportación está apagada por defecto.

results/historical_scores.csv registra métricas previas y su protocolo. Nuevos runs van a data/ap_2026_q_2/processed/experiment_runs.csv. La meta 0.94 aún no está alcanzada.


## Investigación y nuevos experimentos

- `results/research_and_validation.md`: búsqueda de datasets/soluciones relacionadas, auditoría de fuga y protocolo experimental.
- `results/experiment_attempts.csv`: diez intentos en un holdout agrupado por familia. Mejor: CatBoost enriquecido, accuracy **0,9276**; meta 0,94 aún pendiente.
- `notebooks/modelos/10_catboost_enriquecido.ipynb`: mejor pipeline y submission.
- `notebooks/modelos/11_redes_neuronales_visual.ipynb`: redes superficiales/profundas, intuición matemática, diagramas de capas, curvas de pérdida y submission MLP.

Los intentos reutilizan un holdout de 21.000 filas para seleccionar; su score puede estar optimista y necesita confirmación en nuevos folds agrupados o en Kaggle.
