# Competencia AP 2026 Q2

## Orden de ejecución
1. notebooks/00_carga_datos.ipynb localiza o descarga JSONL.
2. notebooks/01_eda_limpieza.ipynb inspecciona los datos y genera cache.
3. Ejecuta un notebook de notebooks/modelos para trabajar un modelo.
4. notebooks/99_comparacion_modelos.ipynb muestra mejores accuracy.

Los modelos usan src/ap2026q2_common.py. Transformaciones aprendidas se ajustan dentro de cada split. Holdout aleatorio 80/20, holdout agrupado por familia y CV agrupada son protocolos distintos. Los datos y corridas locales se guardan en data/ap_2026_q_2 y quedan fuera de Git.

En Colab, autentica el repo privado, monta Drive y configura KAGGLE_API_TOKEN como secret. KaggleHub descarga la competencia cuando faltan los JSONL. AP2026Q2_DATA_DIR permite elegir otro directorio. Se necesita el sample_submission.csv oficial para generar archivos: el helper preserva sus encabezados, valida coincidencia exacta de IDs y alinea predicciones por ID. La celda de exportación está apagada por defecto.

results/historical_scores.csv registra métricas previas y su protocolo. Nuevos runs van a data/ap_2026_q_2/processed/experiment_runs.csv. La meta 0.94 aún no está alcanzada.
