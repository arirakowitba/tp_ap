# Competencia AP 2026 Q2

## Orden de ejecución
1. notebooks/00_carga_datos.ipynb localiza o descarga JSONL.
2. notebooks/01_eda_limpieza.ipynb inspecciona los datos y genera cache.
3. Ejecuta un notebook de notebooks/modelos para trabajar un modelo.
4. notebooks/99_comparacion_modelos.ipynb muestra mejores accuracy.

Los modelos usan src/ap2026q2_common.py. Transformaciones aprendidas se ajustan dentro de cada split. Holdout aleatorio 80/20, holdout agrupado por familia y CV agrupada son protocolos distintos. Los datos y corridas locales se guardan en data/ap_2026_q_2 y quedan fuera de Git.

En Colab, autentica el repo privado, monta Drive y configura KAGGLE_API_TOKEN como secret. KaggleHub descarga la competencia cuando faltan los JSONL. AP2026Q2_DATA_DIR permite elegir otro directorio. La competencia acepta submissions con encabezado `ID,condition` y IDs de fila desde 1 hasta N; el helper aplica esa convención automáticamente aunque Kaggle no publique sample_submission.csv. La celda de exportación está apagada por defecto.

En Windows local, si ya tienes los archivos, puedes evitar KaggleHub con `os.environ["AP2026Q2_DATA_DIR"] = r"C:\ruta\a\la\carpeta"` antes de ejecutar `download_competition()`. Si faltan los JSONL y no hay credenciales, el helper detiene el flujo con instrucciones para `KAGGLE_API_TOKEN`, `kaggle.json` o `KAGGLE_USERNAME`/`KAGGLE_KEY`.

results/historical_scores.csv registra métricas previas y su protocolo. Nuevos runs van a data/ap_2026_q_2/processed/experiment_runs.csv.

## Investigación y nuevos experimentos

- `results/research_and_validation.md`: búsqueda de datasets/soluciones relacionadas, auditoría de fuga y protocolo experimental.
- `results/experiment_attempts.csv`: diez intentos en un holdout agrupado por familia. Mejor histórico: CatBoost enriquecido, accuracy **0,9276**.
- `results/experimental_rounds_15.csv`: tabla consolidada de 15 configuraciones/candidatos, con accuracy, balanced accuracy, macro F1 y protocolo.
- `results/experiments_incrementales.md`: lectura ejecutiva, modelos candidatos y próximos pasos.
- `notebooks/12_experimentos_incrementales.ipynb`: guía educativa con esquemas Mermaid, hiperparámetros, gráfico comparativo y celda de submission.
- `notebooks/modelos/10_catboost_enriquecido.ipynb`: mejor pipeline y submission.
- `notebooks/modelos/11_redes_neuronales_visual.ipynb`: redes superficiales/profundas, intuición matemática, diagramas de capas, curvas de pérdida y submission MLP.

El mejor score local documentado es **0,927619** (CatBoost + SVD + 75 atributos + interacciones). La meta **0,95** aún no se alcanzó. El holdout de 21.000 filas se reutilizó para selección, por lo que el score puede estar optimista; el siguiente ciclo debe confirmar con un segundo fold agrupado y stacking out-of-fold.
