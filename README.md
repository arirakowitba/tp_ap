# Competencia AP 2026 Q2

## Revisión de submissions y diseño de flips

- [23_revision_submissions_y_flips.ipynb](notebooks/23_revision_submissions_y_flips.ipynb): notebook ejecutado y didáctico para exponer el historial, los casos dudosos, la inferencia estadística y el plan adaptable de 75 pruebas. Abre con una grilla guiada de 100 filas, muestra unanimidad y desacuerdo, y calcula cuántas pruebas hacen falta para cobertura, ganancia probable o identificación individual.
- [Datos y guía de actualización](results/revision_submissions/README.md): snapshot ligero y trazabilidad de los resultados; no hace falta entrenar para ejecutar el notebook.
- Corte documentado: 18 de octubre de 2026. Mejor público observado **94,476%**; control combinado **94,485% inferido, sin score propio**; holdout histórico de referencia **94,3952%**. La submission 10 permanece pendiente mientras no se incorpore su resultado.
- Los apartados históricos de abajo conservan las métricas de cada ronda. Para la revisión consolidada de submissions, usar el notebook 23 y su snapshot versionado.


## Pipeline maestro y explicación visual

- `notebooks/21_maestro_pipeline.ipynb`: auditoría, split por familia, variables, SVM OOF, TPE, selección de ensamble, comprobación congelada, refit y exportación.
- `notebooks/22_caso_desde_cero.ipynb`: explicación desde cero con 18 figuras (muchas multipanel), ejemplos reales, errores, calibración y formato de submission.
- Mejor referencia previa: **94,2095%** en selección agrupada; **94,2143%** en comprobación; **94,428% en Kaggle público informado por el usuario**.
- La copia recuperada de `data.zip` tiene 70.000 filas válidas de train y 30.000 de test. El modo normal bloquea archivos incompletos.
- Instalar `requirements_optimized.txt` y configurar `AP2026Q2_DATA_DIR` con la carpeta de los JSONLines.
- Ejecutar todo: abrir el maestro y usar **Run all**. La celda de control impone un límite global de 90 minutos.
- Revisar resultados sin entrenar: `python src/execute_project_notebooks.py --data-dir RUTA --mode report`.
- Pruebas: `python -m unittest discover -s tests -v`.
- Búsquedas complementarias: `relation_features.py`, `cat_native_round.py`, `extended_booster_search.py`. Sus métricas permanecen separadas hasta evaluarse en el mismo holdout.

Se versionan código, notebooks ejecutados, parámetros, auditorías, métricas y submissions. Las matrices, probabilidades intermedias y modelos regenerables quedan fuera de Git. El 95% es la meta; un resultado de selección por sí solo no se declara confirmado.

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


## Ronda complementaria del10 de octubre

Estado: **interrumpida**, con57 configuraciones y métricas respaldadas. La búsqueda conjunta llegó a94,0571% de selección; el stacking de3 bases obtuvo94,0143% en confirmación, por debajo del94,2143% local anterior. No hay una nueva submission confirmada ni evidencia de95%. La referencia pública informada por el usuario sigue siendo94,428%.

El cierre, límites y ensayos pendientes están en [results/round_v3/README.md](results/round_v3/README.md). El maestro actualizado usa `AP_RUN_MODE=report` por defecto para inspeccionar evidencia sin entrenar. Para repetir entrenamientos, instalar `requirements_round3.txt`, configurar los datos oficiales y usar `AP_RUN_MODE=full`, con un presupuesto global de90 minutos. El modo full completo no fue reejecutado en la recuperación.

El notebook ilustrativo conserva18 figuras históricas y agrega3 figuras de esta ronda; su apéndice se ejecutó sin errores. Se verificaron la integridad de la submission anterior, la reconstrucción de caches de stacking3/4/5 y la reserva de tiempo para confirmación ante un timeout de búsqueda.


## Ronda de meta calibración (11 de octubre)

Se agregó `src/meta_calibration_round.py` para ajustar Platt e isotónica con las predicciones agrupadas existentes y evaluarlas en seed 731 sin usar esa confirmación para calibrar. El informe está en `results/round_v4/README.md`. La calibración no alcanzó 95% ni justificó reemplazar el ensamble. No se entrenaron nuevos modelos: el JSONL no está disponible en las carpetas locales y la recuperación del `data.zip` no pudo materializarse en este entorno.

