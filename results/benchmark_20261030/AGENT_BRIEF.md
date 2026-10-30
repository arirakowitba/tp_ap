# Brief compartido de agentes — 30-10-2026 (rama `antigravity/benchmark-recovery-20261030`, base main `b83b64b`)

Estado: mejor pública 48=.94633. 72=.94623, 73=.94600, 74=.94614, 75=.94633 (empate, no promovido), 76=.94633 (duplicado exacto de 48).
Scores públicos = evidencia agregada de bloques; **nunca** etiquetas por ID ni targets. No subir a Kaggle (sólo Ariel). Codex elige envíos.
No expandir bloques 72–74. No presentar 75 como mejora. No repetir SVD+MLP, NB-SVM profundo, gates de historial vendedor, fin_stack, NMF ni léxico aislado sin intervención nueva identificable.

## Protocolo común
- Datos: `data/ap_2026_q_2/raw/train_data.jsonlines` (70k etiquetas reales), `test_data.jsonlines` (30k; ID de submission = número de línea 1..30000). Polaridad: used=1.
- Partición de desarrollo canónica: `split_seller_groups` de `src/revision_submissions/deep_catboost_nbsvm_ablation_20261029.py` (R151/153), semillas 2026 y 314159. Estos confirm **ya fueron inspeccionados**: se llaman "desarrollo/reutilizados", no vírgenes.
- Evaluación adicional predeclarada: misma función con semilla **20261030**, ejecutada una sola vez por configuración congelada. Sus filas tampoco son vírgenes (aparecieron en otros confirm); Agente 3 audita el solapamiento.
- Selección (hiperparámetros, iteraciones, pesos de blend, umbral) sólo con fit (inner GroupKFold por vendedor) y tune. Congelar antes de confirm. TE, texto OOF y stacks: cross-fit dentro de fit.
- Reportar: accuracy, log loss, fixes/breaks vs mejor base, vistos/nuevos, direcciones, bootstrap por vendedor (2000), estabilidad entre semillas.
- Hilos: Agente 1 `thread_count=8`; Agente 2 `6`; Agente 3 `2`.
- Artefactos pesados (modelos `.cbm`, matrices) en `models/benchmark_20261030/` (gitignored) con SHA256 en manifiestos versionados; predicciones/métricas en `results/benchmark_20261030/`.
- Nunca sobrescribir CSV/archivos existentes; versionar. Los agentes no hacen commits: avisan al coordinador, que comitea y pushea checkpoints.

## Agentes
1. **Benchmark** (`src/benchmark_20261030/`): reproducir el mejor CatBoost profundo, explicar R157 .915/.911 vs R151/153 .924/.920, reentrenar referencias en particiones idénticas, persistir matrices/modelos/predicciones con hashes.
2. **Complementario** (`src/complementary_20261030/`): hipótesis nueva dirigida a errores residuales de la base recuperada (texto neuronal sobre tokens/subwords originales o boosting con matriz profunda completa + tokens crudos).
3. **Evaluación y candidatos** (`src/eval_20261030/`): evaluador pareado, auditoría de solapamiento histórico, registro de envíos puntuados, detector de duplicados, matriz test por ID, candidatos con propósito y decisión posterior explícitos.
