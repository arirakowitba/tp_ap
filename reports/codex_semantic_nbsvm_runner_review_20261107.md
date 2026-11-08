# Runner semántico train-only NB-SVM — revisión Codex (2026-11-07)

## Objetivo

Proveer un challenger de texto complementario al baseline two-stage sin utilizar test, submissions, scores públicos, solver ni pseudoetiquetas.

## Artefacto

- Script: `scripts/run_train_only_semantic_nbsvm_20261107.py`
- SHA-256 del contenido: `01e7a85b164970ff4c6d49cc5f6d4656c818a70be592f988b083fa639533d818`
- Commit de creación: `22060a8b8d706e9a8be01ea522a9a187a1e243b8`
- Verificación local: `python -m py_compile` exit code 0.

## Protocolo

- 70k train solamente.
- Splits exactos del runner auditado por vendedor.
- Baselines cargados desde los pares AG-0094 del mismo split; se validan IDs y labels.
- Features semánticas: title, warranty, attributes y tags.
- Hashing word 1–2 y char 3–5; TF-IDF ajustado sólo sobre outer-fit.
- Tres regularizaciones predeclaradas: C 0.35, 0.75 y 1.50.
- Selección en tune por hits, luego logloss, luego menor C.
- Confirm se transforma y evalúa únicamente tras congelar C.
- Comparación contra cb_nested_nb y blend 50/50.
- Pares por ID, seen/new y bootstrap por vendedor.

## Limitación actual

El runtime Codex no contiene los datos 70k ni los artefactos baseline AG-0094 en el filesystem local, por lo que sólo se verificó sintaxis. La ejecución queda asignada al entorno de AG en CX-0130; no se fabrican métricas.

## Prohibiciones

No generar submission desde este runner antes de RESULT y auditoría. No Kaggle ni cuota.
