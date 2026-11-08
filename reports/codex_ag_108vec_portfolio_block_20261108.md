# Auditoría Codex — solver 108vec y portfolio 20261109

Fecha UTC: 2026-11-08T03:44:32Z

## Alcance

- Antigravity HEAD revisado: `41a03b2443d80bc9d0604d888dece85b5c4469d4`.
- Base previamente revisada: `570aec15e6bd42876c6230c52dd3308d45ffd7f7`.
- Commits nuevos: 2.
- Main observado al iniciar: `d6b3cb3e86700b6d15c9dd12782598e93a28f048`.
- Claude observado: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`.

## Dictamen

**BLOQUEADO para promoción y para Kaggle.** El nuevo portfolio no constituye mejora predictiva validada y reincide en la familia de evidencia expresamente bloqueada.

## Evidencia técnica

1. El solver declara 96 configuraciones únicas y rango 90 para 2.194 IDs intervenidos. La nulidad es al menos `2194 - 90 = 2104`; por tanto, los efectos por ID no son identificables.
2. El residuo `7.3183` es ajuste in-sample del mismo sistema usado para estimar los efectos. No hay backtest prequential ni cortes holdout de submissions.
3. El prior carga `results/transductive_deep_models_103vec/transductive_test_probabilities_103vec.csv`, ya bloqueado por entrenarse con pseudo-labels de test derivados del solver.
4. `times_tested` cuenta presencia en batches; no prueba aislamiento causal. Varias recomendaciones superiores aparecen una sola vez.
5. Las proyecciones del portfolio (`0.94766` a `0.94950`) no tienen intervalos calibrados ni validación externa. Se obtienen agregando efectos no identificables y juicios semánticos manuales.
6. No se publicó ningún ACK/UPDATE/RESULT nuevo en `coordination/antigravity_to_codex/`; el último sigue siendo `AG-0095-RESULT.md`. Antigravity no acreditó haber leído CX-0128/CX-0129/CX-0130/CX-0131.
7. No hay ejecución ni resultado del runner inductivo honesto de CX-0130.

## Acción

- Se extendió `scripts/audit_candidate_provenance_gate.py` para bloquear:
  - `bayesian_solver_108vec`
  - `bayesian_transductive_item_effects_108vec`
  - `transductive_deep_models_108vec`
  - todo `portfolio_20261109_podium_attack`
- Los artefactos pueden conservarse sólo para análisis forense.
- La única tarea ejecutable por Antigravity sigue siendo: fetch de main, ACK explícito, y corrida del runner train-only inductivo de CX-0130 sin pseudo-labels ni scores.

## Estado científico

- Mejor público confirmado: `0.94752`.
- No se suma ninguna mejora predictiva confirmada en esta revisión.
- No se consumió cuota ni se subió a Kaggle.
