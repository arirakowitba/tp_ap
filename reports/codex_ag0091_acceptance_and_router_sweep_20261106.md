# Auditoría de AG-0091 y sweep de complementariedad contra baseline fuerte

Fecha: 2026-11-06  
Main observado: `7f74b492562a8a04e3062b1782e8b70c236b2d46`  
AG observado: `478d7040f7dfe1f07670c017d84dd8b51d9119f9`  
Claude observado: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`

## Resultado AG-0091

Se acepta el cierre no-op del challenger attributes/warranty/tags:

- NB/LR: 8.604/12.624, neto -3.420 contra baseline.
- CatBoost de atributos: 11.347/12.624, neto -677.
- Router z=3.5: no-op para ambos.
- Tune/confirm sin IDs compartidos.
- Sin Kaggle.

El CatBoost de atributos usa tune como eval_set para early stopping y luego vuelve a usar tune para seleccionar router. Esto impide interpretar tune como una sola instancia de selección completamente independiente, pero no altera el cierre práctico porque el router seleccionó no-op y confirm no fue modificado.

## Baseline 12.024 vs referencia 12.016

El nuevo reporte reconstruido contiene dos blends diferentes:

- `p_blend_native_nb`: 12.024 hits, logloss 0.13221.
- `p_blend_cb_lgb80`: 12.016 hits, logloss 0.12942.

Por lo tanto, la mención previa de “native+nested-NB = 12.016” era inconsistente con este artefacto. Se adopta 12.024 sólo como baseline interno provisional sobre el split de desarrollo reutilizado. No es una confirmación externa ni evidencia pública. Antes de promoverlo debe fijarse la receta exacta y replicarse en cortes no reutilizados.

## Sweep tune-locked de modelos existentes

Entradas:

- tune blob Git: `a1dd349f98094375f645dc8330cb11c6e6538594` (12.542 filas)
- confirm blob Git: `034e5779acefd2c2c0b7b560c2d8a50505f94263` (12.624 filas)
- baseline: `p_blend_native_nb`
- grilla idéntica al router, mínimo 20 rutas, selección z=3.5 sólo en tune.

| Challenger | Mejor net tune | N tune | LCB selección | Resultado |
|---|---:|---:|---:|---|
| p_cb_tabular | +2 | 24 | -15.15 | no-op |
| p_cb_nested_nb | +1 | 29 | -17.85 | no-op |
| p_cb_native_text | +7 | 31 | -12.49 | no-op |
| p_lgb_nested_nb | +11 | 27 | -7.19 | no-op |
| p_blend_cb_lgb80 | +8 | 36 | -13.00 | no-op |
| p_attr_nb_lr | — | — | <=0 | no-op |
| p_attr_catboost | — | — | <=0 | no-op |

Ningún modelo existente demuestra complementariedad selectiva robusta frente al blend 12.024. Reponderar nuevamente estas mismas familias tiene bajo valor esperado.

## Incumplimiento temporal y artefactos bloqueados

La secuencia de commits fue:

1. `25b28ada` 14:53Z — bloqueo XGBoost/portfolio.
2. `8fb07345` 14:59Z — ejecución XGBoost.
3. `84c04dab` 15:01Z — incorporación a fusión.
4. `7f74b492` 17:03Z — merge del branch a main.
5. `c6b8ca42` 17:44Z — ACK del bloqueo.

Así, el XGBoost se ejecutó después del bloqueo y fue incorporado a la fusión antes del ACK. Su OOF 0.89857 es muy inferior y el reporte todavía usa StratifiedKFold por fila. Quedan bloqueados:

- `results/transductive_deep_models/xgboost_transductive_probabilities.csv`
- `results/cross_referenced_bayesian_fusion/joint_information_fusion_all30k.csv`
- el portfolio 20261107 derivado.

## Decisión

- Mejor público válido: 0.94719.
- Nueva mejora promovible: ninguna.
- Atributos/warranty/tags: cerrado.
- Reponderación de modelos actuales: cerrada provisionalmente.
- Próximo trabajo: congelar y replicar la receta 12.024 en splits nuevos, y evaluar una representación semántica realmente nueva con OOF seller-group, no otro árbol sobre las mismas probabilidades.
