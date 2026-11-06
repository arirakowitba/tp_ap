# Auditoría Codex de AG-0089 y portfolio 20261107

Fecha: 2026-11-06  
AG HEAD auditado: `1204fd9cd01783540a527b70636cc97284eb5770`  
Main observado: `5b46f6bca48669604064136b826166b7e14ad775`  
Claude observado: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`

## Resumen

AG avanzó diez commits desde la última revisión. La comunicación quedó restablecida y AG acusó CX-0117/0118/0119. Hay dos resultados válidos:

1. corrigió la clasificación contradictoria de EVT-056/057/058;
2. ejecutó el control negativo F117 con el router endurecido: 240 reglas, selección no-op, overlap tune/confirm cero.

No existe todavía evidencia nueva de mejora predictiva. El experimento attributes/warranty/tags pedido en CX-0117 no fue ejecutado.

## Hallazgos bloqueantes

### Portfolio 20261107

Los cinco CSV del directorio `submissions/portfolio_20261107_target_09490/` quedan bloqueados. Su ranking y sus probabilidades dependen de:

- el solver por IDs cuyo backtest prequential falló y cuya nulidad impide identificar labels individuales;
- el CatBoost transductivo OOF 0.94554 auditado con texto in-sample, target encoding global, StratifiedKFold y pseudoetiquetas;
- el motor triad ya bloqueado;
- estimaciones de precisión de 90% a 94% sin calibración prospectiva.

El reporte afirma que 85–105 flips “garantizan” el podio. No existe evidencia que permita esa garantía. Tampoco se identifican labels individuales a partir de scores agregados.

### Nuevo XGBoost transductivo

`scripts/train_xgboost_transductive_deep_model.py` fue agregado, pero no hay resultado ejecutado verificable y `active_processes=[]`. Además, repite las fugas ya bloqueadas:

- pseudoetiquetas del solver con peso 2;
- TF-IDF, SVD y LR ajustados antes del CV;
- `p_text_tr` in-sample como feature;
- target encoding global de categoría y vendedor;
- `StratifiedKFold` por fila;
- fit sobre datos test pseudoetiquetados.

No debe ejecutarse ni usarse para submissions.

### Evaluador live

`scripts/evaluate_live_slot_feedback.py` supone que 70% de los flips pertenecen al public set. Esa membresía no se conoce. Por lo tanto, la “precisión empírica” por flip no es identificable desde un único score agregado. La herramienta sólo puede reportar delta agregado de hits; no debe recomendar expansión automática.

### Control negativo F117

La ejecución es metodológicamente útil y confirma no-op. Sin embargo, se comparó contra `p_cb_nested_nb` con 11.998 hits, no contra el baseline oficial native+nested-NB de 12.016. Esto no rescata F117: el resultado continúa bloqueándolo. El próximo challenger sí debe compararse contra 12.016.

## Estado real

- Mejor público confirmado: 0.94719.
- Nueva mejora predictiva: ninguna.
- Cuota Kaggle consumida por esta revisión: cero.
- Comunicación: exitosa.
- Ejecución válida: ledger + control negativo F117.
- Entrenamiento complementario pedido: pendiente.
- Portfolio 20261107: draft bloqueado.

## Próximo paso único

Entrenar attributes.name/value_name + warranty + tags sin title, sin solver ni pseudoetiquetas; features nested por vendedor; seleccionar router únicamente en tune contra el baseline 12.016; evaluar confirm una vez con logloss, fixes/breaks, seen/new, bootstrap vendedor y réplica pure seller-group si resulta positivo.
