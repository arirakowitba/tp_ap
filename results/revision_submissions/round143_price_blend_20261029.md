# Ronda 143: mezcla tabular sobre CatBoost fuerte (29-10-2026)

Se tomaron las probabilidades por ID del CatBoost fuerte de ronda 13 y del modelo tabular con precio relativo de ronda 142. El holdout histórico de 14.000 filas se dividió por `seller_id` con `GroupShuffleSplit(random_state=143)` en tune 6.775 y confirm 7.225, sin vendedores compartidos. Se predeclararon pesos tabulares 0, .025, .05, .10 y .20. Se eligió mayor accuracy en tune, desempate por log loss, y sólo entonces se abrió confirm.

Tune eligió .20: +4 aciertos netos (27 fixes, 23 breaks), accuracy .941993→.942583, log loss **empeoró** .148989→.162735. En confirm: 18 fixes/30 breaks, **−12 netos**, accuracy .946298→.944637, log loss .145915→.160247. Seen .965469→.964676; new .902104→.898445. Bootstrap por vendedor sobre neto confirm (2.000 réplicas) IC95 [−25,+1].

**Descartar** la mezcla y la señal de precio relativo como corrector de CatBoost. El corte de ronda 13 es histórico y ha sido examinado; la división interna por vendedor evita reutilizar etiquetas de confirm en la elección del peso pero no crea un confirm totalmente virgen. No habilita submission. Mantener 48 como respaldo público.

Código, pares por ID y métricas: Library `libfile_603c8f8729148191ba0614671ad76983`; SHA256 ZIP `5b5ba404b0960c881b07378fef010e94969a655f51f387cdc57bdfc07090c3f9`.
