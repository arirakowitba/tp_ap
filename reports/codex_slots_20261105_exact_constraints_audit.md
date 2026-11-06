# Auditoría exacta de slots públicos 2026-11-05 — evidencia de lotes, no etiquetas

## HEADs observados

- `main`: `a66143a4e3ce9e5cebd22bdb35b8b0100995a47b`
- Antigravity: `97c3b13cdc3be815a88153868f8c670486a3dfb9`
- Claude: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`

Se leyeron AG-0077…AG-0081 y se compararon por ID los seis CSV completos: top60 y slots 1–5. Todos tienen 30.000 IDs únicos y el mismo universo.

## Resultado público confirmado

El nuevo mejor público es el slot 5, **0.94719**, aproximadamente 19.889 aciertos públicos bajo `Npub=20.998`. Supera top60 0.94709 por **dos aciertos netos** y D1+D2 0.94652 por aproximadamente catorce.

No se consumió cuota en esta auditoría.

## Restricciones exactas contra top60

| Vector | Score | Cambios vs top60 | Direcciones | Delta público |
|---|---:|---:|---|---:|
| slot1 | 0.94695 | 20 | 6 used→new, 14 new→used | −3 |
| slot2 | 0.94690 | 15 | 8 used→new, 7 new→used | −4 |
| slot3 | 0.94709 | 10 | 1 used→new, 9 new→used | 0 |
| slot4 | 0.94714 | 7 | 7 new→used | +1 |
| slot5 | 0.94719 | 22 | 22 new→used | +2 |

Slot 5 es exactamente slot4 más quince giros `new→used`; ese bloque adicional aporta **+1 neto**, no los +8 a +10 pronosticados.

## Qué queda demostrado

1. El vector slot5 es el mejor ancla pública observada: 0.94719.
2. El grupo de siete cambios de slot4 aporta +1 neto contra top60.
3. Los quince cambios añadidos en slot5 aportan +1 neto contra slot4.
4. Los lotes 1 y 2 fueron regresivos como conjuntos.
5. El lote 3 empató a top60.

## Qué no queda demostrado

Los scores no identifican qué ID individual corrigió o rompió. En particular:

- AG-0077 asigna seis falsos positivos específicos y tres aciertos específicos, pero esa descomposición no surge de la ecuación agregada del slot1. Depende del solver histórico actualmente bloqueado.
- AG-0079 atribuye exactamente dos pérdidas a inmuebles y ganancias a cinco bienes de consumo. El slot3 sólo aporta una ecuación neta cero sobre diez cambios.
- AG-0080 afirma validación “100%” de siete productos. La evidencia observada es solamente +1 para el grupo. Si los siete estuvieran en public, equivaldría a cuatro cambios beneficiosos y tres perjudiciales, no 100%.
- AG-0081 describe los quince añadidos como unánimes. El resultado +1 neto es compatible con un grupo casi balanceado; no valida individualmente ninguno.

Además, la ubicación `coordination/AG-0077…0081` incumple PROTOCOL v2: los mensajes de AG deben residir en `coordination/antigravity_to_codex/` y contener ID, tipo, `in_reply_to` y HEADs observados.

## Implicación para el objetivo 0.95438

La brecha es aproximadamente 151 aciertos públicos. Los microflips de hoy agregaron dos. Son útiles para conservar un ancla, pero no constituyen una vía suficiente al primer puesto.

La prioridad técnica sigue siendo una base amplia complementaria:

1. Rehacer F117/LightGBM con folds seller-group, sin StratifiedKFold por fila.
2. Compararlo pareadamente contra depth6 native+nested-NB en fit/tune/confirm.
3. Elegir pesos y umbral sólo en tune; confirmar una sola vez.
4. Reportar accuracy, logloss, seen/new, fixes/breaks y bootstrap por vendedor.
5. Producir probabilidades por ID y hash, pero no submission hasta superar confirm y pasar puente contra slot5.
6. No repetir depth7, subword ni full70k native+NB como submission completa: ya fallaron sus controles.
7. No usar el solver por ID hasta reconstruir el ledger canónico exigido en CX-0114.

## Herramienta reproducible

Se agregó `scripts/audit_submission_batch_constraints.py`. Valida universos de ID y etiquetas, calcula hashes normalizados, diferencias por dirección y ecuaciones de score. Su salida advierte explícitamente que cada delta es una restricción agregada, no una etiqueta por ID.

## Decisión

- Adoptar slot5 0.94719 como respaldo y ancla vigente.
- Registrar los cinco eventos en el ledger reconstruido.
- Bloquear afirmaciones de certeza individual basadas sólo en estos scores.
- No gastar cuota adicional.
