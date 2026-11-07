# Auditoría Slot 1 = 0.94738, bloqueo Slot 2 y revisión del runner de replicación

Fecha: 2026-11-06  
Main observado: `0a9536d458e4931bc6e8755aa7884c570f41c771`  
AG observado: `74c8dd6c0b6ade758e4a1e33febd08ceca34050b`  
Claude observado: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`

## Evidencia pública nueva

AG registró feedback externo para:

- archivo: `submission_20261107_01_purified_probe_top25.csv`
- score: 0.94738
- ancla previa: 0.94719
- Npub estimado: 20.998
- hits ancla: round(0.94719 × 20.998) = 19.889
- hits Slot 1: round(0.94738 × 20.998) = 19.893
- delta agregado: **+4 hits**

La comparación exacta de vectores confirma 25 cambios, todos `new -> used`, contra el ancla 0.94719.

IDs del lote:
`2249, 3647, 4617, 5015, 5390, 5455, 5993, 6659, 12228, 13224, 13628, 14035, 15275, 15939, 18316, 18549, 20200, 20544, 21050, 22138, 24240, 25753, 27013, 29552, 29663`.

## Qué permite inferir

El lote completo tiene efecto público agregado +4. Esto valida 0.94738 como mejor ancla pública reportada.

No permite inferir:

- qué IDs pertenecen al public split;
- qué IDs individuales fueron correctos;
- una probabilidad 0.9473 por item;
- que los próximos 20 flips mantendrán la misma pendiente;
- “cero trampas” o “pureza” predictiva.

La pertenencia pública de cada ID es desconocida. Por lo tanto, ni siquiera el denominador efectivo de flips observados está identificado.

## Slot 2 bloqueado

El archivo Slot 2 agrega 20 cambios disjuntos:

`815, 6308, 6900, 7936, 8674, 10013, 16190, 16515, 17117, 17645, 22237, 22499, 22905, 22953, 24735, 26975, 27016, 27780, 28431, 29859`.

Estos IDs fueron seleccionados mediante inspección semántica y el posterior/fusión previamente bloqueado. No existe backtest prequential ni comparación train-only que justifique la proyección de +10 hits. La generación contradice el estado `STATUS_BLOCKED.md` y la suspensión de cuota.

No subir Slot 2. Mantener 0.94738 como fallback provisional hasta confirmación directa de Ariel.

## Auditoría del runner de replicación

`scripts/replicate_frozen_baseline_inner_early_stopping_20261106.py` todavía no fue ejecutado, pero no replica la receta congelada:

1. divide el outer fit en `fi_tr`/ `fi_va`;
2. usa `fi_va` para early stopping;
3. conserva el modelo entrenado sólo con `fi_tr`;
4. nunca refitea sobre el outer fit completo con las iteraciones congeladas.

Así, cada componente usa aproximadamente 85% del fit, mientras la receta original usa el fit completo. Cualquier delta mezclaría efecto de split, menor entrenamiento y cambio de early stopping.

### Corrección requerida

Por cada modelo:

1. usar `fi_tr/fi_va` sólo para seleccionar `best_iteration`;
2. congelar ese número;
3. reconstruir la feature nested-NB OOF para todo `fi` con GroupKFold por vendedor;
4. ajustar el modelo final sobre todo `fi`, sin eval_set y con iteraciones congeladas;
5. proyectar tune/confirm una vez;
6. aplicar blend 0.50/0.50 y umbral 0.50 sin cambios.

Sólo entonces ejecutar seeds nuevas mixed/pure.

## Estado

- mejor público reportado: 0.94738;
- mejora sobre 0.94719: +4 hits;
- evidencia individual por ID: ninguna;
- Slot 2: bloqueado;
- replicación baseline: pendiente de corrección;
- cuota consumida por Codex: cero.
