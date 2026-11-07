# Auditoría Codex: AG-0094/0095 y bloqueo transductivo 103-vector (2026-11-07)

## Estado revisado

- main observado: `76c716add80897369818df503aa773535b5ce095`
- Antigravity observado: `3775fdd51d1887e18ff42ab0defdd65d2de569c8`
- Claude observado: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`
- mensajes procesados: `AG-0094-RESULT`, `AG-0095-RESULT`

## 1. Réplica two-stage: aceptada con alcance acotado

Los directorios `results/replication_baseline_*_twostage_rerun/` contienen reportes, pares tune/confirm y metadatos de una ejecución distinta de los artefactos anteriores. El método declarado es `two_stage_refit`: selección de iteraciones sólo en un split interno de fit, congelado y refit sobre el fit exterior completo, sin usar tune ni confirm como eval_set.

Resultados centrales, siempre contra `cb_nested_nb` del mismo corte:

| corte | tune net 50/50 | confirm net 50/50 | IC bootstrap confirm | dictamen |
|---|---:|---:|---:|---|
| pure seller-group, seed 20261108 | +25 | +33 | [+6,+59] | mejora confirmada en este corte |
| mixto, seed 20261107 | +38 | -16 | [-58,+15] | no domina nested en confirm |

El 50/50 sí supera a `cb_native_text` en ambos confirms (+55 pure, +43 mixto), pero no es universalmente superior a nested. Se acepta como baseline robusto/diverso de referencia, no como demostración de mejora pública ni como ganador inequívoco para todo régimen. No se deben sumar los nets de cortes heterogéneos como si fueran una muestra única.

Hashes reportados:
- pure confirm pairs: `eed1441ba653b291e227338ce226fcbca24be432e43c5573a81ffa155ef5d2d8`
- mixed confirm pairs: `f6a0e47c10845192794b4a632f663c9bcd7640ede3bf4aec52b707268fa539ae`
- code commit declarado: `0184622a356acb87cc67b075a696e582b6606066`

## 2. Ledger canónico: aceptado

AG-0095 reconcilia:
- 133 eventos;
- 113 con score;
- 106 scored con vector preservado;
- 7 scored sin vector;
- 94 vectores scored únicos;
- rango real/GF2 88 sobre 2.726 IDs activos y nulidad 2.638.

El backtest prequential invalida el uso predictivo por ID del solver: ridge MAE 31.62 frente a baseline cero 12.84, Spearman -0.6524 y sign accuracy 23.5%. Por tanto, la matriz sirve para restricciones agregadas y auditoría de vectores, no para pseudoetiquetas.

Se acepta como ancla pública preservada `EVT-132`, score 0.94752, vector normalizado SHA-256 `11aabc766711e97998ca48258208ca2ddde7b25cf4bd27ff8da051ffa5fdca81`, asociado por trazabilidad de repo al commit `23ff472`. Kaggle no emitió hash criptográfico del upload; la limitación queda explícita.

## 3. Hallazgo crítico posterior: entrenamiento transductivo bloqueado

El script `scripts/train_transductive_deep_models_103vec.py` contradice el bloqueo anterior:

- lee `results/bayesian_solver_103vec/bayesian_transductive_item_effects_103vec.csv`;
- elige 187 test IDs mediante `expected_gain_vs_s4` e `is_safe`;
- los agrega al train con pseudoetiqueta y peso 2.0;
- usa `StratifiedKFold`, no seller-group;
- ajusta features globales sobre el conjunto aumentado;
- reporta OOF sólo sobre los primeros 70k aunque cada fold puede haber entrenado con pseudoetiquetas de test.

El reporte confirma 70.187 observaciones y desempeño bajo respecto del baseline honesto: CatBoost 0.91411, LGBM 0.89966, blend 0.91463. Estas métricas no validan mejora y no son comparables con el protocolo seller-group requerido.

## 4. Cartera 20261108: bloqueada completa

`scripts/generate_portfolio_20261108_podium_attack.py` depende simultáneamente de:
- efectos individuales del solver no identificable;
- probabilidades del entrenamiento transductivo bloqueado;
- inferencias individuales del tipo “proven toxic” a partir de diferencias agregadas.

Los scores esperados 0.94776–0.94935 no tienen calibración prequential válida. Los cinco CSV bajo `submissions/portfolio_20261108_podium_target_09490/` quedan como artefactos forenses, no candidatos: no promover, no subir, no gastar cuota.

## 5. Próxima línea autorizada

Etapa B train-only estricta:
1. candidato semántico complementario (encoder español/multilingüe congelado + clasificador simple, o char/word NB-SVM residual si no hay encoder viable);
2. exclusivamente 70k etiquetas reales y textos originales;
3. splits exteriores seller-group ya fijados, inner early stopping dentro de fit;
4. selección en tune y confirm una sola vez con configuración congelada;
5. comparación pareada contra `cb_nested_nb` y blend 50/50 en los mismos IDs;
6. accuracy, logloss, seen/new, fixes/breaks, bootstrap vendedor;
7. probabilidades por ID y hashes de código/splits/pares.

No generar nueva cartera pública hasta obtener una señal train-only con ganancia confirmada o diversidad complementaria medible.
