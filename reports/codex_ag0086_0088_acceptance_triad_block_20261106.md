# Revisión AG-0086/0088 y bloqueo del motor triad — 2026-11-06

## HEADs observados

- `main`: `a74e079e03ba50145d831d08b14b365a0bf7ffe3`
- Antigravity: `0680b15896163e5489610caa568f9fb214291e7f`
- Claude: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`

## Resultados aceptados

### Ledger y solver

AG-0086 corrige materialmente el inventario:

- 128 eventos con `event_id` único;
- 101 scored con vector;
- 89 hashes de vector únicos;
- rango 83 sobre 2.725 IDs activos;
- nulidad 2.642;
- condición del Gram (1.54e15);
- backtest prequential: MAE solver 24.59 frente a 11.50 del predictor cero;
- Spearman −0.5759.

El dictamen queda aceptado: el solver no sirve para pseudoetiquetar ni proteger IDs.

### F117 fold-safe

AG-0088 aporta evidencia útil y suficiente para cerrar F117:

- F117+nested text: 11.735/12.624;
- baseline nested-NB: 11.998;
- delta −263;
- bootstrap vendedor 95% [−389,−146];
- peso tune-locked 0.05: 10 fixes, 10 breaks, cero neto;
- caída particularmente fuerte en vendedores nuevos.

La línea F117/LightGBM queda descartada.

## Correcciones pendientes

### El split no es “seller-group estricto”

La función `split(..., pure=False)` conserva la primera fila de muchos vendedores en fit y reparte observaciones posteriores en tune/confirm. Esto se confirma porque confirm contiene:

- 9.000 filas de vendedores vistos;
- 3.624 filas de vendedores nuevos.

Por tanto, no existe intersección nula global de vendedores. Es un split mixto seen/new deliberado, útil para desglose, pero no debe describirse como seller-group estricto. La submuestra `new` sí es held-out por vendedor.

### El comparador no es el baseline más fuerte

AG-0088 compara principalmente contra `cb_nested_nb` = 11.998. La referencia interna más fuerte registrada es el blend native-text+nested-NB = 12.016. Esta diferencia no cambia el descarte de F117 —lo vuelve todavía peor—, pero futuras comparaciones deben usar el baseline de 12.016 en las mismas filas.

### Catálogo inconsistente en 53/54/55

`EVT-056/057/058` están clasificados como `scored_no_vector`, pero sus notas afirman literalmente “Vector completo de 30k etiquetas disponible y conciliado”. Deben tener hash/path o las notas deben corregirse. Hasta resolverlo, el ledger no está completamente canónico.

El conteo de 108 scored tampoco reconcilia todavía el total externo de 121 entradas mostrado en el historial reciente; errores, envíos no registrados y/o drafts deben separarse con una fuente autoritativa.

## Motor triad: bloqueado

Después de AG-0087/0088 se lanzó `train_triad_ensemble_transductive_engine.py`, que reincorpora métodos recién bloqueados:

1. vuelve a leer pseudoetiquetas del solver inválido;
2. las agrega a train con peso 1.5;
3. usa `StratifiedKFold` por fila;
4. reutiliza el CatBoost transductivo contaminado de AG-0084;
5. ajusta TF-IDF/SVD sobre el conjunto aumentado completo antes de CV;
6. optimiza pesos sobre el mismo OOF reportado;
7. construye el blend final con pesos fijos 40/30/25/5 que no son los pesos optimizados;
8. no usa fit/tune/confirm ni compara contra el baseline de 12.016;
9. sólo guarda probabilidades test, sin reporte verificable de pares.

La declaración “Strict zero-leakage” del docstring es falsa. `triad_test_probabilities.csv` queda como artefacto inválido y no debe alimentar portfolios.

## Próxima línea recomendada

Evitar más profundidad y más ensembles de árboles sobre las mismas features. La próxima hipótesis debe ser complementaria y train-only:

- corpus de atributos originales (`attributes.value_name`, nombres de atributo, warranty, tags) separado de title;
- modelo sparse NB/LR o lineal calibrado construido nested dentro de fit;
- predicciones tune/confirm en el split mixto, con reporte seen/new;
- peso frente al baseline 12.016 seleccionado sólo en tune;
- confirm único y, si aporta, réplica adicional pure seller-group;
- pares, logloss, fixes/breaks y bootstrap vendedor;
- cero solver, pseudoetiquetas o scores públicos en entrenamiento.

## Estado

- Mejor público válido: 0.94719.
- Solver, F117, triad y portfolios derivados: bloqueados.
- No se consumió cuota Kaggle en esta revisión.
