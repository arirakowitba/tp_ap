# Auditoría Codex — réplica two-stage y solver 103-vector

Fecha UTC: 2026-11-07
HEADs observados:
- main: db1d3a6c1abae179480059322a2b22b8174821cb
- Antigravity: 14cc054880d649a1215ea7f18aa5e375726d7c6b
- Claude: 38a8cd7abea8e241886fbc30416f3160cd9bca5b

## Hallazgos materiales

### 1. Nuevo mejor público agregado

Antigravity registra evidencia externa informada por Ariel:
- Slot 1: 0.94738.
- Slot 4: 0.94752, nuevo mejor público demostrado.
- El salto Slot 1 -> Slot 4 equivale a +3 aciertos públicos agregados bajo Npub=20.998.

Este resultado sólo valida el vector completo de Slot 4. No identifica cuáles de sus 16 cambios adicionales fueron correctos ni autoriza efectos por ID.

### 2. La implementación corregida existe, pero las métricas no provienen de ella

El HEAD actual de `scripts/replicate_frozen_baseline_inner_early_stopping_20261106.py` sí contiene la arquitectura requerida:
- inner seller split sólo para elegir iteraciones;
- nested-NB OOF seller-group sobre 100% del outer fit;
- refit final en 100% de outer fit sin eval_set;
- blend y umbral congelados.

Sin embargo, la cronología invalida la atribución de los resultados actuales a ese código:
- `28a7fb1` commiteó los resultados a 01:14:34 UTC;
- `0184622` implementó el runner two-stage a 01:18:14 UTC.

Además, los `report.json` publicados carecen de los campos de provenance generados por el runner nuevo (`method: two_stage_refit`, `stage1_inner_train`, `stage1_frozen_iterations`) y conservan el esquema anterior (`inner_train`, `training_metadata`). Por lo tanto:
- mixed +4 y pure +12 del blend 50/50 son resultados del runner anterior;
- no cuentan como réplica honesta del baseline corregido;
- no deben promover modelos ni construir test predictions.

### 3. Solver 103-vector no identifica efectos individuales

El solver declara:
- 87 configuraciones únicas;
- rango 82;
- 2.578 IDs intervenidos.

Por construcción quedan al menos 2.496 direcciones no identificadas. El ridge con prior elige una solución entre infinitas, pero no convierte los efectos por ID en evidencia observable. Problemas adicionales:
- no reporta incertidumbre posterior ni sensibilidad a lambda/prior;
- usa una matriz binaria de pertenencia al flip-set, sin codificar explícitamente la dirección del cambio;
- redondea scores públicos a hits y los trata como exactos;
- las guardas semánticas sólo filtran candidatos, no resuelven identificabilidad;
- llamar “MAP/Bayesian” a bounded ridge no aporta calibración probabilística.

Consecuencia: bloquear los 213 “safe positive flips” y las listas top-N para promoción o nuevas submissions. Slot 4 queda como ancla por evidencia pública del vector completo, no por la deconvolución.

## Evaluación científica de las métricas antiguas

Aun tomadas sólo como desarrollo:
- mixed 50/50 vs nested: tune +20, confirm +4, IC confirm [-28,+32];
- pure 50/50 vs nested: tune +28, confirm +12, IC confirm [-21,+44];
- mixed 80/20 CB+LGB: tune +4, confirm +14;
- pure 80/20 CB+LGB: tune +27, confirm -2.

No hay mejora estable con IC confirm positivo contra nested. El blend 50/50 conserva interés como receta congelada, pero exige reejecución real con el runner posterior a `0184622`.

## Próxima ejecución obligatoria

1. Reejecutar mixed seed 20261107 y pure seed 20261108 desde HEAD >= `0184622`, en directorios nuevos; no sobrescribir resultados viejos.
2. Guardar commit del código exacto antes de lanzar y registrar `git rev-parse HEAD` dentro de cada report.
3. Exigir campos `method=two_stage_refit`, iteraciones congeladas y hashes de fit/tune/confirm/pares.
4. Comparar tune y confirm pareados contra nested y contra el 50/50 histórico; bootstrap por vendedor.
5. No generar test/submission desde estas réplicas hasta RESULT revisado.
6. Para aprovechar scores históricos, limitarse a efectos identificables por contrastes de conjuntos cuya diferencia esté determinada; calcular nullspace/sensibilidad. No publicar probabilidades por ID desde el sistema rank-deficiente.

No se consumió cuota Kaggle en esta auditoría.
