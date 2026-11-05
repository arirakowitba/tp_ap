# Auditoría de reconciliación del ledger de submissions — 2026-11-05

## Alcance y HEADs observados

- `main`: `84942e53d7804f2701871407002bcc75f766fe20`
- `antigravity/overnight-recovery-coordination-20261030`: `efca08d60c5c782a8bd85ae50e3e22fbc6299402`
- `claude/posterior-y-modelos-20261027`: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`

Fuentes auditadas en la rama de Antigravity:

- `results/forensic_ledger_20261104/reconciliation_110_submissions_report.md`
- `results/forensic_ledger_20261104/reconciliation_110_submissions_table.json`
- `results/forensic_ledger_20261104/submission_ledger_master_85vec.json`
- reportes y CSV canónicos asociados.

No se consumió cuota Kaggle.

## Hallazgo principal

Los artefactos actuales no constituyen un ledger canónico consistente. Antes de ajustar posteriores por ID o diseñar probes debe reconstruirse el registro de eventos y recalcular hashes normalizados de vectores.

| Capa observada | Conteo |
|---|---:|
| Filas del catálogo de reconciliación | 117 |
| Filas declaradas como drafts locales | 19 |
| Envíos nominalmente submitted/scored | 98 |
| Marcados category1, con vector | 86 |
| Marcados category2, “CSV missing” | 12 |
| Aliases recuperables dentro de esos 12 | 8 |
| Envíos scored realmente sin vector recuperable | 4 |

Los ocho aliases recuperables son:

| Entrada category2 | Vector ya disponible como |
|---|---|
| 04 | 4 |
| 05 | 5 |
| 06 | 6 |
| 07 | 7 |
| 59_new | probe_59_new_seller |
| 59_seen_new | probe_59_seen_to_new |
| probe_fin_stack_new150 | fin_stack_new150 |
| probe_fin_stack_seen150 | fin_stack_seen150 |

Por lo tanto, los únicos cuatro scored que siguen realmente sin vector son `01`, `02`, `03` y `10`. El resumen que afirma doce vectores faltantes contradice el propio detalle, que sólo enumera estos cuatro.

Además, los identificadores `53`, `54` y `55` aparecen duplicados: una vez como envíos públicos y otra como drafts locales. El número visible no puede usarse como clave primaria.

## Inconsistencias del maestro “85vec”

El archivo `submission_ledger_master_85vec.json` declara simultáneamente:

- `n_entries_total = 93`
- `n_recoverable_vectors = 90`
- `n_unresolved_entries = 3`
- `matrix_shape = [85, 2572]`

La inspección efectiva da:

- 93 objetos en `entries`;
- 80 con `sha256_vector`;
- 73 hashes de vector distintos entre esos 80;
- 13 objetos sin `sha256_vector`.

Entre los trece sin hash están 53/54/55, cinco envíos de 2026-11-04 y cinco de 2026-11-05. Esto es principalmente deuda de reconstrucción: 53/54/55 y los probes del 04 tienen `flipped_ids`; los cinco del 05 poseen CSV en el repositorio. No deben contarse como desconocidos sin intentar reconstrucción determinista.

El CSV canónico existente representa una fotografía anterior: 85 filas se reducen a 78 vectores únicos y 7 aliases. No representa el maestro actual de 93 entradas ni el catálogo nominal de 98 scored submissions.

## Reconciliación con el historial externo

Con los archivos actuales sólo pueden demostrarse 98 eventos nominalmente scored, no los 106 reportados por Ariel. La diferencia de ocho no se resuelve desde el repositorio. Se necesita exportación o transcripción autoritativa del historial Kaggle para certificar el universo de eventos. El inventario de 111 archivos CSV no sirve como sustituto: mezcla drafts, candidatos, aliases y duplicados.

## Consecuencia estadística

Queda bloqueado promover cualquier posterior individual por ID construido sobre estos conteos. Los scores son ecuaciones agregadas; sin una matriz canónica de vectores y eventos únicos no hay identificabilidad fiable.

La auditoría canónica previa ya muestra el límite práctico del enfoque actual: 78 vectores únicos, rango 72, nulidad 2500 y backtest cronológico peor que el predictor cero (MAE 4.59 contra 2.55, correlaciones de rango negativas y signo balanceado 50%). Esto no impide usar señales agregadas, pero sí invalida presentar el posterior como probabilidad calibrada por etiqueta.

## Reconstrucción exigida

1. Crear un `event_id` inmutable y único, separado del número/nombre visible.
2. Partir del historial Kaggle autoritativo; no inferir eventos desde el inventario de CSV.
3. Normalizar cada vector por `item_id -> target` y hashearlo; el hash de bytes del archivo no basta.
4. Resolver los ocho aliases anteriores sin duplicar ecuaciones.
5. Reconstruir 53/54/55 y los cinco probes del 04 desde `flipped_ids` y su base declarada.
6. Hashear los cinco CSV del 05 y todos los posteriores.
7. Mantener categorías mutuamente excluyentes: scored con vector, scored sin vector, error/no score, draft.
8. Emitir conteos verificables, hashes, aliases, matriz, rango y condición.
9. Reejecutar backtest prequential por cortes temporales no solapados antes de usar el solver.
10. No convertir scores públicos agregados en etiquetas individuales ni probabilidades calibradas sin evidencia de identificabilidad.

## Decisión operativa

- Mantener suspendidos los probes Kaggle.
- No usar el maestro actual para seleccionar flips.
- Priorizar modelos train-only amplios y validación seller-group mientras se corrige el ledger.
- Conservar el mejor score público demostrado como ancla; ninguna inferencia de este ledger constituye mejora predictiva.
