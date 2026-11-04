# Auditoría de duplicados y contaminación del ledger AG-0055

**Fecha:** 2026-11-04  
**Ledger auditado:** `results/forensic_ledger_20261104/submission_ledger_master_85vec.json`  
**HEAD AG:** `4c20a6e8bf4159e550ee6394f82b5282e69a66f2`

## Resumen

- Entradas marcadas `verified_recoverable`: **85**
- Vectores SHA únicos entre las entradas con `sha256_vector`: **78**
- Exceso por duplicados exactos: **7 filas**
- Entradas con mapa target `flips_dict`: **75**
- Entradas verificadas sin `flips_dict`: **10**
- Inconsistencias entre `flipped_ids`, claves de `flips_dict` y `n_flips_vs_48` en las 75 entradas mapeadas: **0**

## Grupos de vectores exactamente duplicados

| Vector | Posiciones en las 85 verificadas | Keys | Score |
|---|---|---|---:|
| Sub48 duplicada | 41, 55 | 48; 76 | 0.94633 |
| D2 duplicada | 56, 61 | D2_sonda2_new_virgin16; probe_d2_new16 | 0.94638 |
| D1 duplicada | 57, 62 | D1_sonda1_seen_virgin7; probe_d1_seen7 | 0.94647 |
| havg full duplicada | 58, 63 | havg_plus_full70k; havg_plus_full70k | 0.94471 |
| variant C duplicada | 59, 64 | havg_b10_variant_c_full70k; havg_variant_c_full70k | 0.94514 |
| D1+D2 triplicada | 60, 65, 70 | candidate_d1_d2_combined_on48; candidate_d1_d2_combined_on48; submission_final_reconfirmation_d1_d2 | 0.94652 |

Todos los duplicados tienen scores coincidentes, por lo que son reconfirmaciones o aliases, no nueva información independiente.

## Contaminación del holdout cronológico

`run_adversarial_posterior_audit.py` usa posiciones 1..65 como entrenamiento y 66..85 como test.

El vector D1+D2 aparece:

- posición 60: train;
- posición 65: train;
- posición 70: test.

Por tanto, el test contiene una copia exacta de un vector observado dos veces en train. No es un holdout cronológico estricto ni independiente.

La fila de test `submission_final_reconfirmation_d1_d2` tiene real +4 hits vs Sub48 y predicción +3.9172, error -0.0828. Ese aparente éxito es prácticamente una recuperación de un vector ya visto.

## Implicaciones

1. El tamaño efectivo de evidencia vectorial es 78, no 85.
2. Rango, nulidad y backtests deben calcularse después de deduplicar por `sha256_vector`.
3. Los aliases/reconfirmaciones pueden conservarse para verificar estabilidad del score, pero no deben contar como observaciones independientes.
4. Los splits deben agrupar por hash de vector y por familia/linaje; ningún hash puede cruzar train/test.
5. La cronología debe derivarse de fecha real de envío, no de posición en un JSON ensamblado.
6. Las métricas AG-0055 quedan invalidadas hasta una réplica deduplicada.

## Entradas verificadas sin target map

`A15`, `A11`, `strong5`, `top2`, `final_global_evidence`, `20261104_01_all80`, `20261104_02_used40`, `20261104_03_new40`, `20261104_04_low20`, `20261104_05_low10`.

Sus conjuntos XOR pueden participar en ecuaciones agregadas si fueron reconstruidos contra la referencia exacta, pero no deben usarse para generar CSV/targets sin recuperar los vectores o mapas correspondientes.

## Requisito de corrección

Crear un ledger canónico con una fila por `sha256_vector`, una tabla separada de aliases/envíos, timestamps verificados y grupos indivisibles por vector+linaje. Repetir luego LOFO y chronological holdout.
