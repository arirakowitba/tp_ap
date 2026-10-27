# Ronda 74 — términos seleccionados para interacciones textuales

Hipótesis: el HistGB de segunda etapa solo recibe los scores word/char OOF, así que se agregaron 96 indicadores de presencia de unigramas/bigramas seleccionados mediante chi² **solo en fit**. Se conservaron todos los insumos originales, incluidos FastText y los predictores OOF; mismo modelo y umbral .475. La arquitectura se probó en dos cortes seller-group a/b; luego se congeló para el tercero c, previamente conservado. Los tres cortes ya habían sido usados históricamente y se solapan.

| Corte | Tune neto | Confirm fixes/breaks | Confirm neto | IC 95% vendedor | Log loss confirm base → candidato |
|---|---:|---:|---:|---:|---:|
| a | +5 | 67/61 | +6 | [−17,+29] | .210628 → .209409 |
| b | +8 | 65/62 | +3 | [−20,+27] | .205913 → .205890 |
| c, configuración congelada | −1 | 55/74 | −19 | [−43,+5] | .215678 → .216041 |

Los dos primeros confirm sugirieron una mejora débil; el tercero la contradice. En a/b la ganancia de cambios hacia new fue +8/+10, pero en c fue −13. No seleccionar términos, pesos ni direcciones después de ver c; descartar esta configuración para submission. La evidencia subraya el riesgo de promover ganancias pequeñas en cortes solapados. No se usaron scores públicos como etiquetas.

Checkpoint `round74_selected_token_interactions_checkpoint_20261027.zip`, Library ID `libfile_c1c0f612809481918119e9da4b08a6d1`, SHA256 `ff7e1c33f9337b126c9baa986c2f25cfc99a89478b899cb3c2efa26b32e3c9c2`. Contiene código, listas de términos, pares, métricas, bootstrap y hashes. Requiere ronda39, ronda50 y data.zip.
