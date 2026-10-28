# Ronda 71 — ablación de variables desplazadas en MLP

Tras detectar el salto OOF/full-fit de las frecuencias categóricas en rondas69–70, se entrenó la misma MLP sin las 11 columnas de frecuencia ni `title_digits` (también desplazada). Se conservaron los otros insumos, incluido texto word/char OOF y FastText; misma referencia HistGB+FastText y mismos cortes seller-group históricos.

| Corte | Tune neto | Confirm fixes/breaks | Confirm neto | Log loss confirm candidato vs referencia |
|---|---:|---:|---:|---:|
| a | −299 | 240/490 | −250 | .258450 vs .210628 |
| b | −308 | 230/591 | −361 | .258518 vs .205913 |

Quitar las columnas evita el fallo extremo de ronda68, pero resulta peor que la corrección de exposición de ronda70 y sigue muy por debajo de HistGB. Cerrar esta línea MLP; nuevas iteraciones deben usar otra hipótesis y no ajustar más este mismo confirm reutilizado. No inferir etiquetas de scores públicos ni preparar submission.

Checkpoint `round71_mlp_drop_shifted_checkpoint_20261027.zip`, Library ID `libfile_b02aed51d9348191b89ca4b16cbad4bf`, SHA256 `887626287bdffc1a78146e53904a406c9c195566eb3aa600f7d0dd4f14dc8bb2`. Código, predicciones pareadas, métricas y hashes. Requiere ronda39 y data.zip.
