# Rondas 69–70 — diagnóstico y corrección parcial de MLP

La ronda68 MLP con los 145 insumos pareados perdió más de 3000 aciertos por confirm. Diagnóstico del corte a: probabilidad media MLP .172 vs prevalencia real .458 y referencia .463. Se auditó la transformación cuantílica train/eval. Los conteos categóricos OOF de fit ven aproximadamente dos tercios de fit; los de eval se calculan con fit completo. En varios índices (listing type, ciudad, estado, shipping, buying mode, moneda) la mediana de eval transformada llegó a +5, mientras la mediana de fit estaba cerca de cero. Esto explica una falla de transferencia de escala, aunque no demuestra ser la única causa.

Ronda70 corrigió **solo** los conteos eval al tamaño esperado OOF, multiplicando el conteo previo a log1p por 2/3. Se mantuvieron arquitectura, insumos restantes, particiones y entrenamiento.

| Corte | Confirm neto MLP original | Confirm neto corregido | Log loss corregido vs HistGB+FastText |
|---|---:|---:|---:|
| a | −3089 | −163 | .241599 vs .210628 |
| b | −3080 | −231 | .243371 vs .205913 |

La corrección rescata la mayor parte de la caída, pero todavía pierde claramente accuracy y log loss. No promover. El diagnóstico revela una limitación de las redes sobre representaciones OOF/full-fit con escalas distintas; un próximo modelo tendría que alinear todas las columnas, entrenar con una simulación de full-fit OOF o usar insumos invariantes. Los cortes son históricos y solapados; no son confirmación nueva. No se usaron etiquetas de test ni scores públicos por fila.

Checkpoint `round69_70_mlp_shift_correction_checkpoint_20261027.zip`, Library ID `libfile_bcdefe4da2bc8191bbadb7f61a24b656`, SHA256 `d82ccd270787f31b2fce874c9c7f2101167ba619370173dd8cc956495bd5e05b`. Contiene script de auditoría, desplazamientos, script corregido, pares, métricas y manifiesto. Requiere ronda39 y data.zip.
