# Ronda 73 — alinear exposición de frecuencias OOF en HistGB

Hipótesis derivada del diagnóstico 69–70: los conteos categóricos de fit OOF se construyen con aproximadamente 2/3 de fit y los de eval con fit completo. Se multiplicó el conteo eval por 2/3 antes de log1p; todos los demás insumos, arquitectura, particiones seller-group y umbral .475 se conservaron. Referencia pareada HistGB+FastText de ronda39. Los cortes ya habían sido usados y se solapan entre sí.

| Corte | Tune neto | Confirm fixes/breaks | Confirm neto | IC vendedor 95% | Log loss confirm base → candidato |
|---|---:|---:|---:|---:|---:|
| a | +1 | 24/30 | −6 | [−20,+8] | .210628 → .210067 |
| b | −22 | 48/77 | −29 | [−55,−6] | .205913 → .205012 |

El log loss mejora en ambos confirm pero accuracy empeora, con evidencia negativa más fuerte en b. No promover a submission ni ajustar un umbral sobre estos confirm reutilizados. El resultado refuerza que mejoras de probabilidad pueden no resolver las decisiones de clase que importan para Kaggle. No se usaron scores públicos como etiquetas.

Checkpoint `round73_hist_frequency_match_checkpoint_20261027.zip`, Library ID `libfile_f78a024fcfb48191bafa8621ed028792`, SHA256 `31173a1be3dedbc85674694d508b9c49379978d53d5bb2eb244d8199cbf6cec3`. Código, pares, bootstrap, métricas y hashes; requiere ronda39 y data.zip.
