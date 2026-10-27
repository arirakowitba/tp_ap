# Ronda 67 — ejes rotados de FastText

Hipótesis: agregar una rotación ortogonal fija y sin etiquetas de los 40 embeddings FastText permite a los árboles de segunda etapa acceder a combinaciones semánticas que no captan con divisiones por eje. Se conservaron todas las variables tabulares, embeddings originales y predictores textuales OOF del HistGB+FastText de referencia. Se evaluaron dos cortes seller-group de ronda 39 (reutilizados, solapados entre sí); umbral .475.

| Corte | Tune neto | Confirm fixes/breaks | Confirm neto | Log loss confirm base → candidato |
|---|---:|---:|---:|---:|
| a | −18 | 68/73 | −5 | .210628 → .210821 |
| b | +2 | 70/90 | −20 | .205913 → .206673 |

La rotación empeora accuracy y log loss en ambos confirm. Descartar, sin submission. La mejor pública observada sigue siendo 48=.94633. Los scores públicos no se usaron como etiquetas.

Checkpoint `round67_rotated_embedding_axes_checkpoint_20261027.zip`, Library ID `libfile_e4b1e541754081919d5fb8b8850c52b4`, SHA256 `9df81502b749d620b3fec320d2a183998ab942c85e0ed358c0b40ce8c1594554`. Incluye código, pares, métricas y hashes; requiere ronda39 y data.zip.
