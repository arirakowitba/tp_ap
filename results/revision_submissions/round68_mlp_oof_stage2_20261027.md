# Ronda 68 — MLP con insumos pareados de segunda etapa

Hipótesis: MLP (96,48) con transformación cuantílica y regularización sobre los mismos insumos de HistGB+FastText (tabular, embeddings40, word/char OOF) aprende interacciones complementarias. Entrenamiento solo fit; tune/confirm seller-group de los dos cortes históricos de ronda39.

| Corte | Tune neto | Confirm fixes/breaks | Confirm neto | Log loss confirm base → MLP |
|---|---:|---:|---:|---:|
| a | −3250 | 753/3842 | −3089 | .210628 → 1.023011 |
| b | −2818 | 725/3805 | −3080 | .205913 → .756301 |

Fracaso grande y repetido. La causa exacta no está determinada; revisar desplazamiento de distribución entre insumos OOF de fit y holdout antes de interpretar como límite de todas las redes. Esta configuración se descarta. No combinar, calibrar ni presentar como candidata. Cortes reutilizados; ninguna etiqueta de test ni score público fue usado en entrenamiento.

Checkpoint `round68_mlp_oof_checkpoint_20261027.zip`, Library ID `libfile_7371f40cf7348191af3fefcbf6555e60`, SHA256 `48154834490c8f743031753a7c0d6b60878bf5477e323fcb1c2b3851ba289dca`. Código, pares, métricas y hashes; requiere ronda39 y data.zip.
