# Dos cortes testlike adicionales para CatBoost profundo + texto OOF — 2026-11-02

Código: `scripts/train_deep_text_20261101.py` en main. Comando para cada semilla: `AP_ZIP=/ruta/ap-2026-q-2.zip python scripts/train_deep_text_20261101.py SEMILLA testlike`. ZIP original train70k SHA256 `3e0b032195766e681a411a1d1ea06fe906940dd025a7317adfc5302791bbe886`. Polaridad used=1. Texto original título/warranty/value_name. Transformador TF-IDF se ajusta dentro de cada fold de fit; tune y confirm se transforman sin sus etiquetas. Agregados categóricos y relativos se ajustan en fit. CatBoost early stopping y umbral en tune; confirm sólo medición.

| Semilla | fit/tune/confirm | seen confirm | base acc/LL | texto acc/LL | net | bootstrap vendedor 95% | seen/new net | hash pares |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 20261111 | 45340/12517/12143 | 9000 | .951083/.129350 | .953883/.122030 | +34 | [-2,70] | +19/+15 | `324c32ece883d47c8fdc6f0bfcfe3487790f64999161f53715ce90b685f95c36` |
| 20261112 | 45196/12307/12497 | 9000 | .944387/.140344 | .948308/.134810 | +49 | [13.975,85] | +28/+21 | `73a5698a8a920ea52bf5ef9db800e4e5c2cc3b245efae7641b70d2567adf8dcc` |

El corte 20261109 previo dio +55 con IC [13,101]. Intersecciones de IDs de confirm: 09/11=2669, 09/12=2671, 11/12=2649, aproximadamente 12% Jaccard. Son repeticiones de desarrollo, no tres confirmaciones independientes ni un holdout virgen. Los cortes seller-group nuevos son disjuntos de fit dentro de cada semilla, mientras los vistos tienen al menos un registro en fit. El 74/72% visto imita la mezcla observada de test, pero la generalización pública no queda demostrada. Este comparador CatBoost fuerte no reconstruye exactamente D1+D2; el candidato completo de 660 cambios **no está promocionado** tras el fracaso previo de OptionA completo .93695. No usar estas métricas para prometer un score .947.

Pares y reportes crudos: Library `libfile_b4b4cde5bc548191aad2f95af874f4e5`, ZIP SHA256 `1457e58369f67a68825223e1f3f419de0d2e5b0177e157cf3d0fed75753e4e77`. Próximo: ablation contra clon D1+D2 en las mismas filas, calibración y segmentos antes de sonda pública pequeña y preespecificada.