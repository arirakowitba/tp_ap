# Validación predeclarada de value_name, seed 20261123

Plan fijado antes de ejecutar: `results/value_name_fresh_seed_20261103_plan.md`. Se ejecutó sin modificar `scripts/ablate_attr_value_vs_historical_text.py 20261123 seller` sobre los 70.000 train originales, ZIP SHA-256 `3e0b032195766e681a411a1d1ea06fe906940dd025a7317adfc5302791bbe886`. Fit/tune/confirm seller-group: 45.960 / 11.768 / 12.272. Base histórica word+char TF-IDF OOF + CatBoost seller; candidata agrega un clasificador OOF de attribute.value_name. Iteraciones y umbrales seleccionados en tune.

| Métrica confirm | Base fuerte | +value_name |
|---|---:|---:|
| Accuracy | .950782 (11.668/12.272) | .950538 (11.665/12.272) |
| Logloss | .128972 | .129982 |
| Umbral tune | .465 | .485 |
| Saldo pareado | — | **−3** (56 fixes, 59 breaks) |
| Net seller nuevo/visto | — | −6 / +3 |
| IC95 bootstrap vendedor | — | [−28,+19] |

A umbral común .5 el candidato gana 2, pero la regla predeclarada de selección por tune perdió 3. El tune favoreció a la candidata (.954878 vs .954028), señal de que se debe evitar escoger sólo por tune en una línea explorada repetidamente.

El nuevo confirm comparte 2.726/2.619/2.997 IDs con los confirm previos 20261109/15/16, por lo que no constituye validación externa independiente completa. Antes estos tres cortes habían dado +24/+10/+9 y menor logloss; la nueva réplica **no confirma** una mejora estable. No promover a Kaggle ni recomendar full70k basado sólo en los tres positivos.

Hashes: confirm-ID `cbff32c9df4cc3652ae03c3d56b005cd8540ea9a05cda58349ae169a7f31e395`; CSV pareado `c6b9caa0f414f51904905989c98d29015bd3416815107027cab8e3b793164250`. El CSV pareado con y_true se conserva fuera del repo; aquí sólo métricas agregadas. Persisten agregados no supervisados del extractor base calculados antes del split y confirm reutilizados en la historia de desarrollo.
