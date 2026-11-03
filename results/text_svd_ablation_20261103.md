# Ablación SVD de texto (2026-11-03)

Se agregaron 32 componentes SVD del texto de palabras al CatBoost histórico de word+char TF-IDF OOF y seller categórico. El vectorizador y SVD se ajustaron sólo en fit. El umbral se eligió en tune; confirm no participó de la selección. Los pares se unieron por item_id con la base histórica, verificando 0 discrepancias de seller y y_true. Código: `scripts/ablate_text_svd_vs_historical_20261103.py`. ZIP original SHA-256 `3e0b032195766e681a411a1d1ea06fe906940dd025a7317adfc5302791bbe886`.

| Seed seller-group | Confirm | Base fuerte | +SVD | Saldo | Fixes / breaks | Logloss base → SVD | IC95 vendedor |
|---|---:|---:|---:|---:|---:|---|---|
| 20261109 | 12.357 | .950716 | .950635 | −1 | 68 / 69 | .137638 → .139960 | [−24,+22] |
| 20261116 | 13.553 | .951966 | .950491 | −20 | 64 / 84 | .126194 → .128820 | [−49,+9] |

El SVD explica aproximadamente 7,25% de la varianza de la matriz word TF-IDF con 32 componentes. Perdió logloss y accuracy en ambos cortes, incluido el umbral común .5 (−17/−19 aciertos). Se descarta como mejora de base y no se propone CSV público. Los cortes confirm se habían reutilizado en desarrollo y las agregaciones no supervisadas del extractor base se computan antes del split; ambos brazos comparten esta limitación.

Hashes confirm-ID: `997893645e14d1c09c8ff0e3ceb3776f13688e21901888bf3196def555df3450`, `6983106dd9194284298816b6b6362f8694a76655bca7f11e189deb0e1f7898c1`. Hashes pares: `6178c4db22b4b916319d1da815dd787575d74842a7c771ed1dfa087d52658139`, `0cde70f5a05c08601ee2f925c659e273ad527172b7325f5d2e82f4451b092989`.

Línea más prometedora entre experimentos propios existentes: value_name OOF adicional al histórico fuerte obtuvo +24/+10/+9 en seeds 20261109/15/16 y mejor logloss en los tres, ver `results/attr_ablation/report_*_seller.json`. Falta probar su combinación con otros cambios en validación estricta e inferencia full70k con procedencia.
