# Umbral por vendedor, baseline histórico fuerte (2026-11-02)

Experimento train-only con 70.000 etiquetas originales y texto word(1–2)+char(3–5) OOF por seller en fit, CatBoost con seller categórico. Código: `scripts/segment_threshold_historical_20261102.py`. ZIP SHA-256: `3e0b032195766e681a411a1d1ea06fe906940dd025a7317adfc5302791bbe886`. Los umbrales global y vistos/nuevos se eligieron únicamente en tune, con grilla 0.400–0.600 paso 0.005, antes de evaluar confirm.

| Seed | Confirm | Umbral global / nuevo / visto | Aciertos global | Aciertos segmentados | Saldo (fixes/breaks) | Net nuevo/visto | IC95 bootstrap vendedor |
|---|---:|---|---:|---:|---:|---|---|
| 20261109 | 12.357 | .400 / .400 / .550 | 11.741 | 11.744 | +3 (47/44) | 0/+3 | [-17,+23] |
| 20261116 | 13.553 | .475 / .455 / .555 | 12.902 | 12.908 | +6 (37/31) | +2/+4 | [-9,+22] |

Las cifras de aciertos se deducen de las accuracies reportadas por el script; comparar los CSV pareados para cualquier auditoría fina. La mejora es pequeña e incierta en ambos cortes; no se promueve a submission. Los cortes confirm ya se habían utilizado para desarrollo y se solapan, de modo que no son dos confirmaciones independientes. El extractor comparte agregados no supervisados calculados antes del split externo. No se usaron scores de Kaggle como etiquetas.

Hashes confirm-ID: 20261109 `997893645e14d1c09c8ff0e3ceb3776f13688e21901888bf3196def555df3450`; 20261116 `6983106dd9194284298816b6b6362f8694a76655bca7f11e189deb0e1f7898c1`. Hashes pares: `ea45d37a40612040a30e9d3ff75d6d28043d8e4e0cf30021f4a8fd7d338d9aa3`, `e5f29c8b59edb23a5bfc0b77eb5ae788f4fbf23ea4dfeaae363658108167349b`.

Conclusión operativa: priorizar modelos de salto amplio. Segmentar el umbral puede entrar en una comparación futura, con tune congelado y confirm nuevo, pero estos números no justifican gastar un cupo público.
