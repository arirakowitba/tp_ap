# ExtraTrees con TE limpio y texto OOF — resultado (2026-11-03)

Se evaluó una arquitectura estructuralmente distinta a CatBoost/LightGBM: ExtraTrees sobre 25 señales numéricas y de patrones, frecuencias fit-only, siete target encodings OOF con prior fold-específico y una probabilidad textual word+char generada con GroupKFold por vendedor. Sólo se usaron las 70k etiquetas reales; vocabulario implícito, transformaciones supervisadas y threshold permanecieron dentro de fit/tune.

Corte seller-group predeclarado 20261143: 44.981 fit, 12.548 tune y 12.471 confirm. El threshold .480 se eligió exclusivamente en tune.

| Modelo | Confirm accuracy | Logloss |
| --- | ---: | ---: |
| ExtraTrees limpio | .898805 | .248980 |
| Comparador fuerte pareado existente | .932724 | — |

ExtraTrees perdió 423 aciertos: 229 fixes y 652 breaks sobre 881 discrepancias. Bootstrap por vendedor (3.000 réplicas): **[-576, -294]**. La distancia es demasiado grande para justificar réplica o blend. La línea queda cerrada.

Código: `scripts/ablate_extratrees_clean_te_20261103.py`. Métricas: `results/extratrees_clean_te/report_20261143.json`. Pares privados SHA-256 `91e34d2ab4d1517e2142afb9c7807965c178df1c9c0f4ce604504f0a6c9c58dd`. Script SHA-256 `695e5f67d2c9a63c361e0627ea404c3aadae38d8e0fb7e07a9b35100255cf3a7`. No se generó CSV de test ni se consumió Kaggle.
