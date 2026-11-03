# Seller LOO como variable de la base fuerte — resultado 2026-11-03

Plan previo: `results/seller_loo_feature_20261103_plan.md`. Split seller testlike seed **20261129**, fit/tune/confirm 44.959/12.740/12.301. ZIP original 70k SHA-256 `3e0b032195766e681a411a1d1ea06fe906940dd025a7317adfc5302791bbe886`. La base y el candidato usan la misma matriz word+char OOF, variables originales, seller_id categórico y CatBoost depth6 lr .07 l2=6. El candidato agrega seller LOO rate y número de antecedentes: cada fila de fit excluye su etiqueta propia; tune/confirm reciben estadísticas sólo de fit con suavizado 10. Código `scripts/ablate_seller_loo_feature_20261103.py`. Pares/labels privados en Library `libfile_1bf657e1b5d08191aa432b8f6b62fa2e`; hash del CSV pareado `0521481e2a0746a3173d4acded0cd85ac5288b18b00fe8a3280ebfba2db2eb33`.

| Modelo | Iteración elegida en tune | Umbral tune | Accuracy confirm | Logloss confirm |
|---|---:|---:|---:|---:|
| Base fuerte | 895 | .455 | **.953337** | **.122897** |
| + seller LOO | 290 | .44 | .949841 | .132009 |

El candidato perdió **43 aciertos** (81 fixes, 124 breaks); a umbral común .5 también perdió 43. Por segmento: vendedor new +2, seen −45. IC 95% bootstrap por vendedor **[-81,−7]**. La parada temprana en 290 frente a 895 y el logloss peor son coherentes con una señal de entrenamiento que induce sobreajuste o mala traslación; no demuestran una causa única. El historial vendedor directo ya había fallado como regla de override contra bases fuertes en round141.

**Decisión:** descartar esta variable y no repetir el mismo diseño en más seeds, ni generar refit o CSV. La base D1+D2 .94652 sigue siendo el mejor score público demostrado. No se consultó Kaggle.
