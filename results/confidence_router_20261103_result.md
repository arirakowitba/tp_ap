# Router por confianza: confirm fresco — 2026-11-03

Plan predeclarado: `results/confidence_router_20261103_plan.md`. Seed seller testlike **20261128**, fit/tune/confirm 44.691/13.038/12.271. ZIP original 70k SHA `3e0b032195766e681a411a1d1ea06fe906940dd025a7317adfc5302791bbe886`. Entrenamiento bag3 tres semillas 20261128/20262125/20263122, iteraciones elegidas en tune 893/883/846; LightGBM iteración 785. Texto word+char OOF y TE supervisada desde fit; ambos modelos en idéntico corte. Código `scripts/ablate_catboost_bag3_20261103.py` y `scripts/ablate_lightgbm_vs_bag3_20261103.py`. Pares/labels privados y hashes en Library `libfile_31cab2c9e0848191bdec0c28a9cf54b3`.

Regla congelada antes de confirm: usar LightGBM sólo si discrepa de bag3, `|p_bag3-.5|<=.1` y `|p_lgb-.5|>=.2`; en los demás casos usar bag3. Umbral .5. En el confirm nuevo:

| Modelo | Accuracy .5 | Logloss | Neto vs bag3 |
|---|---:|---:|---:|
| Histórico single | .954038 | .125779 | +4 |
| Bag3 | .953712 | **.124604** | control |
| LightGBM | .948089 | .137397 | -69 |
| Mezcla .5 universal | .952897 | .123966 | -10 |
| Router por confianza | .953875 | .125718 | **+2** |

El router cambió 56 decisiones de bag3: **29 fixes, 27 breaks**, neto seen/new −2/+4, IC bootstrap vendedor **[-14,+19]**. La ganancia en cuatro cortes de desarrollo (+10/+1/+9/+6), elegida tras examinar varias bandas, no se tradujo en una mejora material en el corte fresco. Además, bag3 perdió 4 aciertos contra su primer modelo en este corte, con IC [-19,+10], aunque bajó logloss .125779→.124604.

**Decisión:** no promover router ni bag3 a submission pública. Los tests de recuperación textual también resultaron débiles como regla de override: coincidencia exacta título+categoría con al menos dos ejemplos de fit casi nunca contradijo a bag3; similitud de título dentro de categoría produjo más errores nuevos que fixes en los cortes explorados. Ninguna etiqueta de test se usó. Mejor público comprobado D1+D2 .94652; no se gastó Kaggle.
