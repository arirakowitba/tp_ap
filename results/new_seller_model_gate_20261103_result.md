# Corte fresco de mezcla sólo para vendedor nuevo — 2026-11-03

Plan congelado antes de abrir confirm: `results/new_seller_model_gate_20261103_plan.md`. Seed seller testlike **20261127**, fit/tune/confirm 45.078/12.335/12.587. ZIP original 70k SHA `3e0b032195766e681a411a1d1ea06fe906940dd025a7317adfc5302791bbe886`. Regla: vendedor visto en fit conserva bag3; vendedor nuevo recibe media fija .5 bag3 + .5 LightGBM; umbral común .5. Iteraciones elegidas sólo en tune (bag3 843/897/848; LightGBM 557). Código: `scripts/ablate_catboost_bag3_20261103.py` y `scripts/ablate_lightgbm_vs_bag3_20261103.py`. Pares, etiquetas y reportes privados en Library `libfile_4b12bfa0deb08191ade439a83d40eff5`.

| Brazo | Accuracy confirm .5 | Logloss confirm | Neto vs bag3 |
|---|---:|---:|---:|
| Histórico single | .950822 | .129187 | 0 |
| Bag3 | .950822 | .128703 | control |
| LightGBM | .945261 | .141575 | -70 |
| Mezcla .5 para todos | .950346 | .128696 | -6 |
| Mezcla .5 sólo seller new | .950822 | **.127371** | **0** |

La regla new-only hizo 31 correcciones y 31 errores nuevos; todos en 3.587 vendedores nuevos de confirm, saldo 0, IC 95% bootstrap vendedor [-21,+19]. El bag3 con umbrales elegidos en tune ganó +4 contra single en este corte, pero a umbral común .5 empató; su IC por vendedor para +4 es [-15,+23]. El efecto de bagging sigue siendo incierto pese a los tres cortes anteriores positivos.

**Decisión:** no promover la mezcla segmentada ni generar CSV. La mejora de logloss sin aciertos no alcanza para el objetivo de score público. Los netos new-seller +2/+10/+10 de los tres confirms de desarrollo no se reprodujeron en el corte fresco. Reasignar esfuerzo a señales o arquitecturas de alcance mayor. D1+D2 .94652 permanece mejor público comprobado. No hubo consultas ni uploads a Kaggle.
