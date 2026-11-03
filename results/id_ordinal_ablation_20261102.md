# Item ID ordinal contra baseline histórico fuerte — 2 de noviembre de 2026

Se probó la señal `item_id_billions`, derivada exclusivamente del ID original de cada publicación, sobre el CatBoost histórico con texto TF-IDF OOF y `seller_id` categórico. El experimento usa las 70.000 etiquetas originales para construir dos cortes seller-group testlike y entrena sólo el candidato; la referencia está formada por predicciones previas del mismo corte unidas y verificadas por `item_id`, `seller_id`, `y_true` y hash de IDs. Umbral e iteraciones se eligen en tune, antes de confirm. El ZIP train tiene SHA-256 `3e0b032195766e681a411a1d1ea06fe906940dd025a7317adfc5302791bbe886`.

| Corte | Confirm | Baseline accuracy | +ID accuracy | Saldo | Fixes / breaks | Log loss baseline → +ID | IC95% bootstrap vendedor | Saldo a umbral 0,50 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 20261109 | 12.357 | 0,95072 | 0,95096 | +3 | 64 / 61 | 0,13764 → 0,13736 | [−21,+28] | −8 |
| 20261116 | 13.553 | 0,95197 | 0,95108 | −12 | 56 / 68 | 0,12619 → 0,12778 | [−34,+10] | −8 |

En el primer corte, nuevos +3 y vistos 0; en el segundo, nuevos −6 y vistos −6. Los cortes reutilizan confirm de desarrollo y pueden solaparse, por lo que no se suman como réplicas independientes. El extractor compartido construye agregados **sin etiquetas** sobre las 70k filas antes de dividir fit/tune/confirm; esta comparación es pareada y train-only, pero no totalmente inductiva respecto de la representación. No hay evidencia estable para promover ID ordinal o refitarlo para test.

Código: `scripts/ablate_id_ordinal_vs_historical_text.py`. Pares por ID y reportes JSON completos se guardaron en el checkpoint privado. El entrenamiento de AG del Ciclo 1 es otra línea de texto estructurado; esta ablación no la duplica. AG-0039 informa +25 frente a un baseline char-only, pero su candidato todavía requiere comparación por ID con la referencia word+char OOF+seller de 11.748/12.357 y log loss 0,13764 del mismo corte.
