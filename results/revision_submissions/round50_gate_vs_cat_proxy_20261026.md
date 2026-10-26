# Ronda 50: compuerta contra segunda base supervisada

Se ajustó CatBoost de segunda etapa sobre el mismo fit de cada uno de tres cortes seller-group usados en la auditoría de compuerta. Entrada: 70.000 etiquetas de train disponibles, textos originales 100.000, maestros word/char con predicciones OOF agrupadas por vendedor y variables estructuradas. Se conservó intacta la compuerta previamente fijada `p_gate>.65` y el umbral CatBoost .5. La comparación se hizo solo en los desacuerdos HistGB/FastText de confirm, con etiquetas reales de train.

| Corte | Desacuerdos | Correctas CatBoost | Correctas compuerta | Neto compuerta − CatBoost | IC 95% por vendedor |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 254 | 140 | 130 | −10 | [−28,+9] |
| 2 | 293 | 161 | 144 | −17 | [−39,+5] |
| 3 | 273 | 140 | 128 | −12 | [−32,+8] |

La pérdida ocurrió en los tres cortes, aunque cada intervalo individual incluye cero y los cortes reutilizan filas de train. La compuerta había ganado +9,+17,+14 frente a HistGB en estos mismos confirm: el efecto depende de la base elegida. Por dirección, los netos frente a CatBoost fueron `used→new` −10,−10,−8 y `new→used` 0,−7,−4. Esto, junto con cero coincidencias entre los tres fits de la compuerta para test, impide promover sus flips.

CatBoost reentrenado en este protocolo es un **proxy de arquitectura**, no la política exacta de submission 41. La 41 no tiene regla trasladable a train para sus cambios manuales. No se usaron etiquetas de test ni scores públicos en el ajuste; no se preparó submission. Los tres NPZ conservan filas, etiquetas, vendedores y predicciones pareadas. Reproducir con `ROUND50_TAG=first|second|third PYTHONPATH=work20/deps python3 work20/round50_gate_vs_cat_proxy.py` tras instalar CatBoost 1.2.10.
