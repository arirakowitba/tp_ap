# Ronda 37 — precio relativo a categoría y límite del proxy de la 33

## Hipótesis estructural

El precio bruto no expresa si un aviso es barato o caro dentro de su mercado.
Se añadieron al HistGB de ronda 30 cuatro variables sin etiquetas: desviación
del log-precio respecto de la mediana y frecuencia de la celda
`categoría × moneda`, y lo mismo para `categoría × moneda × tipo de publicación`.
Las medianas e IQR se estiman solo en fit. Las celdas con menos de 15 avisos
usan el centro global, sin tomar etiquetas ni estadísticas del holdout.
Los predictores OOF de texto original y todo el resto del modelo se mantuvieron.

| Corte por vendedor | Filas | HistGB correctos | Precio contextual correctos | Arreglos | Roturas | Neto |
|---|---:|---:|---:|---:|---:|---:|
| 883 | 14.335 | 13.314 | 13.306 | 58 | 66 | −8 |
| 2026 | 13.493 | 12.390 | 12.378 | 53 | 65 | −12 |

La celda categoría × moneda tiene al menos 15 ejemplos del fit para
7.357/14.335 y 7.366/13.493 avisos. Incluso en ese segmento, el saldo es
−6 y −1; en celdas escasas, −2 y −11. **No entrenar full70k ni proponer
submission** a partir de este feature set.

## Límite del proxy de la submission 33

Se compararon las decisiones de test de modelos completos ya guardados contra
el CSV exacto de la 33; esto mide semejanza de predicciones, no accuracy.
El CatBoost `text_deep_cat_oof_fresh/test.npz` a 0,5 es el más cercano entre
los modelos completos examinados: difiere en 558/30.000 filas (239 de
vendedores vistos y 319 de nuevos). El HistGB
`text_deep_oof/test.npz` a 0,5 difiere en 582 (241 vistos, 341 nuevos).
La propia submission 31 difiere de la 33 en solo 31 decisiones, pero no se
dispone de una función validada que reproduzca sus 30.000 decisiones sobre
train retenido. Por eso ni el modelo más cercano ni la 31 deben tratarse
como una reproducción exacta de la 33 al comparar arreglos y roturas.

Esta limitación ayuda a interpretar el fracaso de 37–39: una ganancia contra
HistGB no garantiza ganancia contra la 33. No prueba la causa del fracaso ni
revela etiquetas de test. Antes de seleccionar otra candidata amplia, hace
falta reconstruir o entrenar un proxy de la **política completa** de la 33 y
probarlo con etiquetas reales en segmentos vistos/nuevos. Ajustar una función
para copiar el CSV de test por sí sola no valida su accuracy.

## Reproducción

Con Python 3.12, scikit-learn 1.8.0, numpy 2.3.5, pandas 2.2.3 y scipy
1.17.0, desde la raíz del checkpoint:

```
python work20/round37_category_price_context.py 883
python work20/round37_category_price_context.py 2026
```

El checkpoint incluye datos originales, `docs` del Sheet original, código,
predicciones pareadas, CSV 31/33 y probabilidades de los dos modelos usados
en la comparación de proxy. Ningún score público se convirtió en etiqueta.
