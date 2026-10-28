# Ronda 49: compuerta en desacuerdos y estabilidad de test

Se auditó una compuerta logística previamente entrenada para elegir entre HistGB y FastText en avisos donde las dos bases discrepan. La regla de compuerta >.65 estaba fijada. En tres confirm seller-group obtuvo fixes/breaks/net frente a HistGB de 16/7/+9 en 23 cambios, 30/13/+17 en 43 y 23/9/+14 en 32. Los bootstrap por vendedor dieron IC 95% de neto [0,18], [5,30] y [3,25]. Son vendedores ausentes de fit y las etiquetas provienen solo de train.

Por dirección, `used→new` aportó +7/7, +13/33 y +5/7; `new→used`, +2/16, +4/10 y +9/25. En test hay 176 desacuerdos de vendedores nuevos examinados. Los tres ajustes de la compuerta eligieron 13, 36 y 13 casos, respectivamente; la intersección de los tres fue **cero**. La unión fue 53 y 44 casos fueron elegidos por un solo ajuste. Hay 9 con dos votos. Esto impide promover una lista robusta de flips desde la compuerta.

La comparación de confirm es contra HistGB, no contra la política exacta de submission 41. La 41 incluye cambios manuales de 33 que no tienen regla reproducible conocida en train y seis cambios específicos de test. No se identificaron etiquetas por resultados públicos ni se generó submission. El checkpoint conserva los pares, el código y los hashes.
