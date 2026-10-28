# Ronda 34 — interacción categoría–lenguaje

## Hipótesis

El mismo término del aviso puede señalar condición distinta según la categoría.
Se entrenó una regresión logística con TF-IDF global de palabras y bigramas
del texto original, más interacciones categoría × término para categorías con
al menos 25 avisos de entrenamiento. Las interacciones comparten regularización
con los términos globales. El vocabulario y el mapa de categorías se ajustan
solo en train. `docs` proviene del Sheet original de 100.000 filas, no de la
reconstrucción de ronda 9.

## Protocolo y resultado

Se reutilizaron los cortes seller-group 883 y 2026 y las predicciones HistGB
pareadas de ronda 30. Ambos holdouts contienen solo vendedores ausentes del
entrenamiento. En el primer corte se compararon el modelo aislado y mezclas
predefinidas al 10% y 25%; el segundo verifica la de 10%. Umbrales: HistGB
0,475; lexical 0,5; mezcla 10% 0,4775; mezcla 25% 0,48125.

| Corte | Filas | HistGB | Lexical | Mezcla 10% | Arreglos/roturas 10% | Neto |
|---|---:|---:|---:|---:|---:|---:|
| 883 | 14.335 | 0,928776 | 0,839344 | 0,928985 | 16/13 | +3 |
| 2026 | 13.493 | 0,918254 | 0,835322 | 0,917587 | 15/24 | −9 |

Los cambios de mezcla están íntegramente en `abs(p_HistGB-0,475)<0,1`:
+3 de 446 y −9 de 431. En categorías vistas por el modelo de interacción,
el saldo es +4/−2; en categorías escasas o nuevas, −1/−7. Los intervalos
bootstrap 95% del neto, agrupando por vendedor, son [−8,+14] y [−21,+4].
No se combinan los dos intervalos como evidencia independiente porque los
cortes reutilizan filas de train y pueden solapar holdouts.

**Decisión:** no promover el modelo ni preparar una submission. El resultado
apoya buscar señal nueva cerca del umbral, pero esta representación léxica
por categoría no aportó una mejora estable. Ningún score público se utilizó
como etiqueta individual.

## Reproducción

Con Python 3.12, numpy 2.3.5, pandas 2.2.3, scipy 1.17.0 y scikit-learn 1.8.0,
ubicar `data.zip` en `recovered/Analisis rpedictivo/`, y los demás archivos en
`work20/`. Ejecutar:

```
python work20/round34_category_lexical.py 883
python work20/round34_category_lexical.py 2026
python work20/round34_paired_audit.py
```

`data.zip` y `original_docs.pkl` están incluidos en el checkpoint junto con los
vectores de referencia HistGB, las salidas pareadas, el código y el manifiesto
SHA256. No se entrenó un modelo full70k porque el holdout no respaldó su uso.
