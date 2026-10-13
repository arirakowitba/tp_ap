# Ronda v12: auditoría de representatividad del holdout

## Hallazgos sin etiquetas de test

El test tiene 30.000 filas y no comparte ningún `parent_item_id` no nulo con las 70.000 filas de train. Sí comparte vendedores: 71,62% de las filas test pertenecen a un vendedor visto en train (46,96% de vendedores únicos test).

La distribución de `category_id` cambia: distancia de variación total train–test **0,2190**. Las categorías más frecuentes del test son muy dispersas: top 150 explican 34,19% de las filas; el resto se agrupa como `OTHER` en la auditoría. Las otras variables categóricas medidas cambian poco (TV de `listing_type_id` 0,0045; `buying_mode` 0,0012). Las distribuciones de precio y cantidades también son cercanas (KS 0,0027–0,0044), y las tasas de ausencia se parecen: `warranty` difiere 0,15 puntos porcentuales y `original_price` 0,014 puntos.

Los folds agrupados actuales tienen cero familias cruzadas, pero sus distribuciones de categorías quedan a TV 0,277–0,280 del test. Por eso el accuracy OOF sin ponderar no refleja exactamente la mezcla de categorías de Kaggle.

## Ajuste de evaluación

Reponderé las 56.000 predicciones OOF seed 42 por `P(category en test) / P(category en OOF)`, con top 150 categorías y un grupo `OTHER`. La cobertura fue 100% y el tamaño efectivo de muestra 55.574.

| Métrica OOF | Original | Reponderada a categorías test |
|---|---:|---:|
| Accuracy | 93,9143% | 93,9314% |
| Log-loss | 0,158692 | 0,158332 |

La corrección cambia poco el accuracy. Es una estimación condicionada a que la tasa de error dentro de cada grupo de categoría se mantenga; no revela ni estima directamente las etiquetas ocultas. Conviene reportar ambos valores al comparar nuevos modelos.

Reproducción:

```powershell
python src/round_v12_holdout_diagnostics.py
python src/round_v12_holdout_weighted_summary.py
```
