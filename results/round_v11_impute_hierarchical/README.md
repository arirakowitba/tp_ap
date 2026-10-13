# Ronda v11: faltantes estructurales y encoding jerárquico

## Protocolo

Se reutilizaron los tres folds outer `StratifiedGroupKFold` ya congelados (seed 2026) sobre las 56.000 filas del holdout de selección. Cada fold valida 18.666–18.667 filas, entrena con 37.333–37.334 y tiene cero `parent_item_id` compartidos. Se comparó contra el OOF baseline almacenado, sin elegir umbrales ni pesos sobre validación.

El paquete candidato añade 39 pares de señales: imputación por mediana de `category_id` estimada solo en el train de cada fold, más un indicador de ausencia para cada feature numérica con observaciones; además sustituye el rate de `seller_id × category_id` por un rate OOF suavizado hacia el prior de categoría. El prior de categoría usa suavizado 20 y el rate vendedor-categoría 10; ambos se recalculan dentro de los folds internos agrupados por familia.

## Resultados

| Variante | Fold 0 accuracy | Fold 1 accuracy |
|---|---:|---:|
| Baseline OOF | 94,0108% | 93,9787% |
| Imputación sola | 94,0430% | 93,8876% |
| Encoding jerárquico solo | 94,0483% | 93,8876% |
| Imputación + encoding | 93,9679% | 93,8662% |

Las mejoras de fold 0 no se replican en fold 1; no se extienden las variantes aisladas al fold 2.

En los tres folds, el paquete combinado obtuvo **93,8429%** (52.552/56.000 aciertos), frente a **93,9143%** (52.592/56.000) del baseline. El blend fijo 10% quedó en **93,9054%** (52.587/56.000). Su log-loss sí baja levemente (0,158546 vs 0,158692), sin aumentar accuracy.

**Decisión:** no promover estas features ni reemplazar la submission. El resultado sugiere que las imputaciones estructurales y el prior de categoría alteran probabilidades, pero no mejoran aciertos de forma estable.

Reproducción:

```powershell
python src/round_v11_impute_hierarchical.py --folds 0 1 2
python src/round_v11_summarize.py
python src/round_v11_ablation.py --fold 0 --variants impute hierarchical
```
