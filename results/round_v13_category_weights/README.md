# Ronda v13: ponderación por distribución de categorías

La auditoría v12 mostró una diferencia de categorías entre train y test, así que se probó covariate weighting. Sin etiquetas de test, se calculó por fold el peso `P(category en test) / P(category en train)`, usando las 150 categorías más frecuentes del test, `OTHER` para el resto y pesos limitados a [0,25; 4]. La misma razón reponderó la métrica de validación para representar la mezcla de categorías del test. Ambos folds mantienen cero solapamiento de familias.

| Fold | Baseline accuracy | Ponderado accuracy | Baseline test-mix accuracy | Ponderado test-mix accuracy |
|---|---:|---:|---:|---:|
| 0 | 94,0108% | 94,0430% | 94,0212% | 94,0564% |
| 1 | 93,9787% | 93,7912% | 94,0204% | 93,8387% |

La ganancia de fold 0 no se transfiere: fold 1 pierde 35 aciertos y baja también la métrica ponderada. El tamaño efectivo de train se mantuvo cerca de 37.000; la ponderación fue moderada, pero no estable.

**Decisión:** no ponderar el fit final ni cambiar la submission. Mantener la reponderación de métricas como chequeo secundario para próximos holdouts, no como sustituto del OOF agrupado.

Reproducción:

```powershell
python src/round_v13_category_weights.py --fold 0
python src/round_v13_category_weights.py --fold 1
```
